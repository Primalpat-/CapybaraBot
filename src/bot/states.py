"""State handlers for each bot state."""

import io
import logging
import random
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageStat

from src.adb.capture import ScreenCapture
from src.adb.input import ADBInput
from src.bot.actions import BotActions
from src.bot.calibration import (
    CoordinateCalibrator,
    ELEMENT_DESCRIPTIONS,
    SCREEN_ELEMENTS,
    NUM_MONUMENT_SLOTS,
)
from src.bot.state_machine import BotState, BotContext
from src.utils.timing import wait
from src.vision.cache import VisionCache
from src.vision.client import VisionClient
from src.vision.element_detector import ElementDetector
from src.vision.minimap_detector import find_minimap_squares, save_detection_debug
from src.vision.parser import (
    parse_screen_identification,
    parse_minimap_colors,
    parse_monument_info,
    parse_navigation_check,
    parse_battle_check,
    parse_post_battle,
    parse_calibration_result,
    parse_timer_seconds,
)
from src.vision.prompts import get_prompt

logger = logging.getLogger(__name__)

_DIAG_DIR = Path(__file__).resolve().parents[2] / "screenshots" / "calibration"


class StateHandlers:
    """All bot state handlers. Each returns the next BotState."""

    def __init__(
        self,
        capture: ScreenCapture,
        adb_input: ADBInput,
        vision: VisionClient,
        cache: VisionCache,
        actions: BotActions,
        config: dict,
        calibrator: CoordinateCalibrator,
        element_detector: ElementDetector | None = None,
    ):
        self.capture = capture
        self.input = adb_input
        self.vision = vision
        self.cache = cache
        self.actions = actions
        self.config = config
        self.calibrator = calibrator
        self.element_detector = element_detector
        self._visited_slots: set[int] = set()
        self._screen_w = config.get("screen", {}).get("width", 1080)
        self._screen_h = config.get("screen", {}).get("height", 1920)
        self._minimap_open_attempts = 0
        self._retries_without_progress = 0  # consecutive failures across states
        self._hibernation_seconds: int | None = None  # countdown from last detection
        self._unbeatable_players: set[str] = set()  # players we've lost to

    async def _tap_ok_button(self, png: bytes, ctx: BotContext, config: dict) -> None:
        """Calibrate and tap the OK button on the battle results screen.

        After tapping, waits past any loading/black screen locally (no Vision).
        """
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)

        self._calibrate_for_screen(png, "battle_result", ctx)
        x, y = self.calibrator.get_pixel("ok_button")
        ctx.log_action(f"Tapping OK button at ({x}, {y})")
        for tap_x, tap_y in [(x, y), (self._screen_w // 2, int(self._screen_h * 0.85))]:
            await self.input.tap(tap_x, tap_y)
            await wait(1.0, jitter, "ok button tap")
            await self.input.tap(tap_x, tap_y)
            await wait(1.0, jitter, "ok button retry")

        # Wait past the post-OK loading screen locally — no Vision needed
        await self._wait_past_loading_local(ctx, config, "post ok")

    async def _dismiss_occupy_prompt(self, png: bytes, ctx: BotContext, config: dict) -> None:
        """Tap Cancel on the 'continue to occupy?' Tips popup.

        This popup appears after clearing a monument when the player already
        occupies another one.  We always cancel — never swap.
        """
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)

        self._calibrate_for_screen(png, "occupy_prompt", ctx)
        x, y = self.calibrator.get_pixel("occupy_cancel_button")
        ctx.log_action(f"Occupy prompt — tapping Cancel at ({x}, {y})")
        await self.input.tap(x, y)
        await wait(timing.get("screen_transition", 2.0), jitter, "occupy cancel")

    # Offsets for tap-and-verify spiral: center, cardinal, diagonal
    _TAP_OFFSETS = [
        (0, 0),
        (0, -30), (30, 0), (0, 30), (-30, 0),       # cardinal ±30px
        (-30, -30), (30, -30), (30, 30), (-30, 30),   # diagonal ±30px
        (0, -60), (60, 0), (0, 60), (-60, 0),         # cardinal ±60px
    ]

    async def _tap_and_verify(
        self,
        element_name: str,
        screen_type: str,
        expected_screens: list[str],
        ctx: BotContext,
        config: dict,
        png: bytes | None = None,
    ) -> tuple[bytes, object, bool]:
        """Tap a calibrated element, verify screen changed, retry with offsets.

        1. Calibrates on current screenshot if needed.
        2. Taps the calibrated point.
        3. Waits, captures, identifies screen.
        4. If screen matches *expected_screens*, saves the working coords and
           returns (png, screen, True).
        5. Otherwise tries offsets around the original point.  When an offset
           works, updates calibration with the corrected coordinates and
           persists them so we never miss again.
        6. Returns (png, screen, False) if all offsets fail.
        """
        timing = config.get("timing", {})
        tap_wait = timing.get("screen_transition", 2.0)
        jitter = timing.get("jitter_factor", 0.3)

        if png is not None:
            self._calibrate_for_screen(png, screen_type, ctx)

        base_x, base_y = self.calibrator.get_pixel(element_name)

        for i, (dx, dy) in enumerate(self._TAP_OFFSETS):
            tx = max(0, min(self._screen_w, base_x + dx))
            ty = max(0, min(self._screen_h, base_y + dy))

            if i == 0:
                ctx.log_action(f"Tapping {element_name} at ({tx}, {ty})")
            else:
                ctx.log_action(
                    f"Tap didn't work — retrying {element_name} with "
                    f"offset ({dx:+d}, {dy:+d}) → ({tx}, {ty})"
                )

            await self.input.tap(tx, ty)
            await wait(tap_wait, jitter, f"{element_name} verify")

            # Check what screen we're on now
            check_png = await self.capture.capture()
            ctx.last_screenshot = check_png

            # Quick local loading check first
            if self._is_loading_screen(check_png):
                check_png = await self._wait_past_loading_local(
                    ctx, config, f"{element_name} post-tap"
                )

            text = self._call_vision(check_png, "identify_screen")
            ctx.stats.api_calls += 1
            screen = parse_screen_identification(text)

            if screen.screen_type in expected_screens:
                if i > 0:
                    # The offset worked — update calibration with corrected coords
                    new_x_pct = tx / self._screen_w * 100
                    new_y_pct = ty / self._screen_h * 100
                    ctx.log_action(
                        f"Offset tap succeeded! Updating {element_name} calibration: "
                        f"({new_x_pct:.1f}%, {new_y_pct:.1f}%) → ({tx}, {ty})"
                    )
                    self.calibrator.store(element_name, new_x_pct, new_y_pct, 1.0)
                    self.calibrator.save()
                return check_png, screen, True

        ctx.log_action(f"All tap offsets failed for {element_name}")
        return check_png, screen, False

    def _enter_hibernation(self, screen, ctx: BotContext) -> BotState:
        """Handle hibernation detection — parse timer and go idle."""
        secs = parse_timer_seconds(screen.timer)
        self._hibernation_seconds = secs
        if secs is not None:
            m, s = divmod(secs, 60)
            h, m = divmod(m, 60)
            ctx.log_action(f"Hibernation active — {h}h{m:02d}m{s:02d}s remaining, sleeping until it ends")
        else:
            ctx.log_action("Hibernation active — could not read timer, will recheck in 60s")
        return BotState.IDLE

    def _save_calibration_diagnostic(self, png_bytes: bytes, elements: list, screen_type: str) -> None:
        """Save a screenshot with crosshairs at calibrated positions for debugging."""
        try:
            _DIAG_DIR.mkdir(parents=True, exist_ok=True)
            image = Image.open(io.BytesIO(png_bytes))
            draw = ImageDraw.Draw(image)
            img_w, img_h = image.size
            colors = ["red", "lime", "cyan", "yellow", "magenta", "orange"]

            for i, el in enumerate(elements):
                color = colors[i % len(colors)]
                ix = int(el.x_percent / 100 * img_w)
                iy = int(el.y_percent / 100 * img_h)

                # Crosshair
                arm = max(20, img_w // 40)
                draw.line([(ix - arm, iy), (ix + arm, iy)], fill=color, width=3)
                draw.line([(ix, iy - arm), (ix, iy + arm)], fill=color, width=3)
                # Circle
                r = arm // 2
                draw.ellipse([(ix - r, iy - r), (ix + r, iy + r)], outline=color, width=2)
                # Label
                tap_x = int(el.x_percent / 100 * self._screen_w)
                tap_y = int(el.y_percent / 100 * self._screen_h)
                label = f"{el.name} ({el.x_percent:.1f}%,{el.y_percent:.1f}%) tap=({tap_x},{tap_y})"
                draw.text((ix + arm + 5, iy - 10), label, fill=color)

            path = _DIAG_DIR / f"calibration_{screen_type}.png"
            image.save(path)
            logger.info(f"Saved calibration diagnostic: {path}")
        except Exception as e:
            logger.warning(f"Could not save calibration diagnostic: {e}")

    def _call_vision(self, png_bytes: bytes, prompt_name: str) -> str:
        """Call vision API with caching and logging."""
        cached = self.cache.get(png_bytes, prompt_name)
        if cached is not None:
            logger.debug(f"Vision cache hit for {prompt_name}")
            return cached

        logger.info(f"Calling Vision API: {prompt_name}")
        system, prompt = get_prompt(prompt_name)
        response = self.vision.analyze_screenshot(png_bytes, prompt, system)
        self.cache.put(png_bytes, prompt_name, response.text)
        return response.text

    def _calibrate_for_screen(self, png_bytes: bytes, screen_type: str, ctx: BotContext) -> None:
        """Calibrate UI elements: try local OpenCV detection first, Vision API fallback."""
        needed = self.calibrator.needs_calibration(screen_type)
        if not needed:
            return

        # ── Step 1: Try local detection (free, fast) ──────────────
        locally_found: set[str] = set()
        if self.element_detector is not None:
            detections = self.element_detector.detect(png_bytes, screen_type)
            for det in detections:
                if det.name in needed and det.confidence >= 0.7:
                    self.calibrator.store(det.name, det.x_percent, det.y_percent, det.confidence)
                    locally_found.add(det.name)
                    logger.info(
                        f"  Local detection: {det.name} at ({det.x_percent:.1f}%, {det.y_percent:.1f}%) "
                        f"[{det.method}] conf={det.confidence:.2f}"
                    )

            if locally_found:
                ctx.log_action(f"Local detection found {len(locally_found)}/{len(needed)} for {screen_type}")

        # ── Step 2: Check what still needs calibration ────────────
        still_needed = [n for n in needed if n not in locally_found]
        if not still_needed:
            self.calibrator.save()
            return

        # ── Step 3: Vision API fallback for remaining elements ────
        descriptions = []
        for i, name in enumerate(still_needed, 1):
            desc = ELEMENT_DESCRIPTIONS.get(name, name)
            descriptions.append(f"{i}. **{name}**: {desc}")
        elements_description = "\n".join(descriptions)

        ctx.log_action(f"Vision API calibrating {len(still_needed)} elements for {screen_type}")
        logger.info(f"Vision API calibrating {len(still_needed)} elements for {screen_type}: {still_needed}")

        system, prompt_template = get_prompt("calibrate_elements")
        prompt = prompt_template.replace("{elements_description}", elements_description)

        logger.info(f"Calling Vision API: calibrate_elements ({screen_type})")
        response = self.vision.analyze_screenshot(png_bytes, prompt, system)
        ctx.stats.api_calls += 1

        result = parse_calibration_result(response.text)
        stored_count = len(locally_found)
        for el in result.elements:
            x_pct = el.x_percent
            y_pct = el.y_percent

            # Detect pixel values returned instead of percentages and convert
            if x_pct > 100 or y_pct > 100:
                logger.warning(
                    f"  Vision returned pixel coords for {el.name}: ({x_pct:.1f}, {y_pct:.1f}) "
                    f"— converting to percentages using image dimensions"
                )
                x_pct = x_pct / self._screen_w * 100
                y_pct = y_pct / self._screen_h * 100

            tap_x = int(x_pct / 100 * self._screen_w)
            tap_y = int(y_pct / 100 * self._screen_h)
            logger.info(
                f"  Vision returned: {el.name} at ({x_pct:.1f}%, {y_pct:.1f}%) "
                f"→ tap ({tap_x}, {tap_y})  confidence={el.confidence:.2f}"
            )
            if el.confidence > 0 and el.name in still_needed:
                self.calibrator.store(el.name, x_pct, y_pct, el.confidence)
                stored_count += 1

                # Auto-capture template for template-matchable elements
                if self.element_detector is not None and el.name in (
                    "star_trek_button", "alien_minefield_button"
                ):
                    self.element_detector.save_template(png_bytes, el.name, x_pct, y_pct)

        # Save annotated screenshot for debugging
        self._save_calibration_diagnostic(png_bytes, result.elements, screen_type)

        if stored_count > 0:
            self.calibrator.save()
            ctx.log_action(f"Calibrated {stored_count}/{len(needed)} elements")
        else:
            ctx.log_action(f"Could not calibrate any elements for {screen_type}")

    @staticmethod
    def _is_loading_screen(png_bytes: bytes, threshold: float = 18.0) -> bool:
        """Check if a screenshot is a loading/black screen using pixel brightness.

        Returns True if the average brightness is below *threshold* (0-255).
        This is a fast, free check — no Vision API call needed.
        """
        try:
            image = Image.open(io.BytesIO(png_bytes)).convert("L")  # grayscale
            mean_brightness = ImageStat.Stat(image).mean[0]
            return mean_brightness < threshold
        except Exception:
            return False

    async def _wait_past_loading_local(
        self, ctx: BotContext, config: dict, label: str = "screen"
    ) -> bytes:
        """Wait until the screen is no longer black, using pixel checks only.

        Returns the first non-loading PNG bytes.  Does NOT call the Vision API.
        """
        timing = config.get("timing", {})
        loading_wait = timing.get("loading_wait", 5.0)
        max_retries = int(timing.get("loading_max_retries", 6))
        jitter = timing.get("jitter_factor", 0.3)

        for attempt in range(max_retries + 1):
            png = await self.capture.capture()
            ctx.last_screenshot = png

            if not self._is_loading_screen(png):
                if attempt > 0:
                    ctx.log_action(f"Loading finished ({label}, {attempt} retries, no Vision)")
                return png

            ctx.log_action(
                f"Black screen ({label}), waiting {loading_wait:.0f}s "
                f"(attempt {attempt + 1}/{max_retries + 1}, no Vision)"
            )
            await wait(loading_wait, jitter, f"loading wait ({label})")

        ctx.log_action(f"Still black after {max_retries + 1} attempts — proceeding anyway")
        return png

    async def _wait_past_loading(
        self, ctx: BotContext, config: dict, label: str = "screen"
    ) -> tuple[bytes, object]:
        """Wait past black screens locally, then identify with Vision.

        Uses pixel brightness to skip loading screens for free, and only calls
        the Vision API once the screen has actual content.
        """
        png = await self._wait_past_loading_local(ctx, config, label)

        text = self._call_vision(png, "identify_screen")
        ctx.stats.api_calls += 1
        screen = parse_screen_identification(text)
        return png, screen

    async def handle_initializing(self, ctx: BotContext, config: dict) -> BotState:
        """Identify current screen and route to the correct state.

        The bot can be started on ANY screen and will pick up from the right
        step. This is the universal entry point.

        Screen routing table (update when adding new screen types):
          main_map         → OPENING_MINIMAP    Normal flow, open minimap to find targets
          minimap          → READING_MINIMAP    Read monument colors, pick a target
          monument_popup   → CHECKING_MONUMENT  Read defenders, decide to attack or skip
          battle_active    → SKIPPING_BATTLE    Tap skip, wait for result
          battle_result    → tap OK → POST_BATTLE  Dismiss Victory/Defeat, check monument
          hibernation      → IDLE               Sleep until hibernation timer expires
          logged_out       → RECONNECTING       Wait delay → Restart → Star Trek → Alien Minefield
          home_screen      → RECONNECTING       Star Trek → Alien Minefield
          mode_select      → RECONNECTING       Alien Minefield
          occupy_prompt    → tap Cancel → INITIALIZING  Never swap, re-identify screen
          loading/menu/unknown → OPENING_MINIMAP  Fallback — retry from minimap

        When adding a NEW screen type:
          1. Add it to the identify_screen prompt in config/prompts.yaml
          2. Add a routing entry below
          3. Add calibration elements if needed (src/bot/calibration.py)
          4. Add logged_out-style detection in other handlers if it can appear mid-flow
        """
        ctx.log_action("Initializing — identifying screen")
        png, screen = await self._wait_past_loading(ctx, config, "initializing")

        ctx.log_action(f"Screen: {screen.screen_type} (conf={screen.confidence:.2f})")

        # Calibrate elements visible on this screen type
        if screen.screen_type in SCREEN_ELEMENTS:
            self._calibrate_for_screen(png, screen.screen_type, ctx)
        elif screen.screen_type in ("main_map", "unknown"):
            self._calibrate_for_screen(png, "main_map", ctx)

        if screen.screen_type == "hibernation":
            return self._enter_hibernation(screen, ctx)
        elif screen.screen_type in ("logged_out", "home_screen", "mode_select"):
            ctx.log_action(f"Not in game ({screen.screen_type}) — entering reconnection flow")
            return BotState.RECONNECTING
        elif screen.screen_type == "minimap":
            return BotState.READING_MINIMAP
        elif screen.screen_type == "monument_popup":
            return BotState.CHECKING_MONUMENT
        elif screen.screen_type == "battle_active":
            return BotState.SKIPPING_BATTLE
        elif screen.screen_type == "battle_result":
            # Tap OK to dismiss the results screen before proceeding
            ctx.log_action("Battle result screen — tapping OK")
            await self._tap_ok_button(png, ctx, config)
            return BotState.POST_BATTLE
        elif screen.screen_type == "occupy_prompt":
            await self._dismiss_occupy_prompt(png, ctx, config)
            return BotState.INITIALIZING
        else:
            return BotState.OPENING_MINIMAP

    async def handle_opening_minimap(self, ctx: BotContext, config: dict) -> BotState:
        """Open the minimap overlay.

        Uses a single tap + local minimap detection instead of _tap_and_verify,
        because the minimap is an overlay — offset retry taps would interact
        destructively with the overlay (tapping monuments, closing it, etc.).
        """
        timing = config.get("timing", {})
        tap_wait = timing.get("screen_transition", 2.0)
        jitter = timing.get("jitter_factor", 0.3)

        self._minimap_open_attempts += 1
        self._retries_without_progress += 1

        # Too many consecutive failures — cool down
        max_retries = config.get("bot", {}).get("max_idle_retries", 8)
        if self._retries_without_progress > max_retries:
            ctx.log_action(
                f"Too many retries without progress ({self._retries_without_progress}) "
                f"— cooling down in idle"
            )
            self._minimap_open_attempts = 0
            self._retries_without_progress = 0
            return BotState.IDLE

        if self._minimap_open_attempts > 2:
            ctx.log_action(
                f"Minimap open attempt #{self._minimap_open_attempts} — retrying"
            )

        # Capture a screenshot to calibrate from (we're on the main map)
        png = await self.capture.capture()
        ctx.last_screenshot = png

        # Calibrate minimap_button position
        self._calibrate_for_screen(png, "main_map", ctx)

        if not self.calibrator.is_calibrated("minimap_button"):
            ctx.log_action("Could not find minimap_button — retrying")
            return BotState.OPENING_MINIMAP

        # Single tap — no offset loop (minimap is an overlay)
        bx, by = self.calibrator.get_pixel("minimap_button")
        ctx.log_action(f"Tapping minimap_button at ({bx}, {by})")
        await self.input.tap(bx, by)
        await wait(tap_wait, jitter, "minimap open")

        # Capture after tap and wait past any loading
        check_png = await self.capture.capture()
        ctx.last_screenshot = check_png
        if self._is_loading_screen(check_png):
            check_png = await self._wait_past_loading_local(
                ctx, config, "minimap open"
            )

        # Verify locally: can we see the minimap's colored squares?
        detection = find_minimap_squares(check_png)
        if detection and len(detection.squares) >= 2:
            ctx.log_action(f"Minimap opened — detected {len(detection.squares)} squares locally")
            self._minimap_open_attempts = 0
            return BotState.READING_MINIMAP

        # Local detection didn't find squares — ask Vision what screen this is
        text = self._call_vision(check_png, "identify_screen")
        ctx.stats.api_calls += 1
        screen = parse_screen_identification(text)

        if screen.screen_type == "minimap":
            # Vision says it's the minimap even though local detection failed
            self._minimap_open_attempts = 0
            return BotState.READING_MINIMAP

        if screen.screen_type == "hibernation":
            return self._enter_hibernation(screen, ctx)

        if screen.screen_type == "logged_out":
            ctx.log_action("Logged out detected — entering reconnection flow")
            return BotState.RECONNECTING

        ctx.log_action(
            f"Minimap not detected after tap (screen={screen.screen_type}), retrying..."
        )
        return BotState.OPENING_MINIMAP

    async def handle_reading_minimap(self, ctx: BotContext, config: dict) -> BotState:
        """Read minimap using pixel color detection (no Vision API needed)."""
        ctx.log_action("Reading minimap")
        png = await self.capture.capture()
        ctx.last_screenshot = png

        # Use OpenCV color detection to find the 4 squares — fast, free, precise
        detection = find_minimap_squares(png)

        # Save debug image on first detection or failures
        debug_path = str(_DIAG_DIR / "minimap_detection.png")
        _DIAG_DIR.mkdir(parents=True, exist_ok=True)
        save_detection_debug(png, detection, debug_path)

        if detection is None or len(detection.squares) < 2:
            ctx.log_action("Could not detect minimap squares via color — falling back to Vision")
            # Fall back to Vision-based color check
            text = self._call_vision(png, "check_minimap_colors")
            ctx.stats.api_calls += 1
            colors = parse_minimap_colors(text)
            ctx.log_action(f"Vision minimap colors: {colors.slot_colors}")
            # With Vision fallback we still need calibrated positions
            if not self.calibrator.is_calibrated("monument_slot_1"):
                self._calibrate_for_screen(png, "minimap", ctx)
                if not self.calibrator.derive_minimap_slots():
                    ctx.log_action("Could not derive slot positions — retrying")
                    return BotState.READING_MINIMAP
            slot_colors = colors.slot_colors
            use_detection_coords = False
        else:
            slot_colors = detection.slot_colors
            use_detection_coords = True
            ctx.log_action(f"Minimap detected: {slot_colors}")

            # Store detected positions in calibrator for consistency
            for sq in detection.squares:
                x_pct = sq.center_x / detection.image_width * 100
                y_pct = sq.center_y / detection.image_height * 100
                self.calibrator.store(
                    f"monument_slot_{sq.slot}", x_pct, y_pct, 1.0
                )
            self.calibrator.save()

        # Find first red (enemy) slot we haven't visited yet
        target_slot = None
        for slot in range(1, NUM_MONUMENT_SLOTS + 1):
            color = slot_colors.get(slot, "unknown")
            if color == "red" and slot not in self._visited_slots:
                target_slot = slot
                break

        if target_slot is None:
            # Try any red slot (maybe revisit)
            for slot in range(1, NUM_MONUMENT_SLOTS + 1):
                if slot_colors.get(slot, "unknown") == "red":
                    target_slot = slot
                    break

        if target_slot is None:
            ctx.log_action("No red monuments remaining — going idle")
            return BotState.IDLE

        # Get tap coordinates
        if use_detection_coords and detection:
            sq = detection.get_square(target_slot)
            if sq:
                # Use pixel coordinates directly from detection.
                # Scale from screenshot space to input tap space (wm size).
                x = int(sq.center_x * self._screen_w / detection.image_width)
                y = int(sq.center_y * self._screen_h / detection.image_height)
            else:
                x, y = self.calibrator.get_pixel(f"monument_slot_{target_slot}")
        else:
            x, y = self.calibrator.get_pixel(f"monument_slot_{target_slot}")

        ctx.current_target = {
            "slot": target_slot,
            "slot_name": f"monument_slot_{target_slot}",
            "x": x,
            "y": y,
        }
        ctx.log_action(f"Target: slot {target_slot} (red) at ({x}, {y})")

        # Tap the monument icon on the minimap
        logger.info(f"Tapping monument slot {target_slot} at ({x}, {y})")
        await self.input.tap(x, y)

        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        await wait(timing.get("screen_transition", 2.0), jitter, "monument tap")

        # Verify the minimap closed — confirms the icon tap worked
        png = await self.capture.capture()
        ctx.last_screenshot = png

        text = self._call_vision(png, "identify_screen")
        ctx.stats.api_calls += 1
        screen = parse_screen_identification(text)

        if screen.screen_type == "minimap":
            ctx.log_action(
                f"Minimap still open after tapping slot {target_slot} — will retry"
            )
            return BotState.READING_MINIMAP

        ctx.log_action("Minimap closed — navigation started")
        self._retries_without_progress = 0
        return BotState.NAVIGATING

    async def handle_navigating(self, ctx: BotContext, config: dict) -> BotState:
        """Poll until we arrive at the monument."""
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)

        await wait(timing.get("navigation_poll_interval", 2.0), jitter, "nav poll")

        png = await self.capture.capture()
        ctx.last_screenshot = png

        text = self._call_vision(png, "check_navigation")
        ctx.stats.api_calls += 1
        nav = parse_navigation_check(text)

        if nav.monument_popup_visible:
            ctx.log_action("Arrived at monument (popup already open)")
            ctx.stats.monuments_visited += 1
            if ctx.current_target:
                self._visited_slots.add(ctx.current_target.get("slot", 0))
            return BotState.CHECKING_MONUMENT

        if nav.arrived:
            ctx.log_action("Arrived at monument — tapping to open popup")
            ctx.stats.monuments_visited += 1
            if ctx.current_target:
                self._visited_slots.add(ctx.current_target.get("slot", 0))
            return BotState.APPROACHING_MONUMENT

        ctx.log_action("Still navigating...")
        return BotState.NAVIGATING

    async def handle_approaching_monument(self, ctx: BotContext, config: dict) -> BotState:
        """Tap the monument in the game world to open its popup."""
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)

        png, screen = await self._wait_past_loading(ctx, config, "approaching monument")

        if screen.screen_type == "monument_popup":
            ctx.log_action("Monument popup detected")
            return BotState.CHECKING_MONUMENT

        if screen.screen_type == "hibernation":
            return self._enter_hibernation(screen, ctx)

        if screen.screen_type == "logged_out":
            ctx.log_action("Logged out detected — entering reconnection flow")
            return BotState.RECONNECTING

        # Calibrate world_monument position if needed (first arrival only)
        self._calibrate_for_screen(png, "arrived_at_monument", ctx)

        # Tap the calibrated world monument position
        x, y = self.calibrator.get_pixel("world_monument")
        ctx.log_action(f"Tapping world monument at ({x}, {y})")
        await self.input.tap(x, y)

        await wait(timing.get("monument_popup_wait", 1.5), jitter, "waiting for popup")

        # Check if popup opened
        png, screen = await self._wait_past_loading(ctx, config, "after monument tap")

        if screen.screen_type == "monument_popup":
            ctx.log_action("Monument popup opened")
            return BotState.CHECKING_MONUMENT

        if screen.screen_type == "hibernation":
            return self._enter_hibernation(screen, ctx)

        if screen.screen_type == "logged_out":
            ctx.log_action("Logged out detected — entering reconnection flow")
            return BotState.RECONNECTING

        ctx.log_action("Popup not detected, retrying tap...")
        return BotState.APPROACHING_MONUMENT

    async def handle_checking_monument(self, ctx: BotContext, config: dict) -> BotState:
        """Read monument popup and decide what to do."""
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        await wait(timing.get("monument_popup_wait", 1.5), jitter, "popup wait")

        png = await self.capture.capture()
        ctx.last_screenshot = png

        # Calibrate popup elements if needed
        self._calibrate_for_screen(png, "monument_popup", ctx)

        text = self._call_vision(png, "check_monument")
        ctx.stats.api_calls += 1
        info = parse_monument_info(text)
        ctx.monument_info = info

        ctx.log_action(
            f"Monument: {info.monument_name or 'unnamed'}, "
            f"ownership={info.ownership}, friendly={info.is_friendly}"
        )

        if info.is_friendly:
            ctx.log_action("Friendly monument — skipping")
            await self.actions.close_popup()
            return BotState.OPENING_MINIMAP

        if info.all_defenders_defeated:
            ctx.log_action("All defenders already defeated")
            if info.action_button.visible:
                await self.actions.tap_action_button()
            await self.actions.close_popup()
            return BotState.OPENING_MINIMAP

        if info.action_button.visible and info.action_button.action_type in ("attack", "unknown"):
            # Check if the next active defender is someone we can't beat
            for defender in info.defenders:
                if defender.status == "active" and defender.name:
                    if defender.name in self._unbeatable_players:
                        ctx.log_action(
                            f"Skipping — defender '{defender.name}' is unbeatable"
                        )
                        await self.actions.close_popup()
                        return BotState.OPENING_MINIMAP
            ctx.log_action(f"Attacking! (button: {info.action_button.text})")
            self._retries_without_progress = 0
            return BotState.ATTACKING

        ctx.log_action("No attack available — moving on")
        await self.actions.close_popup()
        return BotState.OPENING_MINIMAP

    async def handle_attacking(self, ctx: BotContext, config: dict) -> BotState:
        """Tap the attack button, then verify a battle actually started.

        The monument popup can be stale — if another player garrisoned since we
        opened it, tapping Attack flashes a brief error banner and stays on the
        popup.  When that happens we close the popup, reopen it to refresh, and
        re-check the monument.
        """
        await self.actions.tap_action_button()
        ctx.log_action("Attack initiated — waiting for battle to load")

        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        await wait(timing.get("screen_transition", 2.0), jitter, "battle start")

        # Wait past loading locally (free), then verify with Vision
        png, screen = await self._wait_past_loading(ctx, config, "attack verify")

        if screen.screen_type in ("battle_active", "battle_result"):
            ctx.stats.battles_fought += 1
            if screen.screen_type == "battle_result":
                # Battle already finished (fast battles)
                ctx.log_action("Battle already finished")
            return BotState.SKIPPING_BATTLE

        # Attack didn't go through — popup was stale
        ctx.log_action(
            f"Battle did not start (screen={screen.screen_type}) — "
            "popup was likely stale, refreshing"
        )
        await self.actions.close_popup()
        await wait(timing.get("monument_popup_wait", 1.5), jitter, "stale popup close")
        return BotState.REFRESHING_POPUP

    async def handle_skipping_battle(self, ctx: BotContext, config: dict) -> BotState:
        """Poll for battle end, tapping skip/speed-up."""
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)

        # Wait past any loading screen locally before calibrating
        png = await self._wait_past_loading_local(ctx, config, "battle start")

        # Calibrate skip button if needed
        if self.calibrator.needs_calibration("battle_active"):
            self._calibrate_for_screen(png, "battle_active", ctx)

        # Tap skip button (it's fine to tap even if not visible)
        await self.actions.tap_skip_battle()
        await wait(timing.get("battle_poll_interval", 3.0), jitter, "battle poll")

        png = await self.capture.capture()
        ctx.last_screenshot = png

        text = self._call_vision(png, "check_battle")
        ctx.stats.api_calls += 1
        battle = parse_battle_check(text)

        if battle.battle_state in ("victory", "defeat", "results_screen"):
            ctx.log_action(f"Battle ended: {battle.battle_state}")
            if battle.battle_state == "victory":
                ctx.stats.battles_won += 1

            await self._tap_ok_button(png, ctx, config)

            if battle.battle_state == "defeat":
                # Save screenshot of the defeat screen
                defeat_dir = Path(__file__).resolve().parents[2] / "screenshots" / "defeats"
                defeat_dir.mkdir(parents=True, exist_ok=True)
                defeat_path = defeat_dir / f"defeat_{int(time.time())}.png"
                defeat_path.write_bytes(png)
                logger.info(f"Saved defeat screenshot: {defeat_path}")

                opponent = battle.opponent_name.strip()
                if opponent:
                    self._unbeatable_players.add(opponent)
                    ctx.log_action(
                        f"DEFEAT by '{opponent}' — added to unbeatable list "
                        f"({len(self._unbeatable_players)} total: {self._unbeatable_players})"
                    )
                else:
                    ctx.log_action("DEFEAT — could not read opponent name")
                # Don't return to this monument — go find the next target
                await self.actions.close_popup()
                return BotState.OPENING_MINIMAP

            return BotState.POST_BATTLE

        if battle.battle_state == "unknown":
            # Might not be on a battle screen at all — check via screen identification
            text2 = self._call_vision(png, "identify_screen")
            ctx.stats.api_calls += 1
            screen = parse_screen_identification(text2)
            if screen.screen_type not in ("battle_active", "battle_result"):
                ctx.log_action(
                    f"Not on a battle screen (screen={screen.screen_type}) — "
                    "rerouting through initializing"
                )
                return BotState.INITIALIZING

        ctx.log_action("Battle still in progress...")
        return BotState.SKIPPING_BATTLE

    async def handle_post_battle(self, ctx: BotContext, config: dict) -> BotState:
        """Check monument ownership after a battle victory.

        After tapping OK on the battle results screen, the monument popup should
        reappear.  We read it with check_monument and look at the Monument
        Ownership section (blue "Star Spirit" = ours, red = enemy).
        """
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        await wait(timing.get("screen_transition", 2.0), jitter, "post battle wait")

        # Wait past any loading screen
        png, screen = await self._wait_past_loading(ctx, config, "post battle")

        # If we ended up on a completely different screen, reroute
        if screen.screen_type in ("logged_out", "home_screen", "mode_select"):
            ctx.log_action(f"Post-battle landed on {screen.screen_type} — rerouting")
            return BotState.RECONNECTING
        if screen.screen_type == "hibernation":
            return self._enter_hibernation(screen, ctx)

        # Occupy swap prompt — always cancel, never swap monuments
        if screen.screen_type == "occupy_prompt":
            await self._dismiss_occupy_prompt(png, ctx, config)
            ctx.stats.monuments_captured += 1
            ctx.log_action("Monument captured! (dismissed occupy swap prompt)")
            return BotState.INITIALIZING

        # If the popup isn't visible, try tapping the monument to reopen it
        if screen.screen_type != "monument_popup":
            ctx.log_action(f"Post-battle screen is {screen.screen_type} — tapping monument to reopen popup")
            x, y = self.calibrator.get_pixel("world_monument")
            if x > 0 and y > 0:
                await self.input.tap(x, y)
            else:
                await self.input.tap(self._screen_w // 2, self._screen_h // 2)
            await wait(timing.get("monument_popup_wait", 1.5), jitter, "reopen popup")
            png = await self.capture.capture()
            ctx.last_screenshot = png

        # Read the monument popup to check ownership
        text = self._call_vision(png, "check_monument")
        ctx.stats.api_calls += 1
        info = parse_monument_info(text)
        ctx.monument_info = info

        ctx.log_action(
            f"Post-battle monument: ownership={info.ownership}, "
            f"is_friendly={info.is_friendly}, "
            f"ownership_text='{info.ownership_text}'"
        )

        # Check if the monument is now ours (blue "Star Spirit" ownership)
        if info.is_friendly:
            ctx.stats.monuments_captured += 1
            ctx.log_action("Monument captured! (ownership is friendly)")
            await self.actions.close_popup()
            return BotState.OPENING_MINIMAP

        # Not captured yet — are there more defenders to fight?
        if info.action_button.visible and info.action_button.action_type in ("attack", "unknown"):
            # Check unbeatable list before attacking next defender
            for defender in info.defenders:
                if defender.status == "active" and defender.name:
                    if defender.name in self._unbeatable_players:
                        ctx.log_action(
                            f"Next defender '{defender.name}' is unbeatable — moving on"
                        )
                        await self.actions.close_popup()
                        return BotState.OPENING_MINIMAP

            ctx.log_action("More defenders to fight — attacking next")
            return BotState.ATTACKING

        ctx.log_action("Post-battle: no attack available — moving on")
        await self.actions.close_popup()
        return BotState.OPENING_MINIMAP

    async def handle_refreshing_popup(self, ctx: BotContext, config: dict) -> BotState:
        """Close and reopen the popup to fight the next defender."""
        await self.actions.refresh_popup()

        # Tap the world monument again to reopen the popup
        x, y = self.calibrator.get_pixel("world_monument")
        if x > 0 and y > 0:
            ctx.log_action(f"Re-tapping world monument at ({x}, {y})")
            await self.input.tap(x, y)
        else:
            # Fallback: tap center
            x, y = self._screen_w // 2, self._screen_h // 2
            ctx.log_action(f"Tapping center ({x}, {y})")
            await self.input.tap(x, y)

        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        await wait(timing.get("monument_popup_wait", 1.5), jitter, "popup reopen")

        return BotState.CHECKING_MONUMENT

    async def handle_reconnecting(self, ctx: BotContext, config: dict) -> BotState:
        """Handle navigation back into Alien Minefield.

        Entered when:
        - 'logged_out' popup detected (needs delay + restart)
        - 'home_screen' detected (needs Star Trek → Alien Minefield)
        - 'mode_select' detected (needs Alien Minefield tap)

        Checks the current screen first and skips to the right step.
        """
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        max_retries = 10

        # Determine where we are right now (wait past loading)
        png, current = await self._wait_past_loading(ctx, config, "reconnect start")
        ctx.log_action(f"Reconnect: starting on {current.screen_type}")

        # ── Step 1: If logged_out, wait then tap Restart ─────────────
        if current.screen_type == "logged_out":
            bot_cfg = config.get("bot", {})
            delay = bot_cfg.get("logged_out_restart_delay", 600)
            m, s = divmod(delay, 60)
            h, m = divmod(m, 60)
            ctx.log_action(f"Logged out — restarting in {h}h{m:02d}m{s:02d}s")
            await wait(float(delay), 0, "logged out restart delay")

            png = await self.capture.capture()
            ctx.last_screenshot = png
            self._calibrate_for_screen(png, "logged_out", ctx)
            rx, ry = self.calibrator.get_pixel("restart_button")
            ctx.log_action(f"Tapping Restart button at ({rx}, {ry})")
            await self.input.tap(rx, ry)
            await wait(timing.get("screen_transition", 2.0), jitter, "restart tap")

            # Game needs time to load, check for updates, and log in
            ctx.log_action("Waiting for game to load after restart...")
            await wait(20.0, jitter, "post-restart loading")

            # Wait for home_screen
            for attempt in range(max_retries):
                png = await self.capture.capture()
                ctx.last_screenshot = png
                text = self._call_vision(png, "identify_screen")
                ctx.stats.api_calls += 1
                current = parse_screen_identification(text)
                ctx.log_action(f"Reconnect: screen={current.screen_type} (attempt {attempt + 1})")

                if current.screen_type in ("home_screen", "mode_select"):
                    break
                elif current.screen_type == "logged_out":
                    ctx.log_action("Still showing logged out popup — retapping Restart")
                    rx, ry = self.calibrator.get_pixel("restart_button")
                    await self.input.tap(rx, ry)
                    await wait(15.0, jitter, "retry restart loading")
                elif current.screen_type == "loading":
                    await wait(10.0, jitter, "waiting for load")
                else:
                    await wait(10.0, jitter, "waiting for home screen")
            else:
                ctx.log_action("Could not reach home screen — going to error recovery")
                ctx.error_message = "Reconnect failed: home screen not reached"
                return BotState.ERROR_RECOVERY

        # ── Step 2: If home_screen, tap Star Trek with verification ──
        if current.screen_type in ("home_screen", "main_map", "unknown"):
            png, current, ok = await self._tap_and_verify(
                element_name="star_trek_button",
                screen_type="home_screen",
                expected_screens=["mode_select"],
                ctx=ctx, config=config, png=png,
            )
            if not ok:
                ctx.error_message = "Reconnect failed: mode select not reached"
                return BotState.ERROR_RECOVERY

        # ── Step 3: If mode_select, tap Alien Minefield with verification
        if current.screen_type == "mode_select":
            png, current, ok = await self._tap_and_verify(
                element_name="alien_minefield_button",
                screen_type="mode_select",
                expected_screens=["main_map", "loading"],
                ctx=ctx, config=config, png=png,
            )
            # Even if not "ok", the game might still be loading — proceed
            await wait(5.0, jitter, "game loading")

        ctx.log_action("Reconnection complete — reinitializing")
        return BotState.INITIALIZING

    async def handle_idle(self, ctx: BotContext, config: dict) -> BotState:
        """No targets or hibernation active. Sleep then re-check.

        Uses a random idle duration (10-120s) to avoid predictable patterns.
        On resume, presses back to close any open minimap overlay, then
        reopens it — the game doesn't refresh minimap data unless you do so.
        """
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)

        if self._hibernation_seconds is not None and self._hibernation_seconds > 0:
            # Add a 30-second buffer so the timer has fully expired
            sleep_secs = self._hibernation_seconds + 30
            m, s = divmod(sleep_secs, 60)
            h, m = divmod(m, 60)
            ctx.log_action(f"Hibernation sleep — waking in {h}h{m:02d}m{s:02d}s")
            await wait(float(sleep_secs), 0.05, "hibernation sleep")
            self._hibernation_seconds = None
        else:
            idle_secs = random.uniform(10.0, 120.0)
            ctx.log_action(f"Idle — rechecking in {idle_secs:.0f}s")
            await wait(idle_secs, 0.15, "idle wait")

        # Assume the minimap is still open (that's what we were looking at
        # before going idle).  Close it by tapping the X button at center-bottom
        # — the game doesn't refresh minimap data unless you close and reopen.
        # Skip Vision entirely here; OPENING_MINIMAP handles verification locally.
        ctx.log_action("Resuming from idle — closing minimap overlay via X button")
        png = await self.capture.capture()
        ctx.last_screenshot = png

        # Try to find the minimap close button locally
        self._calibrate_for_screen(png, "minimap", ctx)
        if self.calibrator.is_calibrated("minimap_close"):
            cx, cy = self.calibrator.get_pixel("minimap_close")
            ctx.log_action(f"Tapping minimap X at ({cx}, {cy})")
            await self.input.tap(cx, cy)
        else:
            # Fallback: tap bottom-center where the X typically sits
            cx = self._screen_w // 2
            cy = int(self._screen_h * 0.87)
            ctx.log_action(f"minimap_close not found — tapping bottom-center ({cx}, {cy})")
            await self.input.tap(cx, cy)
        await wait(1.5, jitter, "close minimap after idle")

        # Wait past any loading/black screen (no Vision — just brightness check)
        png = await self._wait_past_loading_local(ctx, config, "idle resume")

        # Sanity check: if minimap squares are still visible, the X tap missed —
        # try once more at the center-bottom fallback position
        detection = find_minimap_squares(png)
        if detection and len(detection.squares) >= 2:
            ctx.log_action("Minimap still open — retrying X tap at center-bottom")
            await self.input.tap(self._screen_w // 2, int(self._screen_h * 0.87))
            await wait(1.5, jitter, "close minimap retry")

        # Clear visited slots so we re-evaluate all monuments
        self._visited_slots.clear()

        # Go straight to OPENING_MINIMAP — it will capture a fresh screenshot,
        # tap the minimap button, and verify locally with find_minimap_squares.
        # If that fails it falls back to Vision identify_screen.
        ctx.log_action("Opening fresh minimap")
        return BotState.OPENING_MINIMAP

    async def handle_error_recovery(self, ctx: BotContext, config: dict) -> BotState:
        """Multi-strategy error recovery."""
        self._retries_without_progress += 1
        ctx.log_action(f"Error recovery: {ctx.error_message}")

        # Too many consecutive failures — cool down in idle
        max_retries = config.get("bot", {}).get("max_idle_retries", 8)
        if self._retries_without_progress > max_retries:
            ctx.log_action(
                f"Too many retries without progress ({self._retries_without_progress}) "
                f"— cooling down in idle"
            )
            self._retries_without_progress = 0
            return BotState.IDLE

        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        await wait(timing.get("error_recovery_wait", 3.0), jitter, "error recovery")

        # Strategy 1: Identify current screen (wait past loading)
        try:
            png, screen = await self._wait_past_loading(ctx, config, "error recovery")

            ctx.log_action(f"Recovery: detected {screen.screen_type}")

            if screen.screen_type == "logged_out":
                ctx.log_action("Logged out detected — entering reconnection flow")
                return BotState.RECONNECTING
            elif screen.screen_type == "minimap":
                return BotState.READING_MINIMAP
            elif screen.screen_type == "monument_popup":
                return BotState.CHECKING_MONUMENT
            elif screen.screen_type == "battle_active":
                return BotState.SKIPPING_BATTLE
            elif screen.screen_type == "battle_result":
                return BotState.POST_BATTLE
        except Exception as e:
            ctx.log_action(f"Screen identification failed: {e}")

        # Strategy 2: Try back button
        ctx.log_action("Recovery: pressing back")
        await self.actions.press_back()
        await wait(2.0, jitter, "recovery back")

        # Strategy 3: Restart from minimap
        ctx.log_action("Recovery: restarting from minimap")
        return BotState.OPENING_MINIMAP

    async def handle_paused(self, ctx: BotContext, config: dict) -> BotState:
        """Do nothing while paused. StateMachine handles resume."""
        await wait(1.0, 0, "paused")
        return BotState.PAUSED

    async def handle_stopped(self, ctx: BotContext, config: dict) -> BotState:
        """Terminal state."""
        return BotState.STOPPED
