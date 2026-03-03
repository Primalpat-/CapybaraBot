"""State handlers for each bot state."""

import asyncio
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
from src.bot.state_machine import BotState, BotContext, BotPausedInterrupt
from src.utils.timing import wait
from src.vision.cache import VisionCache
from src.vision.client import VisionClient
from src.vision.element_detector import ElementDetector
from src.vision.minimap_detector import find_minimap_squares, save_detection_debug
from src.vision.parser import (
    parse_screen_identification,
    parse_monument_info,
    parse_post_battle,
    parse_calibration_result,
    parse_recovery_guidance,
    parse_daily_popup_check,
    parse_timer_seconds,
)
from src.vision.ocr_reader import read_monument_popup, check_screen_ocr
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
        state_machine=None,
    ):
        self.capture = capture
        self.input = adb_input
        self.vision = vision
        self.cache = cache
        self.actions = actions
        self.config = config
        self.calibrator = calibrator
        self.element_detector = element_detector
        self._state_machine = state_machine
        self._visited_slots: set[int] = set()
        self._screen_w = config.get("screen", {}).get("width", 1080)
        self._screen_h = config.get("screen", {}).get("height", 1920)
        self._minimap_open_attempts = 0
        self._retries_without_progress = 0  # consecutive failures across states
        self._hibernation_seconds: int | None = None  # countdown from last detection
        self._dormant_seconds: int | None = None  # dormant period countdown
        self._defeat_counts: dict[str, int] = {}  # player_name -> consecutive defeats
        self._unbeatable_players: set[str] = set()  # derived: players exceeding max_defeats
        self._last_unbeatable_decay: float = time.time()  # when we last decayed the list
        self._attacking_defender: str | None = None  # defender we're currently fighting
        self._battle_result_png: bytes | None = None  # screenshot of battle result for defeat logging
        self._contest_until: float = 0.0  # timestamp — stay at monument and fight until this time
        self._event_logger = None   # persistence.EventLogger, injected from main.py
        self._periodic_saver = None  # persistence.PeriodicSaver, injected from main.py

    def _is_paused(self) -> bool:
        """Check if the bot has been paused (used as interrupt_check for waits)."""
        return self._state_machine is not None and self._state_machine.is_paused

    async def _wait(self, base_delay: float, jitter_factor: float = 0.3,
                    label: str = "") -> float:
        """Pause-aware wait — raises BotPausedInterrupt if paused mid-sleep."""
        result = await wait(base_delay, jitter_factor, label,
                            interrupt_check=self._is_paused)
        if result < 0:
            raise BotPausedInterrupt()
        return result

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
            await self._wait(1.0, jitter, "ok button tap")
            await self.input.tap(tap_x, tap_y)
            await self._wait(1.0, jitter, "ok button retry")

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
        await self._wait(timing.get("screen_transition", 2.0), jitter, "occupy cancel")

    async def _dismiss_daily_popups(self, ctx: BotContext, config: dict,
                                       max_popups: int = 3) -> int:
        """Dismiss up to max_popups daily promotional popups.

        Each popup has a 'Do not show again today' radio/checkbox and an X close
        button. Taps the radio text first, then the X, and loops until no popup
        is visible or max_popups is reached.

        Returns the number of popups dismissed.
        """
        dismissed = 0
        for i in range(max_popups):
            png = await self.capture.capture()
            ctx.last_screenshot = png
            text = self._call_vision(png, "check_daily_popup")
            ctx.stats.vision_calls += 1
            result = parse_daily_popup_check(text)

            if not result.popup_visible:
                break

            ctx.log_action(f"Daily popup #{i + 1} detected: {result.details}")

            if result.do_not_show_found:
                x = int(result.do_not_show_x / 100 * self._screen_w)
                y = int(result.do_not_show_y / 100 * self._screen_h)
                ctx.log_action(f"Tapping 'Do not show again' at ({x}, {y})")
                await self.input.tap(x, y)
                await self._wait(0.8, 0.1, "do not show tap")

            if result.close_found:
                x = int(result.close_x / 100 * self._screen_w)
                y = int(result.close_y / 100 * self._screen_h)
                ctx.log_action(f"Tapping popup close button at ({x}, {y})")
                await self.input.tap(x, y)
                await self._wait(1.5, 0.2, "popup close tap")

            dismissed += 1

        if dismissed > 0:
            ctx.log_action(f"Dismissed {dismissed} daily popup(s)")
        return dismissed

    async def _launch_app(self, ctx: BotContext, config: dict) -> bool:
        """Launch the game app via ADB monkey command.

        Falls back to Vision-calibrated app_icon tap if ADB launch fails.
        Returns True if the launch command succeeded.
        """
        package = config.get("bot", {}).get("app_package", "com.habby.capybara")
        try:
            stdout, stderr, rc = await self.input.connection.run_adb(
                "shell", "monkey", "-p", package,
                "-c", "android.intent.category.LAUNCHER", "1"
            )
            if rc == 0:
                ctx.log_action(f"Launched app via ADB: {package}")
                return True
            logger.warning(f"ADB monkey launch failed (rc={rc}): {stderr}")
        except Exception as e:
            logger.warning(f"ADB monkey launch error: {e}")

        # Fallback: Vision-calibrate android_home screen and tap app_icon
        try:
            png = await self.capture.capture()
            ctx.last_screenshot = png
            self._calibrate_for_screen(png, "android_home", ctx)
            x, y = self.calibrator.get_pixel("app_icon")
            if x > 0 and y > 0:
                ctx.log_action(f"Tapping app icon at ({x}, {y}) (ADB fallback)")
                await self.input.tap(x, y)
                return True
        except Exception as e:
            logger.warning(f"Vision app icon fallback failed: {e}")

        return False

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
            await self._wait(tap_wait, jitter, f"{element_name} verify")

            # Check what screen we're on now
            check_png = await self.capture.capture()
            ctx.last_screenshot = check_png

            # Quick local loading check first
            if self._is_loading_screen(check_png):
                check_png = await self._wait_past_loading_local(
                    ctx, config, f"{element_name} post-tap"
                )

            text = self._call_vision(check_png, "identify_screen")
            ctx.stats.vision_calls += 1
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

    async def _enter_hibernation(self, screen, ctx: BotContext, config: dict) -> BotState:
        """Handle hibernation detection — parse timer and go idle.

        If the timer is broken (negative or unparseable), exits the game mode
        and re-enters via RECONNECTING to get a fresh timer reading.
        """
        max_hibernation = 4 * 3600 + 5 * 60  # hibernation is never longer than ~4 hours
        secs = parse_timer_seconds(screen.timer)
        self._hibernation_seconds = secs
        if secs is not None and 0 < secs <= max_hibernation:
            m, s = divmod(secs, 60)
            h, m = divmod(m, 60)
            ctx.log_action(f"Hibernation active — {h}h{m:02d}m{s:02d}s remaining, sleeping until it ends")
            if self._event_logger:
                self._event_logger.log("hibernation_start", duration=secs)
            return BotState.IDLE

        # Timer is broken (negative, zero, unparseable, or impossibly long) — exit and re-enter
        ctx.log_action(
            f"Hibernation timer broken ('{screen.timer}', parsed={secs}s) — "
            "exiting game mode to get fresh timer"
        )
        png = await self.capture.capture()
        ctx.last_screenshot = png
        self._calibrate_for_screen(png, "main_map", ctx)
        x, y = self.calibrator.get_pixel("exit_mode_button")
        if x > 0 and y > 0:
            ctx.log_action(f"Tapping exit mode button at ({x}, {y})")
            await self.input.tap(x, y)
            timing = config.get("timing", {})
            jitter = timing.get("jitter_factor", 0.3)
            await self._wait(3.0, jitter, "exit mode transition")
        else:
            ctx.log_action("Could not find exit mode button — using Android back")
            await self.input.back()
            await self._wait(2.0, 0.3, "back button transition")
        return BotState.RECONNECTING

    def _enter_dormant(self, screen, ctx: BotContext) -> BotState:
        """Handle dormant period — parse timer and go idle."""
        secs = parse_timer_seconds(screen.timer)
        self._dormant_seconds = secs
        if secs is not None:
            m, s = divmod(secs, 60)
            h, m = divmod(m, 60)
            ctx.log_action(f"Dormant period — {h}h{m:02d}m{s:02d}s remaining, sleeping until it ends")
        else:
            ctx.log_action("Dormant period — could not read timer, will recheck in 60s")
        if self._event_logger:
            self._event_logger.log("dormant_start", duration=secs)
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

    def _call_vision(self, png_bytes: bytes, prompt_name: str, **format_vars) -> str:
        """Call vision API with caching and logging.

        Extra keyword args are used to format {placeholders} in the prompt
        template (e.g., faction_name="Star Spirit").
        """
        cached = self.cache.get(png_bytes, prompt_name)
        if cached is not None:
            logger.debug(f"Vision cache hit for {prompt_name}")
            return cached

        logger.info(f"Calling Vision API: {prompt_name}")
        system, prompt = get_prompt(prompt_name)
        if format_vars:
            prompt = prompt.format(**format_vars)
            system = system.format(**format_vars) if system else system
        response = self.vision.analyze_screenshot(png_bytes, prompt, system)
        self.cache.put(png_bytes, prompt_name, response.text)
        return response.text

    def _read_monument_ocr(self, png: bytes, ctx: BotContext, config: dict):
        """Try reading monument popup with local OCR.

        Returns a MonumentInfo if OCR confidence is above threshold, else None
        (caller should fall back to Vision API).

        Falls back to Vision API when:
        - OCR is disabled in config
        - Overall confidence is below threshold
        - Active defenders found but ALL have power=0 (likely misaligned crop regions)
        """
        from src.vision.parser import MonumentInfo, DefenderInfo, ActionButton

        ocr_cfg = config.get("ocr", {})
        if not ocr_cfg.get("enabled", True):
            return None

        threshold = ocr_cfg.get("confidence_threshold", 0.4)
        faction = config.get("bot", {}).get("faction", "Star Spirit")

        try:
            reading = read_monument_popup(png, friendly_faction=faction)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return None

        if reading.overall_confidence < threshold:
            logger.info(
                f"OCR confidence too low ({reading.overall_confidence:.2f} < {threshold}) "
                "— falling back to Vision API"
            )
            return None

        # Sanity check: if there are active defenders but ALL have power=0,
        # the OCR crop regions are likely misaligned — fall back to Vision API
        active_defenders = [d for d in reading.defenders if d.status == "active"]
        if active_defenders and all(d.power == 0 for d in active_defenders):
            logger.info(
                f"OCR found {len(active_defenders)} active defenders but all have "
                "power=0 — likely misaligned crop regions, falling back to Vision API"
            )
            return None

        # Convert OCRMonumentReading to MonumentInfo for compatibility
        defenders = []
        for d in reading.defenders:
            defenders.append(DefenderInfo(
                slot=d.slot,
                status=d.status,
                name=d.name,
                power=d.power,
            ))

        btn_text = reading.action_button_text
        btn_lower = btn_text.lower()
        if "attack" in btn_lower:
            action_type = "attack"
        elif "claim" in btn_lower:
            action_type = "claim"
        elif "visit" in btn_lower:
            action_type = "visit"
        else:
            action_type = "unknown"

        info = MonumentInfo(
            ownership="player" if reading.is_friendly else "enemy",
            is_friendly=reading.is_friendly,
            monument_name="",
            defenders=defenders,
            all_defenders_defeated=all(d.status != "active" for d in defenders),
            action_button=ActionButton(
                visible=bool(btn_text),
                text=btn_text,
                action_type=action_type,
            ),
            ownership_text=reading.ownership_text,
            total_garrison_power=reading.total_garrison_power,
        )

        ctx.stats.ocr_reads += 1
        ctx.log_action(
            f"OCR read: friendly={info.is_friendly}, "
            f"button='{btn_text}', power={reading.total_garrison_power}, "
            f"conf={reading.overall_confidence:.2f}"
        )
        return info

    def _update_monument_tracker(self, slot: int, info, ctx: BotContext, config: dict) -> None:
        """Update monument tracker with popup reading data (from OCR or Vision).

        Handles: status tracking, flip detection, flip velocity, power tracking,
        and progress timestamping. Extracted from handle_checking_monument to
        share with handle_post_battle and handle_contesting.
        """
        if slot not in ctx.monument_tracker:
            return

        rec = ctx.monument_tracker[slot]
        old_status = rec.last_status
        new_status = "friendly" if info.is_friendly else "enemy"
        rec.last_checked = time.time()
        rec.check_count += 1
        rec.last_status = new_status
        rec.owner_name = info.ownership_text or info.ownership or ""
        rec.garrison_count = sum(
            1 for d in info.defenders if d.status == "active"
        )

        # Power tracking
        rec.garrison_power = info.total_garrison_power
        rec.defender_powers = [d.power for d in info.defenders if d.status == "active"]
        rec.defender_names = [d.name for d in info.defenders if d.status == "active"]

        # Monument check counts as meaningful progress
        ctx.last_progress_time = time.time()
        ctx.stagnation_recovery_attempts = 0

        # Track consecutive enemy checks
        if new_status == "enemy":
            rec.consecutive_enemy_checks += 1
        else:
            rec.consecutive_enemy_checks = 0

        # Track flips
        if old_status and old_status != "unknown" and new_status != old_status:
            now = time.time()
            # Compute flip velocity (exponential moving average)
            if rec.last_flip_time > 0:
                hours_since = (now - rec.last_flip_time) / 3600.0
                if hours_since > 0:
                    instant_velocity = 1.0 / hours_since
                    alpha = 0.3
                    rec.flip_velocity = alpha * instant_velocity + (1 - alpha) * rec.flip_velocity
            else:
                # First flip — estimate at 1/hour
                rec.flip_velocity = 1.0

            rec.flip_history.append({
                "time": now,
                "from": old_status,
                "to": new_status,
            })
            rec.prune_flip_history()

            rec.last_flip_time = now
            rec.last_flip_from = old_status
            rec.last_flip_to = new_status
            if new_status == "enemy":
                rec.flipped_to_enemy += 1
                # Clear from visited for immediate re-prioritization
                self._visited_slots.discard(slot)
            elif new_status == "friendly":
                rec.flipped_to_friendly += 1
            if self._event_logger:
                self._event_logger.log(
                    "monument_flip", slot=slot,
                    from_status=old_status, to_status=new_status,
                    owner=rec.owner_name,
                    flip_velocity=round(rec.flip_velocity, 2),
                    garrison_power=rec.garrison_power,
                )

    def _can_beat_defender(self, defender, ctx: BotContext, config: dict) -> bool:
        """Check if we should attempt to fight a defender.

        Returns False if the defender is unbeatable or too powerful.
        """
        if not defender.name:
            return True  # unknown defender, always try

        if defender.name in self._unbeatable_players:
            return False

        max_power = config.get("bot", {}).get("max_beatable_defender_power", 0)
        if max_power > 0 and defender.power > max_power:
            logger.info(
                f"Defender '{defender.name}' power {defender.power} exceeds max "
                f"beatable {max_power} — skipping"
            )
            return False

        return True

    def _record_defeat(self, defender_name: str, ctx: BotContext, config: dict) -> None:
        """Record a defeat against a defender, mark unbeatable after N consecutive."""
        if not defender_name:
            return

        max_defeats = config.get("bot", {}).get("max_defeats_before_skip", 2)
        count = self._defeat_counts.get(defender_name, 0) + 1
        self._defeat_counts[defender_name] = count

        if count >= max_defeats:
            self._unbeatable_players.add(defender_name)
            ctx.log_action(
                f"Marked '{defender_name}' as unbeatable after {count} defeats "
                f"({len(self._unbeatable_players)} total)"
            )
        else:
            ctx.log_action(
                f"Defeated by '{defender_name}' ({count}/{max_defeats} before skip)"
            )

    def _record_victory(self, defender_name: str) -> None:
        """Record a victory — reset consecutive defeat counter for this player."""
        if defender_name and defender_name in self._defeat_counts:
            del self._defeat_counts[defender_name]
        if defender_name and defender_name in self._unbeatable_players:
            self._unbeatable_players.discard(defender_name)

    def _decay_unbeatable_list(self) -> None:
        """Periodically clear unbeatable list so we retry players after cooldown.

        Called from handle_idle. Clears every 30 minutes.
        """
        decay_interval = 1800  # 30 minutes
        if time.time() - self._last_unbeatable_decay >= decay_interval:
            if self._unbeatable_players or self._defeat_counts:
                logger.info(
                    f"Decaying unbeatable list: clearing {len(self._unbeatable_players)} "
                    f"players and {len(self._defeat_counts)} defeat counts"
                )
                self._unbeatable_players.clear()
                self._defeat_counts.clear()
            self._last_unbeatable_decay = time.time()

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
        ctx.stats.vision_calls += 1

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
            await self._wait(loading_wait, jitter, f"loading wait ({label})")

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
        ctx.stats.vision_calls += 1
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
          dormant_period   → IDLE               Sleep until dormant timer expires
          logged_out       → RECONNECTING       Wait delay → Restart → Star Trek → Alien Minefield
          home_screen      → RECONNECTING       Star Trek → Alien Minefield
          mode_select      → RECONNECTING       Alien Minefield
          android_home     → RECONNECTING       Launch app via ADB → navigate to game
          daily_popup      → dismiss popups → INITIALIZING  Tap "do not show" + X, re-identify
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
            return await self._enter_hibernation(screen, ctx, config)
        elif screen.screen_type == "dormant_period":
            return self._enter_dormant(screen, ctx)
        elif screen.screen_type == "daily_popup":
            ctx.log_action("Daily popup detected — dismissing")
            await self._dismiss_daily_popups(ctx, config)
            return BotState.INITIALIZING
        elif screen.screen_type in ("logged_out", "home_screen", "mode_select", "android_home"):
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

        # Too many consecutive failures — use Vision once to figure out where we are
        max_retries = config.get("bot", {}).get("max_idle_retries", 8)
        if self._retries_without_progress > max_retries:
            ctx.log_action(
                f"Too many retries without progress ({self._retries_without_progress}) "
                f"— reinitializing to identify screen"
            )
            self._minimap_open_attempts = 0
            self._retries_without_progress = 0
            return BotState.INITIALIZING

        if self._minimap_open_attempts > 2:
            ctx.log_action(
                f"Minimap open attempt #{self._minimap_open_attempts} — retrying"
            )

        # After 3 failed attempts, do a Vision check to make sure we're actually
        # in the game — we might be stuck on home_screen/mode_select/logged_out
        if self._minimap_open_attempts == 3:
            ctx.log_action("Multiple minimap failures — verifying we're in the game")
            png = await self.capture.capture()
            ctx.last_screenshot = png
            _, screen = await self._identify_screen(png, ctx, config)
            if screen.screen_type in ("home_screen", "mode_select", "logged_out", "android_home"):
                ctx.log_action(f"Not in game ({screen.screen_type}) — entering reconnection flow")
                self._minimap_open_attempts = 0
                self._retries_without_progress = 0
                return BotState.RECONNECTING
            elif screen.screen_type in ("hibernation", "dormant_period"):
                self._minimap_open_attempts = 0
                self._retries_without_progress = 0
                return BotState.INITIALIZING

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
        await self._wait(tap_wait, jitter, "minimap open")

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

        # Local detection didn't find squares — retry without Vision
        ctx.log_action("Minimap not detected after tap, retrying...")
        return BotState.OPENING_MINIMAP

    def _score_monument_slot(self, slot: int, tracker, config: dict) -> tuple[int, float]:
        """Score a monument slot for rotation priority.

        Returns (tier, tiebreaker) — sorted ascending, lower = higher priority.
        Tier 0: recently captured (post-capture watch)
        Tier 1: high flip velocity or vulnerable friendly (low garrison/power)
        Tier 2: enemy monument — needs capturing
        Tier 3: default — needs checking
        Tier 4: well-defended (skip until stale)
        """
        rec = tracker[slot]
        now = time.time()
        persist_cfg = config.get("persistence", {})
        contest_cfg = config.get("contest", {})
        watch_secs = persist_cfg.get("post_capture_watch_seconds", 300)
        recheck_secs = persist_cfg.get("recheck_interval_seconds", 900)
        flip_vel_threshold = contest_cfg.get("flip_velocity_threshold", 2.0)
        power_vuln_threshold = contest_cfg.get("power_vulnerability_threshold", 5000)

        # Compute effective flip velocity with time-based decay.
        # Half-life of 30 minutes: if no flip for 30 min, velocity halves.
        # This prevents monuments from being stuck on "urgent" forever.
        # Write the decayed value back so the dashboard UI reflects reality.
        effective_flip_velocity = rec.flip_velocity
        if rec.last_flip_time > 0 and rec.flip_velocity > 0:
            hours_since = (now - rec.last_flip_time) / 3600.0
            effective_flip_velocity = rec.flip_velocity * (0.5 ** (hours_since / 0.5))
            rec.flip_velocity = effective_flip_velocity

        # Tier 0: recently captured — watch for counter-attack
        if rec.captured_at and (now - rec.captured_at) < watch_secs:
            return (0, rec.captured_at)

        # Tier 1: high flip velocity — monument is actively contested
        if effective_flip_velocity >= flip_vel_threshold:
            return (1, -effective_flip_velocity)

        # Tier 1: vulnerable friendly — low garrison or low power
        if rec.last_status == "friendly":
            is_low_garrison = rec.garrison_count >= 0 and rec.garrison_count <= 1
            is_low_power = (rec.garrison_power >= 0
                            and rec.garrison_power < power_vuln_threshold
                            and rec.garrison_count > 0)
            if is_low_garrison or is_low_power:
                # Lower power = higher priority (more vulnerable)
                tiebreaker = rec.garrison_power if rec.garrison_power >= 0 else 999999
                return (1, tiebreaker)

        # Tier 2: enemy monument — needs capturing
        if rec.last_status == "enemy":
            # Lower power = easier to take, higher priority
            tiebreaker = rec.garrison_power if rec.garrison_power >= 0 else 999999
            return (2, tiebreaker)

        # Tier 4: SAFE — only the highest-power friendly monument gets skipped.
        # Weaker friendly monuments stay in the rotation so we check them first.
        if (rec.last_status == "friendly"
                and effective_flip_velocity < 0.5
                and rec.garrison_count >= 3
                and rec.garrison_power > 0
                and rec.last_checked > 0
                and (now - rec.last_checked) < recheck_secs):
            # Compare against all friendly monuments with known power
            max_friendly_power = max(
                (r.garrison_power for r in tracker.values()
                 if r.last_status == "friendly" and r.garrison_power > 0),
                default=0,
            )
            if rec.garrison_power >= max_friendly_power:
                return (4, rec.last_checked)

        # Tier 3: default — needs checking (lower power = check sooner)
        tiebreaker = rec.garrison_power if rec.garrison_power > 0 else -1
        return (3, tiebreaker)

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
            ctx.log_action("Could not detect minimap squares — retrying from minimap open")
            return BotState.OPENING_MINIMAP

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

        # Pick next slot using smart rotation scoring
        target_slot = None
        tracker = ctx.monument_tracker
        persist_cfg = config.get("persistence", {})
        watch_secs = persist_cfg.get("post_capture_watch_seconds", 300)

        # Build candidates: unvisited slots + recently-captured slots (post-capture watch)
        candidates = []
        for s in range(1, NUM_MONUMENT_SLOTS + 1):
            in_watch = (tracker[s].captured_at
                        and (time.time() - tracker[s].captured_at) < watch_secs)
            if s not in self._visited_slots or in_watch:
                candidates.append(s)

        # Always prefer red (enemy) slots over blue (friendly) ones
        red_candidates = [s for s in candidates if slot_colors.get(s) == "red"]
        if red_candidates:
            candidates = red_candidates

        # Score all slots and store priority tier for dashboard display
        for s in tracker:
            tier, _ = self._score_monument_slot(s, tracker, config)
            tracker[s].priority_tier = tier

        # Split candidates by urgency:
        # - Urgent (tier 0-2): every cycle
        # - Check (tier 3): every cycle when no urgent; every 10 min when urgent exist
        # - Safe (tier 4): every 15 min (handled by scoring — stale SAFE becomes tier 3)
        check_recheck_secs = persist_cfg.get("check_recheck_interval_seconds", 600)
        now = time.time()

        urgent = [s for s in candidates if tracker[s].priority_tier <= 2]
        if urgent:
            # Urgent every cycle + CHECK only if stale (10 min)
            filtered = list(urgent)
            for s in candidates:
                if tracker[s].priority_tier == 3:
                    if (tracker[s].last_checked <= 0
                            or (now - tracker[s].last_checked) >= check_recheck_secs):
                        filtered.append(s)
            candidates = filtered
        else:
            # No urgent — cycle through all CHECK monuments, skip SAFE
            candidates = [s for s in candidates if tracker[s].priority_tier < 4]

        if candidates:
            candidates.sort(key=lambda s: self._score_monument_slot(s, tracker, config))
            target_slot = candidates[0]

        if target_slot is None:
            ctx.log_action("All monuments checked this cycle — going idle")
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
        self._contest_until = 0.0  # clear contest timer when switching targets
        ctx.log_action(f"Target: slot {target_slot} at ({x}, {y})")

        # Tap the monument icon on the minimap
        logger.info(f"Tapping monument slot {target_slot} at ({x}, {y})")
        await self.input.tap(x, y)

        # No verify needed — we have precise coordinates from detection and the
        # tap almost always works.  Verifying caused a race condition: if the
        # minimap closing animation was still playing, find_minimap_squares would
        # report "still open" → READING_MINIMAP → detection fails (now closed)
        # → OPENING_MINIMAP → reopens minimap → cancels navigation.
        # If the tap missed, APPROACHING_MONUMENT's retry loop handles it.
        self._retries_without_progress = 0
        ctx.log_action("Minimap slot tapped — navigation started")
        return BotState.NAVIGATING

    async def handle_navigating(self, ctx: BotContext, config: dict) -> BotState:
        """Wait a fixed time for the character to walk, then proceed.

        Uses a configurable wait instead of Vision polling — saves API calls
        and avoids overload.  The character always arrives within a few seconds.
        """
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        nav_wait = timing.get("navigation_wait", 8.0)

        ctx.log_action(f"Navigating to monument — waiting {nav_wait:.0f}s")
        await self._wait(nav_wait, jitter, "navigation wait")

        ctx.stats.monuments_visited += 1
        if ctx.current_target:
            self._visited_slots.add(ctx.current_target.get("slot", 0))

        return BotState.APPROACHING_MONUMENT

    async def handle_approaching_monument(self, ctx: BotContext, config: dict) -> BotState:
        """Tap the monument in the game world to open its popup.

        Uses local ElementDetector instead of Vision to check for the popup.
        Stuck detection (60s timeout) catches persistent issues like
        hibernation/logged_out — routes through INITIALIZING for one Vision call.
        """
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)

        # Wait past any loading screen locally (no Vision)
        png = await self._wait_past_loading_local(ctx, config, "approaching monument")

        # Check locally if monument popup is already open
        if self.element_detector is not None:
            popup_els = self.element_detector.detect(png, "monument_popup")
            if any(e.name in ("action_button", "close_popup") for e in popup_els):
                ctx.log_action("Monument popup already open (detected locally)")
                return BotState.CHECKING_MONUMENT

        # Calibrate world_monument position if needed (first arrival only)
        self._calibrate_for_screen(png, "arrived_at_monument", ctx)

        # Tap the calibrated world monument position
        x, y = self.calibrator.get_pixel("world_monument")
        ctx.log_action(f"Tapping world monument at ({x}, {y})")
        await self.input.tap(x, y)

        await self._wait(timing.get("monument_popup_wait", 1.5), jitter, "waiting for popup")

        # Check locally again after tap
        png = await self.capture.capture()
        ctx.last_screenshot = png
        if self.element_detector is not None:
            popup_els = self.element_detector.detect(png, "monument_popup")
            if any(e.name in ("action_button", "close_popup") for e in popup_els):
                ctx.log_action("Monument popup opened (detected locally)")
                return BotState.CHECKING_MONUMENT

        ctx.log_action("Popup not detected locally, retrying tap...")
        return BotState.APPROACHING_MONUMENT

    async def handle_checking_monument(self, ctx: BotContext, config: dict) -> BotState:
        """Read monument popup and decide what to do."""
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        await self._wait(timing.get("monument_popup_wait", 1.5), jitter, "popup wait")

        png = await self.capture.capture()
        ctx.last_screenshot = png

        # Calibrate popup elements if needed
        self._calibrate_for_screen(png, "monument_popup", ctx)

        # Try OCR first (free, ~200ms), fall back to Vision API
        info = self._read_monument_ocr(png, ctx, config)
        if info is None:
            faction = config.get("bot", {}).get("faction", "Star Spirit")
            text = self._call_vision(png, "check_monument", faction_name=faction)
            ctx.stats.vision_calls += 1
            info = parse_monument_info(text)
        ctx.monument_info = info

        # Cross-check: action button "Attack" means enemy, regardless of ownership text
        btn_text = (info.action_button.text or "").lower()
        btn_type = info.action_button.action_type
        if info.is_friendly and (btn_type == "attack" or "attack" in btn_text):
            ctx.log_action(
                f"Ownership says friendly but action button says '{info.action_button.text}' — "
                "overriding to enemy"
            )
            info.is_friendly = False

        # Update monument tracker with popup result
        slot = ctx.current_target.get("slot", 0) if ctx.current_target else 0
        self._update_monument_tracker(slot, info, ctx, config)

        ctx.log_action(
            f"Monument: {info.monument_name or 'unnamed'}, "
            f"ownership={info.ownership}, friendly={info.is_friendly}, "
            f"power={info.total_garrison_power}"
        )

        if info.is_friendly:
            # If contest timer active, stay and guard for flips
            if time.time() < self._contest_until:
                remaining = int(self._contest_until - time.time())
                ctx.log_action(
                    f"Friendly but contesting — guarding ({remaining}s remaining)"
                )
                await self.actions.close_popup()
                await self._wait(15.0, 0.3, "contest guard wait")
                return BotState.APPROACHING_MONUMENT
            # Check if we should enter contest mode for this friendly monument
            slot = ctx.current_target.get("slot", 0) if ctx.current_target else 0
            if slot and self._should_enter_contest(slot, ctx, config):
                ctx.log_action("Friendly monument — entering contest mode (vulnerable/contested)")
                await self.actions.close_popup()
                return BotState.CONTESTING
            ctx.log_action("Friendly monument — skipping")
            self._contest_until = 0.0
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
                if defender.status == "active":
                    if not self._can_beat_defender(defender, ctx, config):
                        ctx.log_action(
                            f"Skipping — defender '{defender.name}' "
                            f"(power={defender.power}) is unbeatable or too strong"
                        )
                        self._contest_until = 0.0
                        await self.actions.close_popup()
                        return BotState.OPENING_MINIMAP
                    break  # only check the first active defender
            # Start 5-minute contest timer on first attack at this monument
            contest_secs = config.get("bot", {}).get("contest_duration", 300)
            if self._contest_until == 0.0:
                self._contest_until = time.time() + contest_secs
                ctx.log_action(f"Starting {contest_secs}s contest timer for this monument")
            # Store the defender we're about to fight for post-battle tracking
            self._attacking_defender = None
            for defender in info.defenders:
                if defender.status == "active" and defender.name:
                    self._attacking_defender = defender.name
                    break
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
        await self._wait(timing.get("screen_transition", 2.0), jitter, "battle start")

        # Wait past loading locally (free) — no Vision needed
        png = await self._wait_past_loading_local(ctx, config, "attack verify")

        # Detect battle state locally: skip button = active, ok button = fast finish
        skip_found = False
        ok_found = False
        if self.element_detector is not None:
            battle_elements = self.element_detector.detect(png, "battle_active")
            result_elements = self.element_detector.detect(png, "battle_result")
            skip_found = any(e.name == "skip_battle" for e in battle_elements)
            ok_found = any(e.name == "ok_button" for e in result_elements)

        if skip_found or ok_found:
            ctx.stats.battles_fought += 1
            if ok_found:
                ctx.log_action("Battle already finished (detected OK button locally)")
            else:
                ctx.log_action("Battle started (detected skip button locally)")
            return BotState.SKIPPING_BATTLE

        # Neither button found — retry once after a short wait (battle may still load)
        ctx.log_action("No battle UI detected locally — retrying after short wait")
        await self._wait(2.0, jitter, "battle detect retry")
        png = await self.capture.capture()
        ctx.last_screenshot = png

        if self.element_detector is not None:
            battle_elements = self.element_detector.detect(png, "battle_active")
            result_elements = self.element_detector.detect(png, "battle_result")
            skip_found = any(e.name == "skip_battle" for e in battle_elements)
            ok_found = any(e.name == "ok_button" for e in result_elements)

        if skip_found or ok_found:
            ctx.stats.battles_fought += 1
            if ok_found:
                ctx.log_action("Battle detected on retry (OK button)")
            else:
                ctx.log_action("Battle detected on retry (skip button)")
            return BotState.SKIPPING_BATTLE

        # Still nothing — popup was likely stale
        ctx.log_action(
            "Battle did not start (no battle UI detected locally) — "
            "popup was likely stale, refreshing"
        )
        await self.actions.close_popup()
        await self._wait(timing.get("monument_popup_wait", 1.5), jitter, "stale popup close")
        return BotState.REFRESHING_POPUP

    async def handle_skipping_battle(self, ctx: BotContext, config: dict) -> BotState:
        """Poll for battle end, tapping skip/speed-up."""
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)

        # Capture screenshot directly — do NOT use _wait_past_loading_local here.
        # The battle animation screen is intentionally dark (mean brightness < 18)
        # and would be mistaken for a loading screen, delaying skip by 30+ seconds.
        png = await self.capture.capture()
        ctx.last_screenshot = png

        # Calibrate skip button if needed
        if self.calibrator.needs_calibration("battle_active"):
            self._calibrate_for_screen(png, "battle_active", ctx)

        # Tap skip button (it's fine to tap even if not visible)
        await self.actions.tap_skip_battle()
        await self._wait(timing.get("battle_poll_interval", 3.0), jitter, "battle poll")

        png = await self.capture.capture()
        ctx.last_screenshot = png

        # Detect battle end locally — look for OK button (yellow)
        ok_found = False
        skip_found = False
        if self.element_detector is not None:
            result_elements = self.element_detector.detect(png, "battle_result")
            ok_found = any(e.name == "ok_button" for e in result_elements)

        if ok_found:
            # Battle ended — save screenshot for potential defeat analysis, tap OK
            ctx.log_action("Battle ended (OK button detected locally)")
            self._battle_result_png = png
            await self._tap_ok_button(png, ctx, config)
            return BotState.POST_BATTLE

        # No OK button — check if skip button is still visible (battle active)
        if self.element_detector is not None:
            battle_elements = self.element_detector.detect(png, "battle_active")
            skip_found = any(e.name == "skip_battle" for e in battle_elements)

        if not skip_found and not ok_found:
            # Neither button found — stuck detection (60s timeout) is our safety net
            ctx.log_action("No battle UI detected locally — retrying")
        else:
            ctx.log_action("Battle still in progress...")

        return BotState.SKIPPING_BATTLE

    async def handle_post_battle(self, ctx: BotContext, config: dict) -> BotState:
        """Check monument state after a battle.

        After tapping OK on the battle results screen, the monument popup should
        reappear.  We read it with check_monument and look at the Monument
        Ownership section (blue "Star Spirit" = ours, red = enemy).

        Defeat detection: if the defender we attacked is still "active" in the
        popup, we lost.  The opponent name comes from _attacking_defender (stored
        before the battle started).
        """
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        await self._wait(timing.get("screen_transition", 2.0), jitter, "post battle wait")

        # Wait past any loading screen locally (no Vision)
        png = await self._wait_past_loading_local(ctx, config, "post battle")

        # Check locally for occupy prompt (pink cancel button)
        if self.element_detector is not None:
            occupy_els = self.element_detector.detect(png, "occupy_prompt")
            if any(e.name == "occupy_cancel_button" for e in occupy_els):
                await self._dismiss_occupy_prompt(png, ctx, config)
                ctx.stats.monuments_captured += 1
                ctx.last_progress_time = time.time()
                ctx.stagnation_recovery_attempts = 0
                slot = ctx.current_target.get("slot", 0) if ctx.current_target else 0
                if slot in ctx.monument_tracker:
                    rec = ctx.monument_tracker[slot]
                    rec.captured_at = time.time()
                    rec.times_captured += 1
                if self._event_logger:
                    self._event_logger.log("monument_captured", slot=slot)
                ctx.log_action("Monument captured! (dismissed occupy swap prompt)")
                self._attacking_defender = None
                self._battle_result_png = None
                return BotState.INITIALIZING

        # Check locally for monument popup (action button or close button)
        popup_detected = False
        if self.element_detector is not None:
            popup_els = self.element_detector.detect(png, "monument_popup")
            popup_detected = any(
                e.name in ("action_button", "close_popup") for e in popup_els
            )

        if not popup_detected:
            # Popup not visible — tap the monument to reopen it
            ctx.log_action("Post-battle popup not visible — tapping monument to reopen")
            x, y = self.calibrator.get_pixel("world_monument")
            if x > 0 and y > 0:
                await self.input.tap(x, y)
            else:
                await self.input.tap(self._screen_w // 2, self._screen_h // 2)
            await self._wait(timing.get("monument_popup_wait", 1.5), jitter, "reopen popup")
            png = await self.capture.capture()
            ctx.last_screenshot = png

        # Read the monument popup — try OCR first, Vision API fallback
        info = self._read_monument_ocr(png, ctx, config)
        if info is None:
            faction = config.get("bot", {}).get("faction", "Star Spirit")
            text = self._call_vision(png, "check_monument", faction_name=faction)
            ctx.stats.vision_calls += 1
            info = parse_monument_info(text)
        ctx.monument_info = info

        # Update tracker with power data
        slot = ctx.current_target.get("slot", 0) if ctx.current_target else 0
        self._update_monument_tracker(slot, info, ctx, config)

        ctx.log_action(
            f"Post-battle monument: ownership={info.ownership}, "
            f"is_friendly={info.is_friendly}, "
            f"ownership_text='{info.ownership_text}', "
            f"power={info.total_garrison_power}"
        )

        # Detect defeat: the defender we attacked is still active
        if self._attacking_defender:
            still_active = any(
                d.status == "active" and d.name == self._attacking_defender
                for d in info.defenders
            )
            if still_active:
                ctx.stats.defeats += 1
                ctx.last_progress_time = time.time()
                ctx.stagnation_recovery_attempts = 0
                if self._event_logger:
                    self._event_logger.log(
                        "battle_lost", slot=slot,
                        defender=self._attacking_defender,
                    )
                # Save defeat screenshot if we have it
                if self._battle_result_png:
                    defeat_dir = Path(__file__).resolve().parents[2] / "screenshots" / "defeats"
                    defeat_dir.mkdir(parents=True, exist_ok=True)
                    defeat_path = defeat_dir / f"defeat_{int(time.time())}.png"
                    defeat_path.write_bytes(self._battle_result_png)
                    logger.info(f"Saved defeat screenshot: {defeat_path}")
                self._record_defeat(self._attacking_defender, ctx, config)
                self._attacking_defender = None
                self._battle_result_png = None
                # If contest timer active, stay and keep watching for flips
                if time.time() < self._contest_until:
                    remaining = int(self._contest_until - time.time())
                    ctx.log_action(
                        f"Defeated but contesting — staying to watch ({remaining}s remaining)"
                    )
                    await self.actions.close_popup()
                    await self._wait(15.0, 0.3, "contest post-defeat wait")
                    return BotState.APPROACHING_MONUMENT
                self._contest_until = 0.0
                await self.actions.close_popup()
                return BotState.OPENING_MINIMAP
            else:
                # Defender was defeated — it's a victory
                ctx.last_progress_time = time.time()
                ctx.stagnation_recovery_attempts = 0
                self._record_victory(self._attacking_defender)
                if self._event_logger:
                    self._event_logger.log(
                        "battle_won", slot=slot,
                        defender=self._attacking_defender,
                    )

        self._attacking_defender = None
        self._battle_result_png = None

        # Check if the monument is now ours (blue "Star Spirit" ownership)
        if info.is_friendly:
            ctx.stats.monuments_captured += 1
            ctx.last_progress_time = time.time()
            ctx.stagnation_recovery_attempts = 0
            if slot in ctx.monument_tracker:
                rec = ctx.monument_tracker[slot]
                rec.captured_at = time.time()
                rec.times_captured += 1
            if self._event_logger:
                self._event_logger.log("monument_captured", slot=slot)
            ctx.log_action("Monument captured! (ownership is friendly)")

            # Enter contest mode if the monument is contested
            if self._should_enter_contest(slot, ctx, config):
                ctx.log_action("Post-capture — entering contest mode to guard")
                await self.actions.close_popup()
                return BotState.CONTESTING

            self._contest_until = 0.0
            await self.actions.close_popup()
            return BotState.OPENING_MINIMAP

        # Not captured yet — are there more defenders to fight?
        if info.action_button.visible and info.action_button.action_type in ("attack", "unknown"):
            # Check if next defender is beatable
            for defender in info.defenders:
                if defender.status == "active":
                    if not self._can_beat_defender(defender, ctx, config):
                        ctx.log_action(
                            f"Next defender '{defender.name}' "
                            f"(power={defender.power}) is unbeatable — moving on"
                        )
                        self._contest_until = 0.0
                        await self.actions.close_popup()
                        return BotState.OPENING_MINIMAP
                    break  # only check the first active defender

            # Store the next defender before attacking
            self._attacking_defender = None
            for defender in info.defenders:
                if defender.status == "active" and defender.name:
                    self._attacking_defender = defender.name
                    break
            ctx.log_action("More defenders to fight — attacking next")
            return BotState.ATTACKING

        ctx.log_action("Post-battle: no attack available — moving on")
        self._contest_until = 0.0
        await self.actions.close_popup()
        return BotState.OPENING_MINIMAP

    def _should_enter_contest(self, slot: int, ctx: BotContext, config: dict) -> bool:
        """Decide whether to enter contest mode for a monument slot.

        Enter contest when:
        - Just captured (< 30s ago)
        - High flip velocity (above threshold)
        - Recent flip (within recent_flip_seconds)
        - Vulnerable friendly with recent check
        """
        if slot not in ctx.monument_tracker:
            return False

        contest_cfg = config.get("contest", {})
        rec = ctx.monument_tracker[slot]
        now = time.time()

        # Just captured
        if rec.captured_at and (now - rec.captured_at) < 30:
            return True

        # High flip velocity
        vel_threshold = contest_cfg.get("flip_velocity_threshold", 2.0)
        if rec.flip_velocity >= vel_threshold:
            return True

        # Recent flip
        recent_secs = contest_cfg.get("recent_flip_seconds", 120)
        if rec.last_flip_time > 0 and (now - rec.last_flip_time) < recent_secs:
            return True

        # Vulnerable friendly
        power_threshold = contest_cfg.get("power_vulnerability_threshold", 5000)
        if (rec.last_status == "friendly"
                and rec.garrison_power >= 0
                and rec.garrison_power < power_threshold
                and rec.last_checked > 0
                and (now - rec.last_checked) < 60):
            return True

        return False

    async def handle_contesting(self, ctx: BotContext, config: dict) -> BotState:
        """Rapid-poll a contested monument.

        Immediately closes and reopens the popup to read status (no long wait
        before checking). If still friendly after the read, waits poll_interval
        before the next check. Attacks immediately on enemy flip.
        Exits when the monument stabilizes or max duration is exceeded.
        """
        contest_cfg = config.get("contest", {})
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        poll_interval = contest_cfg.get("poll_interval_seconds", 18)
        max_duration = contest_cfg.get("max_duration_seconds", 300)
        stable_secs = contest_cfg.get("stable_seconds", 120)

        slot = ctx.current_target.get("slot", 0) if ctx.current_target else 0
        rec = ctx.monument_tracker.get(slot) if slot else None

        # Check exit: max duration
        if time.time() - ctx.state_enter_time > max_duration:
            ctx.log_action(f"Contest mode expired ({max_duration}s) — moving on")
            self._contest_until = 0.0
            return BotState.OPENING_MINIMAP

        # Check exit: monument stable (friendly for stable_secs)
        if rec and rec.last_status == "friendly" and rec.last_checked > 0:
            friendly_duration = time.time() - rec.last_flip_time if rec.last_flip_time > 0 else 999999
            if friendly_duration >= stable_secs:
                ctx.log_action(
                    f"Monument stable for {friendly_duration:.0f}s — exiting contest"
                )
                self._contest_until = 0.0
                return BotState.OPENING_MINIMAP

        # Close popup, reopen, read — no long wait before checking
        await self.actions.close_popup()
        await self._wait(0.5, 0.1, "contest close")

        # Tap monument to reopen popup
        x, y = self.calibrator.get_pixel("world_monument")
        if x > 0 and y > 0:
            await self.input.tap(x, y)
        else:
            await self.input.tap(self._screen_w // 2, self._screen_h // 2)
        await self._wait(timing.get("monument_popup_wait", 1.5), jitter, "contest popup wait")

        # Read popup
        png = await self.capture.capture()
        ctx.last_screenshot = png

        info = self._read_monument_ocr(png, ctx, config)
        if info is None:
            faction = config.get("bot", {}).get("faction", "Star Spirit")
            text = self._call_vision(png, "check_monument", faction_name=faction)
            ctx.stats.vision_calls += 1
            info = parse_monument_info(text)
        ctx.monument_info = info

        # Cross-check button text
        btn_text = (info.action_button.text or "").lower()
        btn_type = info.action_button.action_type
        if info.is_friendly and (btn_type == "attack" or "attack" in btn_text):
            info.is_friendly = False

        # Update tracker
        self._update_monument_tracker(slot, info, ctx, config)

        ctx.log_action(
            f"Contest poll: friendly={info.is_friendly}, "
            f"power={info.total_garrison_power}"
        )

        # Enemy detected — attack immediately
        if not info.is_friendly:
            if info.action_button.visible and info.action_button.action_type in ("attack", "unknown"):
                # Check if defender is beatable
                for defender in info.defenders:
                    if defender.status == "active":
                        if not self._can_beat_defender(defender, ctx, config):
                            ctx.log_action(
                                f"Contest: defender '{defender.name}' unbeatable — exiting"
                            )
                            self._contest_until = 0.0
                            await self.actions.close_popup()
                            return BotState.OPENING_MINIMAP
                        break

                # Set up contest timer and attack
                contest_secs = config.get("bot", {}).get("contest_duration", 300)
                self._contest_until = time.time() + contest_secs
                self._attacking_defender = None
                for defender in info.defenders:
                    if defender.status == "active" and defender.name:
                        self._attacking_defender = defender.name
                        break
                ctx.log_action("Contest: enemy flip detected — attacking!")
                return BotState.ATTACKING

        # Still friendly — wait before next poll, then loop back
        await self._wait(poll_interval, jitter, "contest poll wait")
        return BotState.CONTESTING

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
        await self._wait(timing.get("monument_popup_wait", 1.5), jitter, "popup reopen")

        return BotState.CHECKING_MONUMENT

    async def handle_reconnecting(self, ctx: BotContext, config: dict) -> BotState:
        """Handle navigation back into Alien Minefield.

        Entered when:
        - 'android_home' detected (app crashed — launch via ADB)
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

        # ── Step 0: If android_home, launch app via ADB ──────────────
        if current.screen_type == "android_home":
            ctx.log_action("Android home detected — launching app")
            launched = await self._launch_app(ctx, config)
            if not launched:
                ctx.error_message = "Failed to launch app from Android home"
                return BotState.ERROR_RECOVERY

            # Wait for the game to load
            await self._wait(20.0, jitter, "app launch loading")

            # Poll for game screen
            for attempt in range(max_retries):
                png = await self.capture.capture()
                ctx.last_screenshot = png
                text = self._call_vision(png, "identify_screen")
                ctx.stats.vision_calls += 1
                current = parse_screen_identification(text)
                ctx.log_action(f"Reconnect post-launch: screen={current.screen_type} (attempt {attempt + 1})")

                if current.screen_type in ("home_screen", "mode_select", "main_map"):
                    break
                elif current.screen_type == "daily_popup":
                    ctx.log_action("Daily popup detected — will dismiss")
                    break
                elif current.screen_type == "android_home":
                    ctx.log_action("Still on Android home — launch may have failed")
                    ctx.error_message = "App did not launch from Android home"
                    return BotState.ERROR_RECOVERY
                else:
                    await self._wait(10.0, jitter, "waiting for game after launch")
            else:
                ctx.error_message = "Reconnect failed: game not reached after app launch"
                return BotState.ERROR_RECOVERY

            # Dismiss any daily popups that appeared after launch
            await self._dismiss_daily_popups(ctx, config)
            # Re-identify screen after popup dismissal
            png, current = await self._wait_past_loading(ctx, config, "post-popup check")

        # ── Step 1: If logged_out, wait then tap Restart ─────────────
        if current.screen_type == "logged_out":
            bot_cfg = config.get("bot", {})
            delay = bot_cfg.get("logged_out_restart_delay", 600)
            m, s = divmod(delay, 60)
            h, m = divmod(m, 60)
            ctx.log_action(f"Logged out — restarting in {h}h{m:02d}m{s:02d}s")
            await self._wait(float(delay), 0, "logged out restart delay")

            png = await self.capture.capture()
            ctx.last_screenshot = png
            self._calibrate_for_screen(png, "logged_out", ctx)
            rx, ry = self.calibrator.get_pixel("restart_button")
            ctx.log_action(f"Tapping Restart button at ({rx}, {ry})")
            await self.input.tap(rx, ry)
            await self._wait(timing.get("screen_transition", 2.0), jitter, "restart tap")

            # Game needs time to load, check for updates, and log in
            ctx.log_action("Waiting for game to load after restart...")
            await self._wait(20.0, jitter, "post-restart loading")

            # Wait for home_screen
            for attempt in range(max_retries):
                png = await self.capture.capture()
                ctx.last_screenshot = png
                text = self._call_vision(png, "identify_screen")
                ctx.stats.vision_calls += 1
                current = parse_screen_identification(text)
                ctx.log_action(f"Reconnect: screen={current.screen_type} (attempt {attempt + 1})")

                if current.screen_type in ("home_screen", "mode_select"):
                    break
                elif current.screen_type == "daily_popup":
                    ctx.log_action("Daily popup detected — will dismiss")
                    break
                elif current.screen_type == "logged_out":
                    ctx.log_action("Still showing logged out popup — retapping Restart")
                    rx, ry = self.calibrator.get_pixel("restart_button")
                    await self.input.tap(rx, ry)
                    await self._wait(15.0, jitter, "retry restart loading")
                elif current.screen_type == "loading":
                    await self._wait(10.0, jitter, "waiting for load")
                else:
                    await self._wait(10.0, jitter, "waiting for home screen")
            else:
                ctx.log_action("Could not reach home screen — going to error recovery")
                ctx.error_message = "Reconnect failed: home screen not reached"
                return BotState.ERROR_RECOVERY

            # Dismiss any daily popups that appeared after restart
            await self._dismiss_daily_popups(ctx, config)
            # Re-identify screen after popup dismissal
            png, current = await self._wait_past_loading(ctx, config, "post-popup check")

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
                expected_screens=["main_map", "loading", "hibernation"],
                ctx=ctx, config=config, png=png,
            )
            # Wait for the game mode to fully load before proceeding
            png, current = await self._wait_past_loading(ctx, config, "alien minefield loading")

        ctx.log_action("Reconnection complete — reinitializing")
        return BotState.INITIALIZING

    async def handle_idle(self, ctx: BotContext, config: dict) -> BotState:
        """No targets or hibernation active. Sleep then re-check.

        Uses a random idle duration (10-120s) to avoid predictable patterns.
        On resume, checks locally whether the minimap is still open and routes
        accordingly.  Monument status only refreshes by opening a monument's
        popup, so we just clear visited_slots and retry all monuments.
        """
        if self._hibernation_seconds is not None and self._hibernation_seconds > 0:
            # Add a 30-second buffer so the timer has fully expired
            sleep_secs = self._hibernation_seconds + 30
            m, s = divmod(sleep_secs, 60)
            h, m = divmod(m, 60)
            ctx.log_action(f"Hibernation sleep — waking in {h}h{m:02d}m{s:02d}s")
            await self._wait(float(sleep_secs), 0.05, "hibernation sleep")
            self._hibernation_seconds = None
        elif self._dormant_seconds is not None and self._dormant_seconds > 0:
            sleep_secs = self._dormant_seconds + 30
            m, s = divmod(sleep_secs, 60)
            h, m = divmod(m, 60)
            ctx.log_action(f"Dormant sleep — waking in {h}h{m:02d}m{s:02d}s")
            await self._wait(float(sleep_secs), 0.05, "dormant sleep")
            self._dormant_seconds = None
        else:
            idle_secs = random.uniform(10.0, 45.0)
            ctx.log_action(f"Idle — rechecking in {idle_secs:.0f}s")
            await self._wait(idle_secs, 0.15, "idle wait")

        # Decay unbeatable list so we retry players after cooldown
        self._decay_unbeatable_list()

        # Clear visited slots so we retry all monuments (their popup is the
        # only way to get real status, so we need to reopen each one).
        self._visited_slots.clear()

        # Quick OCR check for hibernation/dormant before resuming
        png = await self.capture.capture()
        ctx.last_screenshot = png

        ocr_screen = check_screen_ocr(png)
        if ocr_screen.screen_type == "hibernation":
            secs = parse_timer_seconds(ocr_screen.timer)
            if secs and secs > 0:
                self._hibernation_seconds = secs
                ctx.log_action(f"OCR detected hibernation ({ocr_screen.timer}) — sleeping")
                if self._event_logger:
                    self._event_logger.log("hibernation_start", duration=secs, source="ocr")
                return BotState.IDLE
            else:
                ctx.log_action("OCR detected hibernation but timer unreadable — using Vision")
                return BotState.INITIALIZING
        elif ocr_screen.screen_type == "dormant_period":
            secs = parse_timer_seconds(ocr_screen.timer)
            if secs and secs > 0:
                self._dormant_seconds = secs
                ctx.log_action(f"OCR detected dormant period ({ocr_screen.timer}) — sleeping")
                if self._event_logger:
                    self._event_logger.log("dormant_start", duration=secs, source="ocr")
                return BotState.IDLE
            else:
                ctx.log_action("OCR detected dormant but timer unreadable — using Vision")
                return BotState.INITIALIZING

        # Check locally whether the minimap is still open — no Vision needed
        detection = find_minimap_squares(png)

        if detection and len(detection.squares) >= 2:
            ctx.log_action("Resuming from idle — minimap still open, re-reading")
            return BotState.READING_MINIMAP

        ctx.log_action("Resuming from idle — opening minimap")
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
        await self._wait(timing.get("error_recovery_wait", 3.0), jitter, "error recovery")

        # Strategy 1: Identify current screen (wait past loading)
        try:
            png, screen = await self._wait_past_loading(ctx, config, "error recovery")

            ctx.log_action(f"Recovery: detected {screen.screen_type}")

            if screen.screen_type in ("logged_out", "android_home", "home_screen", "mode_select"):
                ctx.log_action(f"{screen.screen_type} detected — entering reconnection flow")
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
        await self._wait(2.0, jitter, "recovery back")

        # Strategy 3: Restart from minimap
        ctx.log_action("Recovery: restarting from minimap")
        return BotState.OPENING_MINIMAP

    async def handle_stagnation_recovery(self, ctx: BotContext, config: dict) -> BotState:
        """Vision-guided recovery when the bot has been stuck without progress.

        1. Identify the current screen.
        2. If it's a known screen, route to the normal handler.
        3. If unknown, ask Vision for recovery guidance and follow its advice.
        4. Return to INITIALIZING to re-identify after taking action.
        """
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)

        ctx.log_action(
            f"Stagnation recovery attempt {ctx.stagnation_recovery_attempts}"
        )
        if self._event_logger:
            self._event_logger.log(
                "stagnation_recovery",
                attempt=ctx.stagnation_recovery_attempts,
            )

        # Take screenshot and identify screen
        png, screen = await self._wait_past_loading(ctx, config, "stagnation recovery")
        ctx.log_action(f"Stagnation recovery: screen={screen.screen_type}")

        # If it's a recognizable screen, route normally
        if screen.screen_type in ("android_home", "logged_out", "home_screen", "mode_select"):
            return BotState.RECONNECTING
        elif screen.screen_type == "minimap":
            return BotState.READING_MINIMAP
        elif screen.screen_type == "main_map":
            return BotState.OPENING_MINIMAP
        elif screen.screen_type == "monument_popup":
            return BotState.CHECKING_MONUMENT
        elif screen.screen_type == "battle_active":
            return BotState.SKIPPING_BATTLE
        elif screen.screen_type == "battle_result":
            await self._tap_ok_button(png, ctx, config)
            return BotState.POST_BATTLE
        elif screen.screen_type == "hibernation":
            return await self._enter_hibernation(screen, ctx, config)
        elif screen.screen_type == "dormant_period":
            return self._enter_dormant(screen, ctx)
        elif screen.screen_type == "occupy_prompt":
            await self._dismiss_occupy_prompt(png, ctx, config)
            return BotState.INITIALIZING

        # Unknown screen — ask Vision for guidance
        ctx.log_action("Unknown screen — requesting recovery guidance from Vision")
        text = self._call_vision(png, "recovery_guidance")
        ctx.stats.vision_calls += 1
        guidance = parse_recovery_guidance(text)

        ctx.log_action(
            f"Recovery guidance: {guidance.suggested_action} "
            f"({guidance.diagnosis}, confidence={guidance.confidence:.2f})"
        )

        if guidance.suggested_action == "tap" and guidance.confidence >= 0.3:
            tx = int(guidance.tap_x_percent / 100 * self._screen_w)
            ty = int(guidance.tap_y_percent / 100 * self._screen_h)
            ctx.log_action(f"Recovery: tapping ({tx}, {ty}) — {guidance.tap_description}")
            await self.input.tap(tx, ty)
            await self._wait(timing.get("screen_transition", 2.0), jitter, "recovery tap")
        elif guidance.suggested_action == "back":
            ctx.log_action("Recovery: pressing back")
            await self.actions.press_back()
            await self._wait(2.0, jitter, "recovery back")
        elif guidance.suggested_action == "wait":
            ctx.log_action("Recovery: waiting for loading")
            await self._wait(10.0, jitter, "recovery wait")
        elif guidance.suggested_action == "launch_app":
            ctx.log_action("Recovery: launching app")
            await self._launch_app(ctx, config)
            await self._wait(20.0, jitter, "recovery app launch")
        else:
            ctx.log_action(f"Recovery: no actionable guidance ({guidance.suggested_action})")

        return BotState.INITIALIZING

    async def handle_paused(self, ctx: BotContext, config: dict) -> BotState:
        """Do nothing while paused. StateMachine handles resume."""
        await asyncio.sleep(1.0)  # not self._wait() — must not interrupt while paused
        return BotState.PAUSED

    async def handle_stopped(self, ctx: BotContext, config: dict) -> BotState:
        """Terminal state."""
        return BotState.STOPPED
