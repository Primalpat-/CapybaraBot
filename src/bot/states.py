"""State handlers for each bot state."""

import logging

from src.adb.capture import ScreenCapture
from src.adb.input import ADBInput
from src.bot.actions import BotActions
from src.bot.state_machine import BotState, BotContext
from src.bot.strategy import select_next_monument
from src.utils.timing import wait
from src.vision.cache import VisionCache
from src.vision.client import VisionClient
from src.vision.parser import (
    parse_screen_identification,
    parse_minimap_reading,
    parse_monument_info,
    parse_navigation_check,
    parse_battle_check,
    parse_post_battle,
)
from src.vision.prompts import get_prompt

logger = logging.getLogger(__name__)


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
    ):
        self.capture = capture
        self.input = adb_input
        self.vision = vision
        self.cache = cache
        self.actions = actions
        self.config = config
        self._visited_monument_ids: set[int] = set()
        self._screen_w = config.get("screen", {}).get("width", 1080)
        self._screen_h = config.get("screen", {}).get("height", 1920)

    def _call_vision(self, png_bytes: bytes, prompt_name: str) -> str:
        """Call vision API with caching."""
        cached = self.cache.get(png_bytes, prompt_name)
        if cached is not None:
            return cached

        system, prompt = get_prompt(prompt_name)
        response = self.vision.analyze_screenshot(png_bytes, prompt, system)
        self.cache.put(png_bytes, prompt_name, response.text)
        return response.text

    async def handle_initializing(self, ctx: BotContext, config: dict) -> BotState:
        """Identify current screen and decide where to go."""
        ctx.log_action("Initializing — identifying screen")
        png = await self.capture.capture()
        ctx.last_screenshot = png

        text = self._call_vision(png, "identify_screen")
        ctx.stats.api_calls += 1
        screen = parse_screen_identification(text)

        ctx.log_action(f"Screen: {screen.screen_type} (conf={screen.confidence:.2f})")

        if screen.screen_type == "minimap":
            return BotState.READING_MINIMAP
        elif screen.screen_type == "monument_popup":
            return BotState.CHECKING_MONUMENT
        elif screen.screen_type == "battle_active":
            return BotState.SKIPPING_BATTLE
        elif screen.screen_type == "battle_result":
            return BotState.POST_BATTLE
        else:
            return BotState.OPENING_MINIMAP

    async def handle_opening_minimap(self, ctx: BotContext, config: dict) -> BotState:
        """Open the minimap overlay."""
        ctx.log_action("Opening minimap")
        await self.actions.open_minimap()

        png = await self.capture.capture()
        ctx.last_screenshot = png

        text = self._call_vision(png, "identify_screen")
        ctx.stats.api_calls += 1
        screen = parse_screen_identification(text)

        if screen.screen_type == "minimap":
            return BotState.READING_MINIMAP

        ctx.log_action("Minimap not detected, retrying...")
        return BotState.OPENING_MINIMAP

    async def handle_reading_minimap(self, ctx: BotContext, config: dict) -> BotState:
        """Read monument positions from the minimap."""
        ctx.log_action("Reading minimap")
        png = await self.capture.capture()
        ctx.last_screenshot = png

        text = self._call_vision(png, "read_minimap")
        ctx.stats.api_calls += 1
        minimap = parse_minimap_reading(text)
        ctx.minimap_data = minimap

        ctx.log_action(f"Found {minimap.total_monuments_visible} monuments")

        target = select_next_monument(minimap, self._visited_monument_ids)
        if target is None:
            ctx.log_action("No more targets — going idle")
            return BotState.IDLE

        ctx.current_target = {
            "id": target.id,
            "x_percent": target.x_percent,
            "y_percent": target.y_percent,
            "type": target.likely_type,
        }
        ctx.log_action(f"Target: monument {target.id} ({target.likely_type})")

        # Tap the target on the minimap
        await self.actions.tap_monument_on_minimap(
            target.x_percent, target.y_percent,
            self._screen_w, self._screen_h,
        )

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

        if nav.arrived or nav.monument_popup_visible:
            ctx.log_action("Arrived at monument")
            ctx.stats.monuments_visited += 1
            if ctx.current_target:
                self._visited_monument_ids.add(ctx.current_target["id"])
            return BotState.CHECKING_MONUMENT

        ctx.log_action("Still navigating...")
        return BotState.NAVIGATING

    async def handle_checking_monument(self, ctx: BotContext, config: dict) -> BotState:
        """Read monument popup and decide what to do."""
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        await wait(timing.get("monument_popup_wait", 1.5), jitter, "popup wait")

        png = await self.capture.capture()
        ctx.last_screenshot = png

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
            ctx.log_action(f"Attacking! (button: {info.action_button.text})")
            return BotState.ATTACKING

        ctx.log_action("No attack available — moving on")
        await self.actions.close_popup()
        return BotState.OPENING_MINIMAP

    async def handle_attacking(self, ctx: BotContext, config: dict) -> BotState:
        """Tap the attack button — mechanical, no vision needed."""
        await self.actions.tap_action_button()
        ctx.stats.battles_fought += 1
        ctx.log_action("Attack initiated")

        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        await wait(timing.get("screen_transition", 2.0), jitter, "battle start")
        return BotState.SKIPPING_BATTLE

    async def handle_skipping_battle(self, ctx: BotContext, config: dict) -> BotState:
        """Poll for battle end, tapping skip/speed-up."""
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)

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
            # Tap continue if visible
            if battle.continue_button_visible:
                await self.actions.tap_action_button()
            return BotState.POST_BATTLE

        ctx.log_action("Battle still in progress...")
        return BotState.SKIPPING_BATTLE

    async def handle_post_battle(self, ctx: BotContext, config: dict) -> BotState:
        """Check monument state after battle."""
        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        await wait(timing.get("screen_transition", 2.0), jitter, "post battle wait")

        png = await self.capture.capture()
        ctx.last_screenshot = png

        text = self._call_vision(png, "post_battle")
        ctx.stats.api_calls += 1
        post = parse_post_battle(text)

        if post.monument_captured:
            ctx.stats.monuments_captured += 1
            ctx.log_action("Monument captured!")
            await self.actions.close_popup()
            return BotState.OPENING_MINIMAP

        if post.all_defenders_defeated:
            ctx.log_action("All defenders defeated — claiming")
            if post.action_button.visible:
                await self.actions.tap_action_button()
            ctx.stats.monuments_captured += 1
            await self.actions.close_popup()
            return BotState.OPENING_MINIMAP

        if post.next_action_available == "attack_next_defender":
            ctx.log_action("More defenders — refreshing popup")
            return BotState.REFRESHING_POPUP

        ctx.log_action("Post-battle: moving on")
        await self.actions.close_popup()
        return BotState.OPENING_MINIMAP

    async def handle_refreshing_popup(self, ctx: BotContext, config: dict) -> BotState:
        """Close and reopen the popup to fight the next defender."""
        await self.actions.refresh_popup()

        # Re-tap the monument to reopen its popup
        if ctx.current_target:
            await self.actions.tap_monument_on_minimap(
                ctx.current_target["x_percent"],
                ctx.current_target["y_percent"],
                self._screen_w, self._screen_h,
            )
        else:
            # Fallback: just tap center
            await self.actions.tap_action_button()

        return BotState.CHECKING_MONUMENT

    async def handle_idle(self, ctx: BotContext, config: dict) -> BotState:
        """No more monuments. Wait or stop."""
        ctx.log_action("Idle — no targets remaining")
        await wait(5.0, 0, "idle wait")
        return BotState.IDLE

    async def handle_error_recovery(self, ctx: BotContext, config: dict) -> BotState:
        """Multi-strategy error recovery."""
        ctx.log_action(f"Error recovery: {ctx.error_message}")

        timing = config.get("timing", {})
        jitter = timing.get("jitter_factor", 0.3)
        await wait(timing.get("error_recovery_wait", 3.0), jitter, "error recovery")

        # Strategy 1: Identify current screen
        try:
            png = await self.capture.capture()
            ctx.last_screenshot = png
            text = self._call_vision(png, "identify_screen")
            ctx.stats.api_calls += 1
            screen = parse_screen_identification(text)

            ctx.log_action(f"Recovery: detected {screen.screen_type}")

            if screen.screen_type == "minimap":
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
