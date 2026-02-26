"""Tests for app crash recovery and stagnation auto-pause."""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.bot.state_machine import BotState, BotContext, StateMachine
from src.vision.parser import parse_recovery_guidance, RecoveryGuidance


class TestCheckStagnation:
    """Tests for StateMachine._check_stagnation()."""

    def _make_sm(self, **bot_overrides):
        bot_cfg = {
            "stagnation_timeout_seconds": 1800,
            "max_recovery_attempts": 5,
            "recovery_interval_seconds": 120,
        }
        bot_cfg.update(bot_overrides)
        return StateMachine({"bot": bot_cfg})

    def test_stagnation_fires_when_no_progress(self):
        sm = self._make_sm(stagnation_timeout_seconds=10)
        sm._running = True
        sm.state = BotState.OPENING_MINIMAP
        sm.context.last_progress_time = time.time() - 20  # 20s ago, threshold is 10s
        assert sm._check_stagnation() is True

    def test_no_stagnation_when_progress_recent(self):
        sm = self._make_sm(stagnation_timeout_seconds=1800)
        sm._running = True
        sm.state = BotState.OPENING_MINIMAP
        sm.context.last_progress_time = time.time()  # just now
        assert sm._check_stagnation() is False

    def test_exempt_states_skip_stagnation(self):
        sm = self._make_sm(stagnation_timeout_seconds=10)
        sm._running = True
        sm.context.last_progress_time = time.time() - 20

        for exempt_state in (
            BotState.PAUSED, BotState.STOPPED, BotState.IDLE,
            BotState.STAGNATION_RECOVERY, BotState.RECONNECTING,
        ):
            sm.state = exempt_state
            assert sm._check_stagnation() is False, f"Should be exempt: {exempt_state}"

    def test_rate_limited(self):
        sm = self._make_sm(stagnation_timeout_seconds=10, recovery_interval_seconds=120)
        sm._running = True
        sm.state = BotState.OPENING_MINIMAP
        sm.context.last_progress_time = time.time() - 20
        sm.context.last_stagnation_recovery_time = time.time() - 10  # only 10s ago
        assert sm._check_stagnation() is False

    def test_max_attempts_auto_pauses(self):
        sm = self._make_sm(stagnation_timeout_seconds=10, max_recovery_attempts=5)
        sm._running = True
        sm.state = BotState.OPENING_MINIMAP
        sm.context.last_progress_time = time.time() - 20
        sm.context.stagnation_recovery_attempts = 5  # already at max
        sm.context.last_stagnation_recovery_time = 0.0

        result = sm._check_stagnation()
        assert result is False  # returns False because it pauses instead
        assert sm.state == BotState.PAUSED
        assert sm.is_paused

    def test_progress_resets_stagnation(self):
        sm = self._make_sm(stagnation_timeout_seconds=1800)
        sm._running = True
        sm.state = BotState.OPENING_MINIMAP
        # Progress just happened
        sm.context.last_progress_time = time.time()
        sm.context.stagnation_recovery_attempts = 3
        assert sm._check_stagnation() is False


class TestParseRecoveryGuidance:
    def test_valid_json(self):
        text = '''
        {
            "diagnosis": "Game is on Android home screen",
            "suggested_action": "launch_app",
            "tap_target": {
                "x_percent": 45.0,
                "y_percent": 67.0,
                "description": "Capybara Go app icon"
            },
            "confidence": 0.85
        }
        '''
        result = parse_recovery_guidance(text)
        assert isinstance(result, RecoveryGuidance)
        assert result.diagnosis == "Game is on Android home screen"
        assert result.suggested_action == "launch_app"
        assert result.tap_x_percent == 45.0
        assert result.tap_y_percent == 67.0
        assert result.tap_description == "Capybara Go app icon"
        assert result.confidence == 0.85

    def test_missing_tap_target(self):
        text = '{"diagnosis": "blank screen", "suggested_action": "wait", "confidence": 0.5}'
        result = parse_recovery_guidance(text)
        assert result.suggested_action == "wait"
        assert result.tap_x_percent == 50  # default
        assert result.tap_y_percent == 50  # default

    def test_defaults_on_empty(self):
        text = '{}'
        result = parse_recovery_guidance(text)
        assert result.suggested_action == "give_up"
        assert result.confidence == 0

    def test_markdown_fenced_json(self):
        text = '''```json
        {
            "diagnosis": "popup visible",
            "suggested_action": "tap",
            "tap_target": {"x_percent": 50, "y_percent": 80, "description": "OK button"},
            "confidence": 0.9
        }
        ```'''
        result = parse_recovery_guidance(text)
        assert result.suggested_action == "tap"
        assert result.confidence == 0.9


class TestAndroidHomeRouting:
    """Test that android_home screen type routes correctly in handle_initializing."""

    @pytest.mark.asyncio
    async def test_android_home_routes_to_reconnecting(self):
        """Verify handle_initializing routes android_home → RECONNECTING."""
        from src.bot.states import StateHandlers

        # Build a minimal StateHandlers with mocked dependencies
        handlers = StateHandlers.__new__(StateHandlers)
        handlers.capture = AsyncMock()
        handlers.input = AsyncMock()
        handlers.vision = MagicMock()
        handlers.cache = MagicMock()
        handlers.actions = AsyncMock()
        handlers.config = {"screen": {"width": 1080, "height": 1920}}
        handlers.calibrator = MagicMock()
        handlers.calibrator.needs_calibration.return_value = []
        handlers.element_detector = None
        handlers._state_machine = MagicMock()
        handlers._state_machine.is_paused = False
        handlers._screen_w = 1080
        handlers._screen_h = 1920

        # Mock _wait_past_loading to return android_home screen
        mock_screen = MagicMock()
        mock_screen.screen_type = "android_home"
        mock_screen.confidence = 0.9
        handlers._wait_past_loading = AsyncMock(return_value=(b"fake_png", mock_screen))

        ctx = BotContext()
        config = {"screen": {"width": 1080, "height": 1920}}

        result = await handlers.handle_initializing(ctx, config)
        assert result == BotState.RECONNECTING


class TestStagnationRecoveryIntegration:
    """Test the stagnation check wiring in the run loop."""

    def test_stagnation_increments_attempts(self):
        sm = StateMachine({"bot": {
            "stagnation_timeout_seconds": 10,
            "max_recovery_attempts": 5,
            "recovery_interval_seconds": 0,
        }})
        sm._running = True
        sm.state = BotState.OPENING_MINIMAP
        sm.context.last_progress_time = time.time() - 20

        assert sm._check_stagnation() is True
        # Simulate what the run loop does
        sm.context.stagnation_recovery_attempts += 1
        sm.context.last_stagnation_recovery_time = time.time()
        assert sm.context.stagnation_recovery_attempts == 1

    def test_check_stuck_exempts_stagnation_recovery(self):
        sm = StateMachine({"bot": {"stuck_timeout": 1}})
        sm.state = BotState.STAGNATION_RECOVERY
        sm.context.state_enter_time = time.time() - 100  # way past timeout
        assert sm._check_stuck() is False

    def test_get_status_includes_stagnation_fields(self):
        sm = StateMachine({})
        sm.context.last_progress_time = 12345.0
        sm.context.stagnation_recovery_attempts = 3
        status = sm.get_status()
        assert status["last_progress_time"] == 12345.0
        assert status["stagnation_recovery_attempts"] == 3
