"""
Tests for visibility checking.

The current visibility implementation is a placeholder (all events valid).
These tests verify the placeholder contract and are designed so that when
real sky-constraint logic is added, the tests describe the expected behaviour.
"""

import numpy as np
import pytest

from ariel_rl.simulator.event_generator import _check_visibility


class TestVisibilityPlaceholder:
    def test_all_visible_single_event(self):
        mid_times = np.array([2462867.5])
        valid = _check_visibility(mid_times, dur_days=0.1)
        assert valid.all()
        assert len(valid) == 1

    def test_all_visible_many_events(self):
        mid_times = np.linspace(2462867.5, 2462867.5 + 100, 200)
        valid = _check_visibility(mid_times, dur_days=0.05)
        assert valid.all()
        assert len(valid) == 200

    def test_output_is_boolean_array(self):
        mid_times = np.array([2462867.5, 2462868.5])
        valid = _check_visibility(mid_times, dur_days=0.1)
        assert valid.dtype == bool

    def test_empty_input(self):
        mid_times = np.array([], dtype=float)
        valid = _check_visibility(mid_times, dur_days=0.1)
        assert len(valid) == 0

    def test_output_length_matches_input(self):
        for n in [1, 5, 100]:
            mid_times = np.ones(n) * 2462867.5
            valid = _check_visibility(mid_times, dur_days=0.05)
            assert len(valid) == n
