from humanoid_companion.speed_sweep import usable_range


def row(cmd, got, stops=()):
    return {"command_vx": cmd, "achieved_vx": got, "safety_stops": list(stops)}


def test_usable_range_is_the_tracked_positive_commands():
    rows = [row(0.0, 0.0), row(0.1, 0.001), row(0.2, 0.002), row(0.3, 0.337), row(0.5, 0.491),
            row(0.8, 0.519), row(-0.3, -0.25)]
    assert usable_range(rows) == (0.3, 0.5)


def test_standing_still_does_not_track_a_slow_command():
    assert usable_range([row(0.1, 0.001), row(0.2, 0.002)]) is None
    assert usable_range([row(0.1, 0.07)]) == (0.1, 0.1)


def test_a_safety_stop_disqualifies_a_speed_and_nothing_tracked_is_none():
    assert usable_range([row(0.3, 0.3, ["tilt"])]) is None
    assert usable_range([row(0.2, 0.0)]) is None
