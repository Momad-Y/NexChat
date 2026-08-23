def test_has_capacity_true_when_under_limit():
    from rate_limit import has_capacity

    session_state = {}
    assert has_capacity(session_state, limit=20) is True


def test_has_capacity_false_when_at_limit():
    from rate_limit import has_capacity

    session_state = {"request_count": 20}
    assert has_capacity(session_state, limit=20) is False


def test_has_capacity_false_when_over_limit():
    from rate_limit import has_capacity

    session_state = {"request_count": 21}
    assert has_capacity(session_state, limit=20) is False


def test_record_request_increments_from_zero():
    from rate_limit import record_request

    session_state = {}
    record_request(session_state)
    assert session_state["request_count"] == 1


def test_record_request_increments_existing_count():
    from rate_limit import record_request

    session_state = {"request_count": 4}
    record_request(session_state)
    assert session_state["request_count"] == 5


def test_remaining_requests_counts_down():
    from rate_limit import remaining_requests

    session_state = {"request_count": 12}
    assert remaining_requests(session_state, limit=20) == 8


def test_remaining_requests_never_negative():
    from rate_limit import remaining_requests

    session_state = {"request_count": 30}
    assert remaining_requests(session_state, limit=20) == 0


def test_default_limit_is_twenty():
    from rate_limit import has_capacity, REQUEST_LIMIT

    assert REQUEST_LIMIT == 20
    session_state = {"request_count": 19}
    assert has_capacity(session_state) is True
    session_state = {"request_count": 20}
    assert has_capacity(session_state) is False
