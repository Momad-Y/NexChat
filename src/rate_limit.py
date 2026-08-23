REQUEST_LIMIT = 20
REQUEST_COUNT_KEY = "request_count"

RATE_LIMIT_MESSAGE = (
    "This session has reached its request limit for now — refresh the page to reset it."
)


def has_capacity(session_state: dict, limit: int = REQUEST_LIMIT) -> bool:
    return session_state.get(REQUEST_COUNT_KEY, 0) < limit


def record_request(session_state: dict) -> None:
    session_state[REQUEST_COUNT_KEY] = session_state.get(REQUEST_COUNT_KEY, 0) + 1


def remaining_requests(session_state: dict, limit: int = REQUEST_LIMIT) -> int:
    return max(0, limit - session_state.get(REQUEST_COUNT_KEY, 0))
