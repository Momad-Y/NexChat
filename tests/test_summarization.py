def test_summarize_text_calls_router_endpoint_with_key(monkeypatch):
    from nlp.summarization import summarize_text

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"summary_text": "a short summary"}]

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("nlp.summarization.requests.post", fake_post)

    chunks = list(summarize_text("some text to summarize", "test-hf-key"))

    assert (
        captured["url"]
        == "https://router.huggingface.co/hf-inference/models/pszemraj/led-large-book-summary"
    )
    assert captured["headers"]["Authorization"] == "Bearer test-hf-key"
    assert captured["json"]["parameters"]["max_new_tokens"] == 180
    assert "".join(chunks)


def test_summarize_text_yields_error_message_on_request_failure(monkeypatch):
    from nlp.summarization import summarize_text

    def fake_post(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr("nlp.summarization.requests.post", fake_post)

    chunks = list(summarize_text("some text", "test-hf-key"))

    assert chunks == ["An error occurred while generating the summary."]


def test_summarize_text_chunked_path_yields_error_on_failure(monkeypatch):
    from nlp.summarization import summarize_text, MAX_CHUNK_SIZE

    def fake_post(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr("nlp.summarization.requests.post", fake_post)

    long_text = "word " * ((MAX_CHUNK_SIZE // 5) + 100)  # exceeds MAX_CHUNK_SIZE
    assert len(long_text) > MAX_CHUNK_SIZE
    chunks = list(summarize_text(long_text, "test-hf-key"))

    assert chunks == ["An error occurred while generating the summary."]


def test_summarize_text_chunked_path_sends_multiple_requests(monkeypatch):
    from nlp.summarization import summarize_text, MAX_CHUNK_SIZE

    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"summary_text": "chunk summary"}]

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return FakeResponse()

    monkeypatch.setattr("nlp.summarization.requests.post", fake_post)

    long_text = "word " * ((MAX_CHUNK_SIZE // 5) + 100)
    chunks = list(summarize_text(long_text, "test-hf-key"))

    assert len(calls) >= 2
    assert all(call["parameters"]["max_new_tokens"] == 180 for call in calls)
    assert "".join(chunks)
