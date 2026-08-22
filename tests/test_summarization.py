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
        return FakeResponse()

    monkeypatch.setattr("nlp.summarization.requests.post", fake_post)

    chunks = list(summarize_text("some text to summarize", "test-hf-key"))

    assert captured["url"] == "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
    assert captured["headers"]["Authorization"] == "Bearer test-hf-key"
    assert "".join(chunks)


def test_summarize_text_yields_error_message_on_request_failure(monkeypatch):
    from nlp.summarization import summarize_text

    def fake_post(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr("nlp.summarization.requests.post", fake_post)

    chunks = list(summarize_text("some text", "test-hf-key"))

    assert chunks == ["An error occurred while generating the summary."]
