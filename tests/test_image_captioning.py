def test_caption_image_calls_glm_vision_endpoint_with_key(monkeypatch):
    from cv.image_captioning import caption_image

    captured = {}

    class FakeResponse:
        def json(self):
            return {"choices": [{"message": {"content": "a photo of a cat"}}]}

        def raise_for_status(self):
            pass

    class FakeUploadedFile:
        type = "image/png"

        def getvalue(self):
            return b"fake-bytes"

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("cv.image_captioning.requests.post", fake_post)

    caption = caption_image(FakeUploadedFile(), "test-glm-key")

    assert captured["url"] == "https://api.z.ai/api/paas/v4/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-glm-key"
    assert captured["json"]["model"] == "glm-4.6v-flash"

    content = captured["json"]["messages"][0]["content"]
    image_part = next(part for part in content if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    assert caption == "A photo of a cat."


def test_caption_image_returns_error_message_on_failure(monkeypatch):
    from cv.image_captioning import caption_image

    class FakeUploadedFile:
        type = "image/jpeg"

        def getvalue(self):
            return b"fake-bytes"

    def fake_post(url, headers=None, json=None, timeout=None):
        raise RuntimeError("network error")

    monkeypatch.setattr("cv.image_captioning.requests.post", fake_post)

    caption = caption_image(FakeUploadedFile(), "test-glm-key")

    assert caption == "An error occurred while generating the caption."


def test_caption_image_defaults_mime_type_when_missing(monkeypatch):
    from cv.image_captioning import caption_image

    captured = {}

    class FakeResponse:
        def json(self):
            return {"choices": [{"message": {"content": "a shape"}}]}

        def raise_for_status(self):
            pass

    class FakeUploadedFile:
        # No `type` attribute at all — simulate an upload that doesn't report a MIME type.
        def getvalue(self):
            return b"fake-bytes"

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("cv.image_captioning.requests.post", fake_post)

    caption_image(FakeUploadedFile(), "test-glm-key")

    content = captured["json"]["messages"][0]["content"]
    image_part = next(part for part in content if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
