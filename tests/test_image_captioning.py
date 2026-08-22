def test_caption_image_calls_router_endpoint_with_key(monkeypatch):
    from cv.image_captioning import caption_image

    captured = {}

    class FakeResponse:
        def json(self):
            return [{"generated_text": "a photo of a cat"}]

    class FakeUploadedFile:
        def getvalue(self):
            return b"fake-bytes"

    def fake_post(url, headers=None, data=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("cv.image_captioning.requests.post", fake_post)

    caption = caption_image(FakeUploadedFile(), "test-hf-key")

    assert captured["url"] == "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-base"
    assert captured["headers"]["Authorization"] == "Bearer test-hf-key"
    assert caption == "A photo of a cat."
