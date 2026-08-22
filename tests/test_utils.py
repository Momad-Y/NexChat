def test_custom_message_generator_yields_words_without_delay():
    from utils import custom_message_generator
    import time as time_module

    start = time_module.time()
    chunks = list(custom_message_generator("hello world"))
    elapsed = time_module.time() - start

    assert chunks == ["hello ", "world "]
    assert elapsed < 0.05


def test_read_pdf_handles_pages_with_no_extractable_text(monkeypatch):
    from utils import read_pdf

    class FakePage:
        def extract_text(self):
            return None

    class FakeReader:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def pages(self):
            return [FakePage()]

    monkeypatch.setattr("utils.PyPDF2.PdfReader", FakeReader)

    result = read_pdf(object())

    assert result == "\n"
