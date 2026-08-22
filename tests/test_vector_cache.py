from nlp.vector_cache import compute_files_fingerprint, get_cached_vector_store, store_vector_store


class FakeUploadedFile:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


def test_fingerprint_is_stable_for_identical_files():
    files_a = [FakeUploadedFile("doc.pdf", b"hello world")]
    files_b = [FakeUploadedFile("doc.pdf", b"hello world")]
    assert compute_files_fingerprint(files_a, "test-huggingface-key") == compute_files_fingerprint(
        files_b, "test-huggingface-key"
    )


def test_fingerprint_changes_when_content_changes():
    files_a = [FakeUploadedFile("doc.pdf", b"hello world")]
    files_b = [FakeUploadedFile("doc.pdf", b"different content")]
    assert compute_files_fingerprint(files_a, "test-huggingface-key") != compute_files_fingerprint(
        files_b, "test-huggingface-key"
    )


def test_fingerprint_changes_when_file_set_changes():
    one_file = [FakeUploadedFile("doc.pdf", b"hello world")]
    two_files = one_file + [FakeUploadedFile("doc2.pdf", b"more")]
    assert compute_files_fingerprint(one_file, "test-huggingface-key") != compute_files_fingerprint(
        two_files, "test-huggingface-key"
    )


def test_fingerprint_changes_when_key_changes():
    files = [FakeUploadedFile("doc.pdf", b"hello world")]
    assert compute_files_fingerprint(files, "key-a") != compute_files_fingerprint(files, "key-b")


def test_fingerprint_stable_for_same_files_and_key():
    files_a = [FakeUploadedFile("doc.pdf", b"hello world")]
    files_b = [FakeUploadedFile("doc.pdf", b"hello world")]
    assert compute_files_fingerprint(files_a, "same-key") == compute_files_fingerprint(files_b, "same-key")


def test_cache_roundtrip_within_session():
    session_state = {}
    fingerprint = "abc123"
    sentinel_store = object()

    assert get_cached_vector_store(session_state, fingerprint) is None

    store_vector_store(session_state, fingerprint, sentinel_store)

    assert get_cached_vector_store(session_state, fingerprint) is sentinel_store


def test_cache_miss_when_fingerprint_differs():
    session_state = {}
    store_vector_store(session_state, "fingerprint-a", object())

    assert get_cached_vector_store(session_state, "fingerprint-b") is None


def test_fingerprint_distinguishes_ambiguous_name_content_boundary():
    files_a = [FakeUploadedFile("ab", b"cd")]
    files_b = [FakeUploadedFile("a", b"bcd")]
    assert compute_files_fingerprint(files_a, "test-huggingface-key") != compute_files_fingerprint(
        files_b, "test-huggingface-key"
    )
