def test_pytest_can_import_src():
    from utils import get_file_extension

    assert get_file_extension("report.pdf") == "pdf"
