def test_asset_path_resolves_existing_icon():
    from paths import asset_path

    resolved = asset_path("icon.png")
    assert resolved.name == "icon.png"
    assert resolved.exists()


def test_asset_path_resolves_regardless_of_cwd(tmp_path, monkeypatch):
    from paths import asset_path

    monkeypatch.chdir(tmp_path)
    resolved = asset_path("logo.png")
    assert resolved.exists()
