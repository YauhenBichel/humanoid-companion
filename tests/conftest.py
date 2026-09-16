import pytest


@pytest.fixture(autouse=True)
def isolated_settings_folder(tmp_path_factory, monkeypatch):
    """Tests never read the settings or teammates of the person running them."""
    monkeypatch.setenv("HUMANOID_CONFIG_DIR", str(tmp_path_factory.mktemp("config")))
