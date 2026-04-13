import concurrent.futures
import importlib.util
import sys
import types
from pathlib import Path


class _StubProcessPoolExecutor:

    def __init__(self, *args, **kwargs):
        pass


concurrent.futures.ProcessPoolExecutor = _StubProcessPoolExecutor

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / 'src'
HUB_MODULE_PATH = SRC_DIR / 'twinkle' / 'hub' / 'hub.py'

twinkle_pkg = types.ModuleType('twinkle')
twinkle_pkg.__path__ = [str(SRC_DIR / 'twinkle')]
sys.modules.setdefault('twinkle', twinkle_pkg)

twinkle_utils_pkg = types.ModuleType('twinkle.utils')
twinkle_utils_pkg.requires = lambda package: None
sys.modules.setdefault('twinkle.utils', twinkle_utils_pkg)

twinkle_hub_pkg = types.ModuleType('twinkle.hub')
twinkle_hub_pkg.__path__ = [str(SRC_DIR / 'twinkle' / 'hub')]
sys.modules.setdefault('twinkle.hub', twinkle_hub_pkg)

spec = importlib.util.spec_from_file_location('twinkle.hub.hub', HUB_MODULE_PATH)
hub_module = importlib.util.module_from_spec(spec)
sys.modules['twinkle.hub.hub'] = hub_module
spec.loader.exec_module(hub_module)

HubOperation = hub_module.HubOperation


def test_download_model_short_circuits_existing_local_path(tmp_path, monkeypatch):
    local_model_dir = tmp_path / 'local-model'
    local_model_dir.mkdir()

    def _unexpected_get_hub_class(*args, **kwargs):
        raise AssertionError('Local model directories must not resolve a hub backend.')

    monkeypatch.setattr(HubOperation, '_get_hub_class', _unexpected_get_hub_class)

    assert HubOperation.download_model(str(local_model_dir), ignore_model=True) == str(local_model_dir.resolve())


def test_download_model_keeps_remote_repo_flow(monkeypatch):
    called = {}

    class DummyHub:

        @staticmethod
        def download_model(**kwargs):
            called.update(kwargs)
            return '/tmp/downloaded-model'

    monkeypatch.setattr(HubOperation, '_get_hub_class', lambda resource_name: DummyHub)

    result = HubOperation.download_model('ms://Qwen/Qwen3.5-4B', revision='main', token='secret-token')

    assert result == '/tmp/downloaded-model'
    assert called['model_id_or_path'] == 'Qwen/Qwen3.5-4B'
    assert called['revision'] == 'main'
    assert called['token'] == 'secret-token'
