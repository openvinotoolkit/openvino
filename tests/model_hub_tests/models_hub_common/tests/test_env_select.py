# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for the `env:<name>` tag stripping in models_hub_common.utils.get_models_list, and the
# selection/tagging logic in models_hub_common.env_select. Uses plain duck-typed stand-ins for
# pytest's Item/CallSpec (a real Item needs a live pytest Session/Config) instead of spinning up a
# real pytest run.
#
# Not part of any CI-invoked test path (not under pytorch/ or transformation_tests/); run directly,
# e.g.: PYTHONPATH=tests/model_hub_tests <venv>/bin/python -m pytest
#       tests/model_hub_tests/models_hub_common/tests/test_env_select.py -q
import pytest

from models_hub_common import env_select, utils


# --- get_models_list / MODEL_ENV_TAGS ---

def _write_list(tmp_path, content):
    path = tmp_path / "models_list"
    path.write_text(content)
    return str(path)


def test_env_tag_stripped_from_2_column_line(tmp_path):
    utils.MODEL_ENV_TAGS.clear()
    path = _write_list(tmp_path, "some/model,https://example.com/some/model,env:legacy\n")
    models = utils.get_models_list(path)
    assert models == [("some/model", "https://example.com/some/model", None, None)]
    assert utils.MODEL_ENV_TAGS == {"some/model": "legacy"}


def test_env_tag_stripped_from_4_column_line(tmp_path):
    utils.MODEL_ENV_TAGS.clear()
    path = _write_list(tmp_path, "m,link,skip,some reason,env:legacy\n")
    models = utils.get_models_list(path)
    assert models == [("m", "link", "skip", "some reason")]
    assert utils.MODEL_ENV_TAGS == {"m": "legacy"}


def test_env_tag_stripped_from_greater_than_4_column_line(tmp_path):
    utils.MODEL_ENV_TAGS.clear()
    path = _write_list(tmp_path, "m,link,,,ts_name:Foo,layer:Bar,env:legacy\n")
    models = utils.get_models_list(path)
    assert models == [("m", "link", None, None, ["Foo"], ["Bar"], None)]
    assert utils.MODEL_ENV_TAGS == {"m": "legacy"}


def test_untagged_line_does_not_populate_registry(tmp_path):
    utils.MODEL_ENV_TAGS.clear()
    path = _write_list(tmp_path, "m,link\n")
    utils.get_models_list(path)
    assert utils.MODEL_ENV_TAGS == {}


def test_two_models_can_have_different_tags(tmp_path):
    utils.MODEL_ENV_TAGS.clear()
    path = _write_list(tmp_path, "a,link_a,env:legacy\nb,link_b,env:gptq\nc,link_c\n")
    utils.get_models_list(path)
    assert utils.MODEL_ENV_TAGS == {"a": "legacy", "b": "gptq"}
    assert "c" not in utils.MODEL_ENV_TAGS


# --- env_select.resolve_item_env ---

class _FakeCallSpec:
    def __init__(self, params):
        self.params = params


class _FakeItem:
    def __init__(self, params=None):
        self.callspec = _FakeCallSpec(params or {})
        self.user_properties = []


def test_resolve_item_env_finds_tag_inside_tuple_param(monkeypatch):
    monkeypatch.setitem(utils.MODEL_ENV_TAGS, "some/model", "legacy")
    item = _FakeItem(params={"model_info_tuple": (object(), "some/model", "link", None, None)})
    assert env_select.resolve_item_env(item) == "legacy"


def test_resolve_item_env_does_not_substring_match(monkeypatch):
    # regression guard: "minicpm" must not match a param value of "minicpm3"/"minicpmv-2_6" etc.
    monkeypatch.setitem(utils.MODEL_ENV_TAGS, "optimum-intel-internal-testing/tiny-random-minicpm", "legacy")
    item = _FakeItem(params={"model_info_tuple": (object(), "optimum-intel-internal-testing/tiny-random-minicpm3", "link", None, None)})
    assert env_select.resolve_item_env(item) is None


def test_resolve_item_env_untagged_returns_none():
    item = _FakeItem(params={"model_info_tuple": (object(), "untagged/model", "link", None, None)})
    assert env_select.resolve_item_env(item) is None


# --- env_select.pytest_collection_modifyitems ---

class _FakeHook:
    def __init__(self):
        self.deselected_calls = []

    def pytest_deselected(self, items):
        self.deselected_calls.append(list(items))


class _FakeConfig:
    def __init__(self):
        self.hook = _FakeHook()


def test_collection_modifyitems_noop_when_env_unset(monkeypatch):
    monkeypatch.delenv("OV_TEST_ENV", raising=False)
    monkeypatch.setattr(env_select, "OV_TEST_ENV", None)
    items = [_FakeItem(), _FakeItem()]
    env_select.pytest_collection_modifyitems(_FakeConfig(), items)
    assert len(items) == 2  # nothing deselected, no user_properties added
    assert items[0].user_properties == []


def test_collection_modifyitems_keeps_matching_env_deselects_others(monkeypatch):
    monkeypatch.setattr(env_select, "OV_TEST_ENV", "legacy")
    monkeypatch.setattr(env_select, "OV_TEST_DEFAULT_ENV", "main")
    monkeypatch.setitem(utils.MODEL_ENV_TAGS, "tagged/model", "legacy")

    legacy_item = _FakeItem(params={"model_info_tuple": (object(), "tagged/model", "l", None, None)})
    default_item = _FakeItem(params={"model_info_tuple": (object(), "untagged/model", "l", None, None)})

    config = _FakeConfig()
    items = [legacy_item, default_item]
    env_select.pytest_collection_modifyitems(config, items)

    assert items == [legacy_item]
    assert config.hook.deselected_calls == [[default_item]]
    assert ("ov_test_env", "legacy") in legacy_item.user_properties


def test_collection_modifyitems_default_env_run_keeps_untagged_only(monkeypatch):
    monkeypatch.setattr(env_select, "OV_TEST_ENV", "main")
    monkeypatch.setattr(env_select, "OV_TEST_DEFAULT_ENV", "main")
    monkeypatch.setitem(utils.MODEL_ENV_TAGS, "tagged/model", "legacy")

    legacy_item = _FakeItem(params={"model_info_tuple": (object(), "tagged/model", "l", None, None)})
    default_item = _FakeItem(params={"model_info_tuple": (object(), "untagged/model", "l", None, None)})

    config = _FakeConfig()
    items = [legacy_item, default_item]
    env_select.pytest_collection_modifyitems(config, items)

    assert items == [default_item]
    assert config.hook.deselected_calls == [[legacy_item]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
