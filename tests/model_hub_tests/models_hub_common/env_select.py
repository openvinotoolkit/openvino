# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# pytest hooks that let a single test module's parametrized test cases be split across several
# independently-installed "environments" (e.g. different pinned transformers/optimum-intel versions),
# selected by the OV_TEST_ENV environment variable.
#
# A test case's owning env is resolved from an `env:<name>` tag on the model-list line it was
# parametrized from (see models_hub_common.utils.get_models_list / MODEL_ENV_TAGS), or else
# OV_TEST_DEFAULT_ENV for untagged models. When OV_TEST_ENV is unset, every test runs and nothing is
# deselected -- this module is a no-op unless the caller opts in by setting OV_TEST_ENV.
import os

from models_hub_common.utils import MODEL_ENV_TAGS

# OV_TEST_ENV names *this* pytest invocation's environment; OV_TEST_DEFAULT_ENV names the one env that
# owns every untagged model in the suite being run and must be the SAME value across every env run of
# one test suite -- otherwise untagged items would be deselected from every env, or selected into more
# than one.
OV_TEST_ENV = os.environ.get("OV_TEST_ENV")
OV_TEST_DEFAULT_ENV = os.environ.get("OV_TEST_DEFAULT_ENV", "main")


def resolve_item_env(item):
    """Returns the env name this test item belongs to, or None if untagged (caller decides the
    default). Scans item.callspec.params (which for these tests holds the model_info_tuple / model
    name values) for a value present in MODEL_ENV_TAGS."""
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    for value in callspec.params.values():
        candidates = value if isinstance(value, (tuple, list)) else (value,)
        for v in candidates:
            if isinstance(v, str) and v in MODEL_ENV_TAGS:
                return MODEL_ENV_TAGS[v]
    return None


def pytest_collection_modifyitems(config, items):
    if not OV_TEST_ENV:
        return

    kept = []
    deselected = []
    for item in items:
        owner = resolve_item_env(item) or OV_TEST_DEFAULT_ENV
        if owner == OV_TEST_ENV:
            item.user_properties.append(("ov_test_env", OV_TEST_ENV))
            kept.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept
