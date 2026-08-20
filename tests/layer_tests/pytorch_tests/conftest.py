# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
import os

from pytorch_layer_test_class import get_params


def _patch_cia_ops_cache():
    """Cache _collect_all_valid_cia_ops to speed up run_decompositions.

    PyTorch's run_decompositions() calls _collect_all_valid_cia_ops() on every
    invocation, scanning 12K+ registered ops across 46 namespaces. The result
    is constant within a process, so caching it saves ~18ms per model conversion.
    """
    try:
        import torch._export.utils as _export_utils
        _orig_collect = _export_utils._collect_all_valid_cia_ops
        _cached_result = None

        def _cached_collect_all_valid_cia_ops():
            nonlocal _cached_result
            if _cached_result is None:
                _cached_result = _orig_collect()
            return _cached_result

        _export_utils._collect_all_valid_cia_ops = _cached_collect_all_valid_cia_ops
        import torch.export.exported_program as _ep_module
        if hasattr(_ep_module, '_collect_all_valid_cia_ops'):
            _ep_module._collect_all_valid_cia_ops = _cached_collect_all_valid_cia_ops
    except (ImportError, AttributeError):
        pass


if os.environ.get("PYTORCH_TRACING_MODE", "").upper() == "EXPORT":
    _patch_cia_ops_cache()


def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")
