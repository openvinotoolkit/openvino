# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Imported GPU ports must survive without retaining their CompiledModel."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("path", ["compile", "cache", "import"])
@pytest.mark.parametrize("retained", ["inputs", "outputs", "result_keys"])
def test_gpu_ports_after_compiled_model_release(device, tmp_path, path, retained):
    if not device.startswith("GPU"):
        pytest.skip("GPU import port ownership regression")
    # A fresh interpreter prevents another test's Core from keeping the GPU
    # library loaded and masking a dangling port's shared-pointer control block.
    code = textwrap.dedent("""
        import gc
        from copy import copy
        import sys
        import weakref
        import numpy as np
        import openvino as ov
        from openvino import opset13 as op

        device, cache, path, retained = sys.argv[1:]
        parameter = op.parameter([1, 4], ov.Type.f32, name="input")
        parameter.output(0).get_tensor().set_names({"input_tensor"})
        activation = op.relu(parameter)
        activation.output(0).get_tensor().set_names({"output_tensor"})
        model = ov.Model([activation], [parameter])
        config = {"INFERENCE_PRECISION_HINT": ov.Type.f32, "NUM_STREAMS": "1"}
        core = ov.Core()
        if path == "cache":
            config["CACHE_DIR"] = cache
            cold = core.compile_model(model, device, config)
            assert not cold.get_property("LOADED_FROM_CACHE")
            del cold, core
            gc.collect()
            core = ov.Core()
            compiled = core.compile_model(model, device, config)
            assert compiled.get_property("LOADED_FROM_CACHE"), "warm load recompiled"
        elif path == "import":
            cold = core.compile_model(model, device, config)
            blob = cold.export_model()
            del cold, core
            gc.collect()
            core = ov.Core()
            compiled = core.import_model(blob, device, config)
        else:
            compiled = core.compile_model(model, device, config)
        values = np.array([[-2, -1, 1, 2]], dtype=np.float32)
        result = compiled(values)
        np.testing.assert_array_equal(next(iter(result.values())), np.maximum(values, 0))
        ports = list(result) if retained == "result_keys" else list(getattr(compiled, retained))
        # Copy the port value, as the C++ API does, without a Python property's
        # reference_internal association to the original CompiledModel wrapper.
        ports = [copy(port) for port in ports]
        compiled_ref = weakref.ref(compiled)
        expected_name = "input_tensor" if retained == "inputs" else "output_tensor"
        del result, compiled, core, model, parameter, activation
        gc.collect()
        assert compiled_ref() is None
        print("Core and CompiledModel released", flush=True)
        assert len(ports) == 1
        assert ports[0].get_partial_shape() == ov.PartialShape([1, 4])
        assert ports[0].get_element_type() == ov.Type.f32
        assert expected_name in ports[0].get_names()
        print("port metadata read", flush=True)
        ports.clear()
        gc.collect()
        print("ports released after Core and CompiledModel", flush=True)
    """)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, "-c", code, device, str(tmp_path / "cache"), path, retained],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stdout}\n{result.stderr}"
    assert "ports released after Core and CompiledModel" in result.stdout
