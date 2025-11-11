# Copyright (C) 2018-2025 Intel
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestQuantile(PytorchLayerTest):
    def _prepare_input(self):
        # Returns the inputs for the Torch model and for OpenVINO
        return (self.input_tensor, self.q_tensor)

    def create_model(self, dim, keepdim, interpolation):
        class AtenQuantile(torch.nn.Module):
            def __init__(self, dim, keepdim, interpolation) -> None:
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim
                self.interpolation = interpolation

            def forward(self, x, q):
                return torch.quantile(
                    x,
                    q,
                    dim=self.dim,
                    keepdim=self.keepdim,
                    interpolation=self.interpolation,
                )

        return AtenQuantile(dim, keepdim, interpolation), None, "aten::quantile"

    # --------------------------
    # General functional test with broad parameterization
    # --------------------------
    @pytest.mark.parametrize("input_shape", [
        (10,),
        (4, 5),
        (2, 3, 6, 9),
    ])
    @pytest.mark.parametrize("q", [
        0.0,
        0.5,
        1.0,
        [0.0, 0.25, 0.5, 0.75, 1.0],
    ])
    @pytest.mark.parametrize("dim", [None, -1, 0,1,2])
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.parametrize("interpolation", ["linear", "nearest", "lower", "higher", "midpoint"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantile(
        self,
        input_shape,
        q,
        dim,
        keepdim,
        interpolation,
        ie_device,
        precision,
        ir_version,
    ):
        # Set random seeds for reproducibility
        np.random.seed(0)
        torch.manual_seed(0)

        # Generate input data (rounded continuous random values)
        # Scale factor determines distribution range
        x = torch.round(torch.randn(*input_shape) * 5.0).to(torch.float32)

        # Skip invalid 'dim' values for the given input shape
        if dim is not None:
            rank = x.dim()
            if dim < -rank or dim >= rank:
                pytest.skip(f"Invalid dim={dim} for shape={input_shape}")

        q_t = torch.tensor(q, dtype=torch.float32)

        # Store inputs for the harness runner
        self.input_tensor = x.detach().cpu().numpy()
        self.q_tensor = q_t.detach().cpu().numpy()

        # Skip test if GPU is requested but not available
        import openvino as ov
        print("OV Python path:", ov.__file__)
        try:
            core = ov.Core()
            if str(ie_device).upper() == "GPU" and "GPU" not in core.available_devices:
                pytest.skip("GPU device is not available in this environment")
        except Exception:
            pass

        # Define numeric tolerance (set to 0 for strict comparison)
        eps = 0.0

        # Run the core test with conversion and comparison
        self._test(
            *self.create_model(dim, keepdim, interpolation),
            ie_device,
            precision,
            ir_version,
            trace_model=True,
            use_convert_model=True,
            dynamic_shapes=False,
            custom_eps=eps,
            ignore_nan=True,  # Ignore NaN mismatches between frameworks
        )

    # ==============================
    # A) Edge cases with small or scalar inputs
    # ==============================
    @pytest.mark.parametrize("x_val", [
        0.75,                 
        [1.0],               
        [[1.0, 2.0]],        
        [[1.0], [2.0]],       
        [[[1.0, 2.0]]],      
    ])
    @pytest.mark.parametrize("q_val", [
        0.0, 
        0.5, 
        1.0,
        [0.25, 0.5, 0.75],
    ])
    @pytest.mark.parametrize("dim", [None, 0])
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.parametrize("interpolation", ["linear", "nearest", "lower", "higher", "midpoint"])
    @pytest.mark.precommit
    def test_quantile_small_and_scalar_inputs(
        self, x_val, q_val, dim, keepdim, interpolation,
        ie_device, precision, ir_version
    ):
        # Build tensors (x may be scalar 0-D)
        x = torch.tensor(x_val, dtype=torch.float32)
        q_t = torch.tensor(q_val, dtype=torch.float32)

        # Skip invalid combination: scalar input with non-None dim
        if x.ndim == 0 and dim is not None:
            pytest.skip(f"Scalar input supports only dim=None (got dim={dim})")

        self.input_tensor = x.detach().cpu().numpy()
        self.q_tensor = q_t.detach().cpu().numpy()

        # Skip test if GPU is requested but not available
        try:
            import openvino as ov
            core = ov.Core()
            if str(ie_device).upper() == "GPU" and "GPU" not in core.available_devices:
                pytest.skip("GPU device is not available in this environment")
        except Exception:
            pass

        eps = 0.0

        # Execute and compare results between Torch and OpenVINO
        self._test(
            *self.create_model(dim, keepdim, interpolation),
            ie_device, precision, ir_version,
            trace_model=True, use_convert_model=True,
            dynamic_shapes=False, custom_eps=eps, ignore_nan=True,
        )

    # =======================
    # B) Inputs containing NaN values
    # =======================
    @pytest.mark.parametrize("x_val", [
        [float('nan')],                                       
        [[float('nan'), float('nan')], [1.0, 2.0]],          
        [[float('nan'), float('nan')], [float('nan'), 2.0]], 
    ])
    @pytest.mark.parametrize("q_val", [
        0.0, 
        0.5,
        1.0,
        [0.25, 0.5, 0.75],
    ])
    @pytest.mark.parametrize("dim", [None, 0, 1])
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.parametrize("interpolation", ["linear", "nearest", "lower", "higher", "midpoint"])
    @pytest.mark.precommit
    def test_quantile_inputs_with_nan(
        self, x_val, q_val, dim, keepdim, interpolation,
        ie_device, precision, ir_version
    ):
        # Prepare input tensors containing NaNs
        x = torch.tensor(x_val, dtype=torch.float32)
        q_t = torch.tensor(q_val, dtype=torch.float32)

        self.input_tensor = x.detach().cpu().numpy()
        self.q_tensor = q_t.detach().cpu().numpy()

        # Skip test if GPU is requested but not available
        try:
            import openvino as ov
            core = ov.Core()
            if str(ie_device).upper() == "GPU" and "GPU" not in core.available_devices:
                pytest.skip("GPU device is not available in this environment")
        except Exception:
            pass

        eps = 0.0

        # Run test and compare outputs, ignoring NaN mismatches
        self._test(
            *self.create_model(dim, keepdim, interpolation),
            ie_device, precision, ir_version,
            trace_model=True, use_convert_model=True,
            dynamic_shapes=False, custom_eps=eps, ignore_nan=True,
        )
