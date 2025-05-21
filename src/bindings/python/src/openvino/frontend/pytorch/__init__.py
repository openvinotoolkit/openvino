# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Package: openvino.

Low level wrappers for the FrontEnd C++ API.
"""

try:
    from openvino.frontend.pytorch.py_pytorch_frontend import _FrontEndPytorchDecoder as Decoder
    from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType
    from openvino.frontend.pytorch.py_pytorch_frontend import ConversionExtensionPytorch as ConversionExtension
    from openvino.frontend.pytorch.py_pytorch_frontend import OpExtensionPytorch as OpExtension
    from openvino.frontend.pytorch.module_extension import ModuleExtension
    from openvino.frontend.pytorch.inlined_extension import inlined_extension
except ImportError as err:
    raise ImportError(f"OpenVINO PyTorch frontend is not available, please make sure the frontend is built. {err}")
