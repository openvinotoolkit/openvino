// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/visibility.hpp"

#ifdef ov_onnx_frontend_EXPORTS
#    define ONNX_IMPORTER_API OPENVINO_CORE_EXPORTS
#else
#    define ONNX_IMPORTER_API OPENVINO_CORE_IMPORTS
#endif
