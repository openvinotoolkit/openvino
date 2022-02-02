// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/visibility.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define ONNX_IMPORTER_API
#else
#    ifdef openvino_onnx_frontend_EXPORTS
#        define ONNX_IMPORTER_API OPENVINO_CORE_EXPORTS
#    else
#        define ONNX_IMPORTER_API OPENVINO_CORE_IMPORTS
#    endif  // openvino_onnx_frontend_EXPORTS
#endif      // OPENVINO_STATIC_LIBRARY
