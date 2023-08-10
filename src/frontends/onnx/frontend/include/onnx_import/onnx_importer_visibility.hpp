// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

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
