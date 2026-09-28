// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/visibility.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define GGUF_FRONTEND_API
#    define GGUF_FRONTEND_C_API
#else
#    ifdef openvino_gguf_frontend_EXPORTS
#        define GGUF_FRONTEND_API   OPENVINO_CORE_EXPORTS
#        define GGUF_FRONTEND_C_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#    else
#        define GGUF_FRONTEND_API   OPENVINO_CORE_IMPORTS
#        define GGUF_FRONTEND_C_API OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS
#    endif  // openvino_gguf_frontend_EXPORTS
#endif      // OPENVINO_STATIC_LIBRARY
