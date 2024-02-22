// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/exception.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define TENSORFLOW_API
#    define TENSORFLOW_C_API
#else
#    ifdef openvino_tensorflow_frontend_EXPORTS
#        define TENSORFLOW_API   OPENVINO_CORE_EXPORTS
#        define TENSORFLOW_C_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#    else
#        define TENSORFLOW_API   OPENVINO_CORE_IMPORTS
#        define TENSORFLOW_C_API OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS
#    endif  // openvino_tensorflow_frontend_EXPORTS
#endif      // OPENVINO_STATIC_LIBRARY
