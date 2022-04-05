// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

// Increment each time when FrontEnd/InputModel/Place interface is changed
#define OV_FRONTEND_API_VERSION 1

#if defined(USE_STATIC_FRONTEND_COMMON) || defined(OPENVINO_STATIC_LIBRARY)
#    define FRONTEND_API
#    define FRONTEND_C_API
#else
// Defined if cmake is building the frontend_common DLL (instead of using it)
#    ifdef IMPLEMENT_OPENVINO_API
#        define FRONTEND_API   OPENVINO_CORE_EXPORTS
#        define FRONTEND_C_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#    else
#        define FRONTEND_API   OPENVINO_CORE_IMPORTS
#        define FRONTEND_C_API OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS
#    endif  // frontend_common_EXPORTS
#endif      // USE_STATIC_FRONTEND_COMMON || OPENVINO_STATIC_LIBRARY
