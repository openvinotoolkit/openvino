// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

// Increment each time when FrontEnd/InputModel/Place interface is changed
// v2: added FrontEndPluginInfo::m_hidden (struct layout change) — bump so a plugin built against
//     the v1 struct is rejected by the loader instead of being read out-of-bounds.
#define OV_FRONTEND_API_VERSION 2

#if defined(OPENVINO_STATIC_LIBRARY)
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
#    endif  // openvino_frontend_common_EXPORTS
#endif      // OPENVINO_STATIC_LIBRARY
