// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/visibility.hpp"

// Increment each time when FrontEnd/InputModel/Place interface is changed
#define OV_FRONTEND_API_VERSION 1

#if defined(USE_STATIC_FRONTEND_MANAGER) || defined(OPENVINO_STATIC_LIBRARY)
#    define FRONTEND_API
#else
// Defined if cmake is building the frontend_manager DLL (instead of using it)
#    ifdef frontend_manager_EXPORTS
#        define FRONTEND_API OPENVINO_CORE_EXPORTS
#    else
#        define FRONTEND_API OPENVINO_CORE_IMPORTS
#    endif  // frontend_manager_EXPORTS
#endif      // USE_STATIC_FRONTEND_MANAGER || OPENVINO_STATIC_LIBRARY
