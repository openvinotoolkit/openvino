// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

/**
 * @file lpt_visibility.hpp
 * @brief Defines visibility settings for OpenVINO LP Transformations library
 */

#ifdef OPENVINO_STATIC_LIBRARY
#    define LP_TRANSFORMATIONS_API
#else
#    ifdef IMPLEMENT_OPENVINO_API
#        define LP_TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    else
#        define LP_TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY
