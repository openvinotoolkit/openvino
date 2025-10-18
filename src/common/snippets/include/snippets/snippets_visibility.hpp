// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

/**
 * @file snippets_visibility.hpp
 * @brief Defines visibility settings for OpenVINO Snippets library
 */

#if defined(OPENVINO_STATIC_LIBRARY) || defined(_WIN32)
#    define SNIPPETS_API
#else
#    ifdef IMPLEMENT_OPENVINO_API
#        define SNIPPETS_API OPENVINO_CORE_EXPORTS
#    else
#        define SNIPPETS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY
