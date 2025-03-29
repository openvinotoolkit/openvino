// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines a macro to get the name of a function
 * @file function_name.hpp
 */

#pragma once

#if defined(__GNUC__) || (defined(__ICC) && (__ICC >= 600))
#    define ITT_FUNCTION_NAME __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
#    define ITT_FUNCTION_NAME __FUNCSIG__
#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600))
#    define ITT_FUNCTION_NAME __FUNCTION__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
#    define ITT_FUNCTION_NAME __func__
#elif defined(_MSC_VER) && (_MSC_VER >= 1900) /* VS2015 */
#    define ITT_FUNCTION_NAME __func__
#elif defined(__cplusplus) && (__cplusplus >= 201103)
#    define ITT_FUNCTION_NAME __func__
#else
#    error "Function name is N/A"
#endif
