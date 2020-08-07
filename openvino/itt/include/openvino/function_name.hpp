//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

/**
 * @brief Defines a macro to get the name of a function
 * @file function_name.hpp
 */

#pragma once

#if defined(__GNUC__) || (defined(__ICC) && (__ICC >= 600))
    #define ITT_FUNCTION_NAME __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
    #define ITT_FUNCTION_NAME __FUNCSIG__
#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600))
    #define ITT_FUNCTION_NAME __FUNCTION__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
    #define ITT_FUNCTION_NAME __func__
#elif defined(__cplusplus) && (__cplusplus >= 201103)
    #define ITT_FUNCTION_NAME __func__
#else
    #error "Function name is N/A"
#endif
