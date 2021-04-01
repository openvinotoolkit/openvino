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
 * @brief Set of macro used by openvino
 * @file pp.hpp
 */

#pragma once

// Macros for string conversion
#define OV_PP_TOSTRING(...) OV_PP_TOSTRING_(__VA_ARGS__)
#define OV_PP_TOSTRING_(...) #__VA_ARGS__

#define OV_PP_EXPAND(X) X

#define OV_PP_NARG(...) OV_PP_EXPAND( OV_PP_NARG_(__VA_ARGS__, OV_PP_RSEQ_N()) )
#define OV_PP_NARG_(...) OV_PP_EXPAND( OV_PP_ARG_N(__VA_ARGS__) )
#define OV_PP_ARG_N(_1, _2, _3, _4, N, ...) N
#define OV_PP_RSEQ_N() 4, 3, 2, 1, 0

// Macros for names concatenation
#define OV_PP_CAT_(x, y) x ## y
#define OV_PP_CAT(x, y) OV_PP_CAT_(x, y)
#define OV_PP_CAT3_(x, y, z) x ## y ## z
#define OV_PP_CAT3(x, y, z) OV_PP_CAT3_(x, y, z)
#define OV_PP_CAT4_(x, y, z, w) x ## y ## z ## w
#define OV_PP_CAT4(x, y, z, w) OV_PP_CAT4_(x, y, z, w)


#define OV_PP_OVERLOAD(NAME, ...) OV_PP_EXPAND( OV_PP_CAT3(NAME, _, OV_PP_EXPAND( OV_PP_NARG(__VA_ARGS__) ))(__VA_ARGS__) )
