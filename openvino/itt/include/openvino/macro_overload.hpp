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
 * @brief Macro overloading feature support
 * @file macro_overload.hpp
 */

#pragma once

#define OV_ITT_MACRO_EXPAND(X) X

#define OV_ITT_MACRO_NARG(...) OV_ITT_MACRO_EXPAND( OV_ITT_MACRO_NARG_(__VA_ARGS__, OV_ITT_MACRO_RSEQ_N()) )
#define OV_ITT_MACRO_NARG_(...) OV_ITT_MACRO_EXPAND( OV_ITT_MACRO_ARG_N(__VA_ARGS__) )
#define OV_ITT_MACRO_ARG_N(_1, _2, _3, _4, N, ...) N
#define OV_ITT_MACRO_RSEQ_N() 4, 3, 2, 1, 0

#define OV_ITT_MACRO_EVAL_(NAME, N) NAME ## _ ## N
#define OV_ITT_MACRO_EVAL(NAME, N) OV_ITT_MACRO_EVAL_(NAME, N)

#define OV_ITT_MACRO_OVERLOAD(NAME, ...) OV_ITT_MACRO_EXPAND( OV_ITT_MACRO_EVAL(NAME, OV_ITT_MACRO_EXPAND( OV_ITT_MACRO_NARG(__VA_ARGS__) ))(__VA_ARGS__) )
