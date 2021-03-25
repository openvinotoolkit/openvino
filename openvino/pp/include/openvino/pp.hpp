// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
#define OV_PP_ARG_N(_0, _1, _2, _3, _4, N, ...) N
#define OV_PP_RSEQ_N() 0, 4, 3, 2, 1, 0
#define OV_PP_NO_ARGS(NAME) ,,,,

// Macros for names concatenation
#define OV_PP_CAT_(x, y) x ## y
#define OV_PP_CAT(x, y) OV_PP_CAT_(x, y)
#define OV_PP_CAT3_(x, y, z) x ## y ## z
#define OV_PP_CAT3(x, y, z) OV_PP_CAT3_(x, y, z)
#define OV_PP_CAT4_(x, y, z, w) x ## y ## z ## w
#define OV_PP_CAT4(x, y, z, w) OV_PP_CAT4_(x, y, z, w)

#define OV_PP_OVERLOAD(NAME, ...) OV_PP_EXPAND( OV_PP_CAT3(NAME, _, OV_PP_EXPAND( OV_PP_NARG(OV_PP_NO_ARGS __VA_ARGS__ (NAME)) ))(__VA_ARGS__) )
