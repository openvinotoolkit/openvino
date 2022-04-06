// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


namespace vpu {

//
// VPU_COMBINE
//

#define VPU_COMBINE_HELPER2(X, Y)  X##Y
#define VPU_COMBINE_HELPER3(X, Y, Z)  X##Y##Z

#define VPU_COMBINE(X, Y)   VPU_COMBINE_HELPER2(X, Y)
#define VPU_COMBINE3(X, Y, Z)   VPU_COMBINE_HELPER3(X, Y, Z)

//
// Packed structure declaration
//

#ifdef _MSC_VER
#   define VPU_PACKED(body) __pragma(pack(push, 1)) struct body __pragma(pack(pop))
#elif defined(__GNUC__)
#   define VPU_PACKED(body) struct __attribute__((packed)) body
#endif

}  // namespace vpu
