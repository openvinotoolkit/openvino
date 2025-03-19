/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#ifdef _WIN32
#include "../../../common/core/cm/core.hpp"
#else
#include "common/core/cm/core.hpp"
#endif

namespace gpu::xetla {

namespace detail {
__XETLA_API void xetla_set_tensor_base_address(
        xetla_tdescriptor_ref desc, uint64_t base_address) {
    desc.xetla_format<uint64_t>().xetla_select<1, 1>(0) = base_address;
}
__XETLA_API void xetla_set_tensor_base_address(
        xetla_tdescriptor_ref desc, uint32_t base_address) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(0) = base_address;
}
__XETLA_API uint64_t xetla_get_tensor_base_address(xetla_tdescriptor desc) {
    return desc.xetla_format<uint64_t>().xetla_select<1, 1>(0)[0];
}

__XETLA_API void xetla_set_tensor_width_x(
        xetla_tdescriptor_ref desc, uint32_t width_x) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(2) = width_x;
}
__XETLA_API uint32_t xetla_get_tensor_width_x(xetla_tdescriptor desc) {
    return desc.xetla_format<uint32_t>().xetla_select<1, 1>(2)[0];
}

__XETLA_API void xetla_set_tensor_width_y(
        xetla_tdescriptor_ref desc, uint32_t width_y) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(3) = width_y;
}
__XETLA_API uint32_t xetla_get_tensor_width_y(xetla_tdescriptor desc) {
    return desc.xetla_format<uint32_t>().xetla_select<1, 1>(3)[0];
}

__XETLA_API void xetla_set_tensor_pitch_x(
        xetla_tdescriptor_ref desc, uint32_t pitch_x) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(4) = pitch_x;
}
__XETLA_API uint32_t xetla_get_tensor_pitch_x(xetla_tdescriptor desc) {
    return desc.xetla_format<uint32_t>().xetla_select<1, 1>(4)[0];
}

__XETLA_API void xetla_set_tensor_offset_x(
        xetla_tdescriptor_ref desc, int32_t offset_x) {
    desc.xetla_format<int32_t>().xetla_select<1, 1>(5) = offset_x;
}
__XETLA_API int32_t xetla_get_tensor_offset_x(xetla_tdescriptor desc) {
    return desc.xetla_format<int32_t>().xetla_select<1, 1>(5)[0];
}

__XETLA_API void xetla_set_tensor_offset_y(
        xetla_tdescriptor_ref desc, int32_t offset_y) {
    desc.xetla_format<int32_t>().xetla_select<1, 1>(6) = offset_y;
}
__XETLA_API int32_t xetla_get_tensor_offset_y(xetla_tdescriptor desc) {
    return desc.xetla_format<int32_t>().xetla_select<1, 1>(6)[0];
}

__XETLA_API void xetla_set_block_widthx_widthy_arrlen(
        xetla_tdescriptor_ref desc, uint32_t block_widthx_widthy_arrlen) {
    desc.xetla_format<uint32_t>().xetla_select<1, 1>(7)
            = block_widthx_widthy_arrlen;
}
__XETLA_API uint8_t xetla_get_block_width_x(xetla_tdescriptor desc) {
    return desc.xetla_format<uint8_t>().xetla_select<1, 1>(28)[0];
}
__XETLA_API uint8_t xetla_get_block_width_y(xetla_tdescriptor desc) {
    return desc.xetla_format<uint8_t>().xetla_select<1, 1>(29)[0];
}
__XETLA_API uint8_t xetla_get_block_array_len(xetla_tdescriptor desc) {
    return desc.xetla_format<uint8_t>().xetla_select<1, 1>(30)[0];
}
} // namespace detail

} // namespace gpu::xetla
