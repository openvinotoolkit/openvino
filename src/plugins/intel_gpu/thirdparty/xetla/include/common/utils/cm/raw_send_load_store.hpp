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
#include "../../../common/utils/cm/common.hpp"
#include "../../../common/utils/cm/limitation.hpp"
#else
#include "common/utils/cm/common.hpp"
#include "common/utils/cm/limitation.hpp"
#endif

namespace gpu::xetla {

/// @addtogroup xetla_util_tensor_load_store
/// @{

/// @} xetla_util_tensor_load_store

/// @addtogroup xetla_util_tensor_load_store
/// @{

/// @brief Tensor descriptor construction(global memory version).
/// Constructs a tensor descriptor based on the given arguments.
/// @tparam Ty is the data type per element.
/// @tparam block_width is the width of the block to be loaded.
/// @tparam block_height is the height of the block to be loaded.
/// @tparam array_len is the array length of the block to be loaded.
/// @param tdesc [in|out] is the reference of tensor descriptor.
/// @param p [in] is the base address pointer of the tensor.
/// @param tensor_width [in] is the width of the tensor.
/// @param tensor_height [in] is the height of the tensor.
/// @param tensor_pitch [in] is the pitch(physical width of tensor in memory).
/// @param offset_x [in] is the x coordinate of the start point.
/// @param offset_y [in] is the y coordinate of the start point.
///
template <typename Ty, uint32_t block_width = 1, uint32_t block_height = 1,
        uint8_t array_len = 1>
__XETLA_API void xetla_fill_tdesc(xetla_tdescriptor_ref tdesc, Ty *p,
        int tensor_width, int tensor_height, int tensor_pitch, int offset_x,
        int offset_y) {
    detail::xetla_set_tensor_base_address(tdesc, (uint64_t)p);
    detail::xetla_set_tensor_width_x(tdesc, tensor_width * sizeof(Ty) - 1);
    detail::xetla_set_tensor_width_y(tdesc, tensor_height - 1);
    detail::xetla_set_tensor_pitch_x(tdesc, tensor_pitch * sizeof(Ty) - 1);
    detail::xetla_set_tensor_offset_x(tdesc, offset_x);
    detail::xetla_set_tensor_offset_y(tdesc, offset_y);
    uint32_t block_widthx_widthy_arrlen = (block_width - 1)
            | ((block_height - 1) << 8) | ((array_len - 1) << 16);
    detail::xetla_set_block_widthx_widthy_arrlen(
            tdesc, block_widthx_widthy_arrlen);
}

/// @brief Tensor descriptor construction(local memory version).
/// Constructs a tensor descriptor based on the given arguments, keep the same format as the global memory version.
/// @tparam Ty is the data type per element.
/// @param tdesc [in|out] is the reference of tensor descriptor.
/// @param base_address [in] is the local memory base address of the tensor.
/// @param tensor_width [in] is the width of the tensor.
/// @param tensor_height [in] is the height of the tensor.
/// @param tensor_pitch [in] is the pitch(physical width of tensor in memory).
/// @param offset_x [in] is the x coordinate of the start point.
/// @param offset_y [in] is the y coordinate of the start point.
///
template <typename Ty>
__XETLA_API void xetla_fill_tdesc(xetla_tdescriptor_ref tdesc,
        uint32_t base_address, int tensor_width, int tensor_height,
        int tensor_pitch, int offset_x, int offset_y) {
    detail::xetla_set_tensor_base_address(tdesc, base_address);
    detail::xetla_set_tensor_width_x(tdesc, tensor_width * sizeof(Ty));
    detail::xetla_set_tensor_width_y(tdesc, tensor_height);
    detail::xetla_set_tensor_pitch_x(tdesc, tensor_pitch * sizeof(Ty));
    detail::xetla_set_tensor_offset_x(tdesc, offset_x);
    detail::xetla_set_tensor_offset_y(tdesc, offset_y);
}

/// @brief Generate a new tensor descriptor(global memory version).
/// Generate a tensor descriptor based on the given arguments.
/// @tparam Ty is the data type per element.
/// @tparam block_width is the width of the block to be loaded.
/// @tparam block_height is the height of the block to be loaded.
/// @tparam array_len is the array length of the block to be loaded.
/// @param p [in] is the base address pointer of the tensor.
/// @param tensor_width [in] is the width of the tensor.
/// @param tensor_height [in] is the height of the tensor.
/// @param tensor_pitch [in] is the pitch(physical width of tensor in memory).
/// @param offset_x [in] is the x coordinate of the start point.
/// @param offset_y [in] is the y coordinate of the start point.
/// @return return a new tensor
///
template <typename Ty, uint32_t block_width = 1, uint32_t block_height = 1,
        uint8_t array_len = 1>
__XETLA_API xetla_tdescriptor xetla_get_tdesc(Ty *p, int tensor_width,
        int tensor_height, int tensor_pitch, int offset_x, int offset_y) {
    xetla_tdescriptor tdesc;
    auto tdesc_ref = tdesc.xetla_format<uint32_t>();
    detail::xetla_set_tensor_base_address(tdesc_ref, (uint64_t)p);
    detail::xetla_set_tensor_width_x(tdesc_ref, tensor_width * sizeof(Ty) - 1);
    detail::xetla_set_tensor_width_y(tdesc_ref, tensor_height - 1);
    detail::xetla_set_tensor_pitch_x(tdesc_ref, tensor_pitch * sizeof(Ty) - 1);
    detail::xetla_set_tensor_offset_x(tdesc_ref, offset_x);
    detail::xetla_set_tensor_offset_y(tdesc_ref, offset_y);
    uint32_t block_widthx_widthy_arrlen = (block_width - 1)
            | ((block_height - 1) << 8) | ((array_len - 1) << 16);
    detail::xetla_set_block_widthx_widthy_arrlen(
            tdesc_ref, block_widthx_widthy_arrlen);
    return tdesc;
}

/// @brief Generate a new tensor descriptor(local memory version).
/// Constructs a tensor descriptor based on the given arguments, keep the same format as the global memory version.
/// @tparam Ty is the data type per element.
/// @param base_address [in] is the local memory base address of the tensor.
/// @param tensor_width [in] is the width of the tensor.
/// @param tensor_height [in] is the height of the tensor.
/// @param tensor_pitch [in] is the pitch(physical width of tensor in memory).
/// @param offset_x [in] is the x coordinate of the start point.
/// @param offset_y [in] is the y coordinate of the start point.
/// @return return a new tensor descriptor
///
template <typename Ty>
__XETLA_API xetla_tdescriptor xetla_get_tdesc(uint32_t base_address,
        int tensor_width, int tensor_height, int tensor_pitch, int offset_x,
        int offset_y) {
    xetla_tdescriptor tdesc;
    auto tdesc_ref = tdesc.xetla_format<uint32_t>();
    detail::xetla_set_tensor_base_address(tdesc_ref, base_address);
    detail::xetla_set_tensor_width_x(tdesc_ref, tensor_width * sizeof(Ty));
    detail::xetla_set_tensor_width_y(tdesc_ref, tensor_height);
    detail::xetla_set_tensor_pitch_x(tdesc_ref, tensor_pitch * sizeof(Ty));
    detail::xetla_set_tensor_offset_x(tdesc_ref, offset_x);
    detail::xetla_set_tensor_offset_y(tdesc_ref, offset_y);
    return tdesc;
}

/// @brief Update the x coordinate in the given tensor descriptor.
/// @param tdesc [in|out] is the reference of tensor descriptor.
/// @param doffset_x [in] is the offset (in number of data elements) in x direction.
__XETLA_API void xetla_update_tdesc_offsetx(
        xetla_tdescriptor_ref tdesc, int32_t doffset_x) {
    detail::xetla_set_tensor_offset_x(
            tdesc, detail::xetla_get_tensor_offset_x(tdesc) + doffset_x);
}

/// @brief Update the y coordinate in the given tensor descriptor.
/// @param tdesc [in|out] is the reference of tensor descriptor.
/// @param doffset_y [in] is the offset (in number of data elements) in y direction.
__XETLA_API void xetla_update_tdesc_offsety(
        xetla_tdescriptor_ref tdesc, int32_t doffset_y) {
    detail::xetla_set_tensor_offset_y(
            tdesc, detail::xetla_get_tensor_offset_y(tdesc) + doffset_y);
}

/// @brief Update the base in the given tensor descriptor.
/// @param tdesc [in|out] is the reference of tensor descriptor.
/// @param base_offset [in] is the offset to update base address.
__XETLA_API void xetla_update_tdesc_base_address(
        xetla_tdescriptor_ref tdesc, int32_t base_offset) {
    detail::xetla_set_tensor_base_address(
            tdesc, detail::xetla_get_tensor_base_address(tdesc) + base_offset);
}

/// @brief Set width in the given tensor descriptor.
/// @tparam Ty is the data type per element.
/// @param tdesc [in|out] is the reference of tensor descriptor.
/// @param size [in] is the new width.
template <typename Ty>
__XETLA_API void xetla_set_tdesc_width(
        xetla_tdescriptor_ref tdesc, int32_t size) {
    detail::xetla_set_tensor_width_x(tdesc, size * sizeof(Ty) - 1);
}

/// @brief Set pitch in the given tensor descriptor.
/// @tparam Ty is the data type per element.
/// @param tdesc [in|out] is the reference of tensor descriptor.
/// @param size [in] is the new pitch.
template <typename Ty>
__XETLA_API void xetla_set_tdesc_pitch(
        xetla_tdescriptor_ref tdesc, int32_t size) {
    detail::xetla_set_tensor_pitch_x(tdesc, size * sizeof(Ty) - 1);
}

/// @brief Set height in the given tensor descriptor.
/// @param tdesc [in|out] is the reference of tensor descriptor.
/// @param size [in] is the new height.
__XETLA_API void xetla_set_tdesc_height(
        xetla_tdescriptor_ref tdesc, int32_t size) {
    detail::xetla_set_tensor_width_y(tdesc, size - 1);
}

/// @brief mask tdesc load_store op. If mask is 1 tdesc is unchanged. If mask is 0 then using this
/// tdesc for load would produce zero result, using this tdesc for store preforms no mem update.
/// @param tdesc [in|out] is the reference of tensor descriptor.
/// @param mask [in] is the mask.
__XETLA_API void xetla_tdesc_mask_op(
        xetla_tdescriptor_ref tdesc, uint16_t mask) {
    // +1 here since width is -1 encoded in tdesc
    uint32_t width = detail::xetla_get_tensor_width_x(tdesc) + 1;

    // adding width to offset_x if mask is 0 would produce zero on load and no memory update
    // on store

    // if mask is 0 then mask-1 is 0xFFFFFFFF and doing & with width update the offset.
    // if mask is 1 then mask-1 is 0x00000000 and doing & with width keeps tdesc as is.
    xetla_update_tdesc_offsetx(tdesc, width & (uint32_t)(mask - 1));
}

///
/// @brief Tensor load API.
/// This is tensor load API from global to registers.
/// @tparam Ty is the data type per element.
/// @tparam N is the total number of elements to load.
/// @tparam L1H is L1$ cache hint.
/// @tparam L2H is L2$ cache hint.
/// @tparam transpose is a flag to indicate whether the data is transposed during load.
/// @tparam transform is a flag to indicate whether the data is transformed (data pack inside dword) during load.
/// @param tdesc [in] is tensor descriptor including tensor base address, tensor dimensions, block size, etc.
/// @return __XETLA_API xetla_vector is data returned from the load.
///
template <typename Ty, uint32_t N, cache_hint L1H = cache_hint::none,
        cache_hint L2H = cache_hint::none, bool transpose = false,
        bool transform = false, gpu_arch arch_tag = gpu_arch::Xe>
__XETLA_API std::enable_if_t<arch_tag == gpu_arch::Xe, xetla_vector<Ty, N>>
xetla_tload_global(xetla_tdescriptor tdesc) {

    constexpr uint32_t numDst = 31 < ((N * sizeof(Ty) + 63) / 64)
            ? 31
            : ((N * sizeof(Ty) + 63) / 64);
    uint32_t msg_desc = 3;
    msg_desc |= (transform ? 1 : 0) << 7;
    msg_desc |= detail::get_element_size_code<sizeof(Ty)>() << 9;
    msg_desc |= (transpose ? 1 : 0) << 15;
    msg_desc |= detail::get_load_cache_hint_code<L1H, L2H, arch_tag>() << 17;
    msg_desc |= 1 << 25;
    msg_desc |= numDst << 20;

    constexpr uint32_t numSrc0 = 1;
    constexpr uint32_t execSize = 0;
    constexpr uint32_t sfid = 0xF;
    constexpr uint32_t exDesc = 0;

    constexpr uint32_t ret_N = (N * sizeof(Ty)) >= 32 ? N : 32 / sizeof(Ty);
    xetla_vector<Ty, ret_N> ret;

    xetla_raw_send<Ty, ret_N, uint32_t, 16, execSize, sfid, numSrc0, numDst>(
            ret.xetla_format<native_type_t<Ty>>(), tdesc, exDesc, msg_desc);

    return ret.xetla_select<N, 1>(0);
}

///
/// @brief Tensor store API.
/// Tensor store API is to store a n-d (e.g. n=2) tensor into global using tensor descriptor.
/// @tparam Ty is the data type per element.
/// @tparam N is the number of elements to store.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @param tdesc [in] is tensor descriptor including tensor base address, tensor dimensions, block size, etc.
/// @param data [in] is tensor data to store.
/// @return __XETLA_API none.
///
template <typename Ty, uint32_t N, cache_hint L1H = cache_hint::none,
        cache_hint L2H = cache_hint::none, gpu_arch arch_tag = gpu_arch::Xe>
__XETLA_API std::enable_if_t<arch_tag == gpu_arch::Xe, void>
xetla_tstore_global(xetla_tdescriptor tdesc, xetla_vector<Ty, N> data) {

    uint32_t msg_desc = 7; // store operation
    msg_desc |= detail::get_element_size_code<sizeof(Ty)>() << 9;
    msg_desc |= detail::get_store_cache_hint_code<L1H, L2H, arch_tag>() << 17;
    msg_desc |= 1 << 25;

    constexpr uint32_t numSrc1 = (N * sizeof(Ty) + 63) / 64;
    constexpr uint32_t numSrc0 = 1;
    constexpr uint32_t execSize = 0;
    constexpr uint32_t sfid = 0xF;
    constexpr uint32_t exDesc = 0;

    xetla_raw_send<uint32_t, 16, Ty, N, execSize, sfid, numSrc0, numSrc1>(
            tdesc, data, exDesc, msg_desc);
}

///
/// @brief Tensor prefetch API.
/// This is tensor prefetch API from global memory to L1$/L2$.
/// @tparam Ty is the data type per element.
/// @tparam L1H is L1$ cache hit.
/// @tparam L2H is L2$ cache hit.
/// @param tdesc is tensor descriptor including tensor base address, tensor dimensions, block size, etc.
/// @return __XETLA_API none.
///
template <typename Ty, cache_hint L1H = cache_hint::cached,
        cache_hint L2H = cache_hint::cached, gpu_arch arch_tag = gpu_arch::Xe>
__XETLA_API std::enable_if_t<arch_tag == gpu_arch::Xe, void>
xetla_tprefetch_global(xetla_tdescriptor tdesc, uint16_t pred = 1) {

    uint32_t msg_desc = 3;
    msg_desc |= 0 << 7;
    msg_desc |= detail::get_element_size_code<sizeof(Ty)>() << 9;
    msg_desc |= 0 << 15;
    msg_desc |= detail::get_prefetch_cache_hint_code<L1H, L2H, arch_tag>()
            << 17;
    msg_desc |= 1 << 25;

    constexpr uint32_t numSrc0 = 1;
    constexpr uint32_t execSize = 0;
    constexpr uint32_t sfid = 0xF;
    constexpr uint32_t exDesc = 0;

    xetla_mask<16> mask = pred;

    xetla_raw_send<uint32_t, 16, execSize, sfid, numSrc0>(
            tdesc, exDesc, msg_desc, mask);
}

///
/// @brief Tensor atomic store API.
/// Tensor atomic store API is to store a n-d (e.g. n=2) tensor into global.
/// @tparam Ty is the data type per element.
/// @tparam N is the number of elements to store.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam Toffset is the offset data type.
/// @param base_address [in] is the 64bit base address of the surface.
/// @param offset [in] is the address offset for each channel, default is 32bits.
/// @param data [in] is tensor data to store.
/// @return none.
///
template <typename Ty, uint32_t N, cache_hint L1H = cache_hint::none,
        cache_hint L2H = cache_hint::none, atomic_op Op,
        gpu_arch arch_tag = gpu_arch::Xe, typename Toffset = uint32_t>
__XETLA_API std::enable_if_t<arch_tag == gpu_arch::Xe, void>
xetla_tatomic_store_global(uint64_t base_address,
        xetla_vector<Toffset, N> offset, xetla_vector<Ty, N> data,
        xetla_mask<N> pred = 1) {

    constexpr uint32_t numSrc0 = (N * sizeof(uint64_t) + 63) / 64;
    constexpr uint32_t numSrc1 = (N * sizeof(Ty) + 63) / 64;
    constexpr uint32_t num_dest = (N * sizeof(Ty) + 63) / 64;

    // disable u16 for now, since there is no usage
    static_assert(
            sizeof(Ty) == 4 || sizeof(Ty) == 8, "element_size not supported!");
    uint32_t element_size_code;
    if constexpr (sizeof(Ty) == 4) {
        element_size_code = 2;
    } else if constexpr (sizeof(Ty) == 8) {
        element_size_code = 3;
    }

    uint32_t msg_desc = detail::get_atomic_opcode<Op>();
    ///only support 64bit address
    msg_desc |= 3 << 7;
    msg_desc |= element_size_code << 9;
    msg_desc |= detail::get_atomic_cache_hint_code<L1H, L2H, arch_tag>() << 17;
    msg_desc |= numSrc0 << 25;

    constexpr uint32_t execSize = gpu::xetla::detail::get_execSize_code<N>();
    constexpr uint32_t sfid = 0xF;
    constexpr uint32_t exDesc = 0;

    xetla_vector<uint64_t, N> address = base_address + offset;

    xetla_raw_send<uint64_t, N, Ty, N, execSize, sfid, numSrc0, numSrc1>(
            address, data, exDesc, msg_desc, pred);
}

/// @} xetla_util_tensor_load_store

} // namespace gpu::xetla
