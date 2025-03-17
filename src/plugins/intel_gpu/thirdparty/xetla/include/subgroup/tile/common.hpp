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

#include "common/common.hpp"

namespace gpu::xetla::subgroup {

namespace detail {

/// @brief Compute next power of 2 of a constexpr with guaranteed compile-time
/// evaluation.
///
/// @tparam N
/// @tparam K
/// @tparam K_gt_eq_N
template <uint32_t N, uint32_t K, bool K_gt_eq_N>
struct NextPowerOf2;

/// @brief
///
/// @tparam N
/// @tparam K
template <uint32_t N, uint32_t K>
struct NextPowerOf2<N, K, true> {
    static constexpr uint32_t get() { return K; }
};

/// @brief
///
/// @tparam N
/// @tparam K
template <uint32_t N, uint32_t K>
struct NextPowerOf2<N, K, false> {
    static constexpr uint32_t get() {
        return NextPowerOf2<N, K * 2, K * 2 >= N>::get();
    }
};

/// @brief Get the Next Power Of2 object
///
/// @tparam N
/// @return constexpr uint32_t
template <uint32_t N>
constexpr uint32_t getNextPowerOf2() {
    return NextPowerOf2<N, 1, (1 >= N)>::get();
}

/// @brief Get the Next Power Of2<0> object
///
/// @tparam
/// @return constexpr uint32_t
template <>
constexpr uint32_t getNextPowerOf2<0>() {
    return 0;
}

/// @brief
///
/// @tparam a
/// @tparam b
template <uint32_t a, uint32_t b>
struct gcd {
    static constexpr uint32_t value = gcd<b, a % b>::value;
};

/// @brief
///
/// @tparam a
template <uint32_t a>
struct gcd<a, 0> {
    static constexpr uint32_t value = a;
};

enum class process_flag : uint8_t { load = 0, store = 1 };

template <uint32_t remained_len, uint32_t base_len, process_flag flag,
        cache_hint L1, cache_hint L2, typename payload_t, typename tile_t>
__XETLA_API typename std::enable_if_t<base_len == 0> process_1d_tail(
        tile_t &tile, payload_t &payload, uint32_t offset) {}

template <uint32_t remained_len, uint32_t base_len, process_flag flag,
        cache_hint L1, cache_hint L2, typename payload_t, typename tile_t>
__XETLA_API typename std::enable_if_t<base_len == 0> process_1d_tail(
        tile_t &tile, payload_t &payload, uint32_t offset,
        uint32_t address_offset) {}

template <uint32_t remained_len, uint32_t base_len, process_flag flag,
        cache_hint L1, cache_hint L2, typename payload_t, typename tile_t>
__XETLA_API typename std::enable_if_t<base_len != 0
        && payload_t::memory_space == mem_space::global>
process_1d_tail(tile_t &tile, payload_t &payload, uint32_t offset) {
    using dtype = typename payload_t::dtype;
    using mem_dtype = typename payload_t::mem_dtype;
    if constexpr (remained_len >= base_len) {
        uint32_t address_offset = offset * sizeof(dtype);
        auto reg_sub
                = tile.reg.xetla_select<base_len * payload_t::scale_factor, 1>(
                        offset);
        if constexpr (flag == process_flag::load) {
            reg_sub.xetla_format<mem_dtype>() = xetla_load_global<mem_dtype,
                    base_len, data_size::default_size, L1, L2>(
                    payload.base_ptr, payload.base_offset + address_offset);
        } else {
            xetla_store_global<mem_dtype, base_len, data_size::default_size, L1,
                    L2>(payload.base_ptr, payload.base_offset + address_offset,
                    reg_sub.xetla_format<mem_dtype>());
        }
        process_1d_tail<remained_len - base_len, (base_len >> 1), flag, L1, L2>(
                tile, payload, offset + base_len * payload_t::scale_factor);
    } else {
        process_1d_tail<remained_len, (base_len >> 1), flag, L1, L2>(
                tile, payload, offset);
    }
}

template <uint32_t remained_len, uint32_t base_len, process_flag flag,
        cache_hint L1, cache_hint L2, typename payload_t, typename tile_t>
__XETLA_API typename std::enable_if_t<base_len != 0
        && payload_t::memory_space == mem_space::local>
process_1d_tail(tile_t &tile, payload_t &payload, uint32_t offset,
        uint32_t address_offset) {
    using dtype = typename payload_t::dtype;
    using mem_dtype = typename payload_t::mem_dtype;
    if constexpr (remained_len >= base_len) {
        auto reg_sub
                = tile.reg.xetla_select<base_len * payload_t::scale_factor, 1>(
                        offset);
        if constexpr (flag == process_flag::load) {
            reg_sub.xetla_format<mem_dtype>() = xetla_load_local<mem_dtype,
                    base_len, data_size::default_size>(
                    payload.base_address + payload.address + address_offset);
        } else {
            xetla_store_local<mem_dtype, base_len>(
                    payload.base_address + payload.address + address_offset,
                    reg_sub.xetla_format<mem_dtype>());
        }
        process_1d_tail<remained_len - base_len, (base_len >> 1), flag, L1, L2,
                payload_t, tile_t>(tile, payload,
                offset + base_len * payload_t::scale_factor,
                address_offset
                        + base_len * payload_t::scale_factor
                                * sizeof(typename tile_t::dtype));
    } else {
        process_1d_tail<remained_len, (base_len >> 1), flag, L1, L2, payload_t,
                tile_t>(tile, payload, offset, address_offset);
    }
}

// This will end up with base_len equal to 8 because we had made tile_size_x
// divisible by 8/16/32, depends on dtype
// this is for prefetch only and use different func arg compare with load/store
template <uint32_t remained_len, uint32_t base_len, cache_hint L1,
        cache_hint L2, typename payload_t>
__XETLA_API typename std::enable_if_t<(base_len < 8)> process_1d_tail(
        payload_t &payload, uint32_t offset) {
    using dtype = typename payload_t::dtype;
    using prefetch_dtype = typename payload_t::prefetch_dtype;
    uint32_t address_offset = offset * sizeof(dtype);
    constexpr uint32_t prefetch_min_size = 64 / sizeof(prefetch_dtype);
    if constexpr (remained_len > 0) {
        xetla_prefetch_global<prefetch_dtype, prefetch_min_size,
                data_size::default_size, L1, L2>(
                payload.base_ptr, payload.base_offset + address_offset);
    }
}

template <uint32_t remained_len, uint32_t base_len, cache_hint L1,
        cache_hint L2, typename payload_t>
__XETLA_API typename std::enable_if_t<base_len >= 8> process_1d_tail(
        payload_t &payload, uint32_t offset) {
    using dtype = typename payload_t::dtype;
    using prefetch_dtype = typename payload_t::prefetch_dtype;
    if constexpr (remained_len >= base_len) {
        uint32_t address_offset = offset * sizeof(dtype);
        xetla_prefetch_global<prefetch_dtype, base_len, data_size::default_size,
                L1, L2>(payload.base_ptr, payload.base_offset + address_offset);
        process_1d_tail<remained_len - base_len, (base_len >> 1), L1, L2,
                payload_t>(
                payload, offset + base_len * payload_t::scale_factor);
    } else {
        process_1d_tail<remained_len, (base_len >> 1), L1, L2, payload_t>(
                payload, offset);
    }
}

template <uint32_t num_tdesc, uint32_t size_x, uint32_t size_y,
        uint32_t scale_factor, uint8_t arr_len, bool trans>
__XETLA_API static void reset_tile_desc_core(
        xetla_matrix_ref<uint32_t, num_tdesc, 16> __REF__ payload_row) {
#pragma unroll
    for (int j = 0; j < num_tdesc; j++) {
        constexpr uint8_t block_width
                = trans ? (size_y / scale_factor) : (size_x / scale_factor);
        constexpr uint8_t block_height = trans ? size_x : size_y;
        constexpr uint32_t block_widthx_widthy_arrlen = (block_width - 1)
                | ((block_height - 1) << 8) | ((arr_len - 1) << 16);
        gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
                payload_row.row(j), block_widthx_widthy_arrlen);
    }
}

} // namespace detail

template <typename T_dst, typename T_src>
struct is_same_layout {
    static constexpr bool value = (T_src::block_size_y == T_dst::block_size_y)
            && (T_src::block_size_x == T_dst::block_size_x)
            && (T_src::tile_size_y == T_dst::tile_size_y)
            && (T_src::tile_size_x == T_dst::tile_size_x);
};

template <typename T_dst, typename T_src>
struct is_1d_src {
    static constexpr bool value = (T_src::tile_elems == T_dst::tile_elems)
            && (T_src::block_size_y == 1) && (T_src::tile_size_y == 1);
};

template <typename T_dst, typename T_src>
struct is_same_tile {
    static constexpr bool value = (T_src::tile_size_y == T_dst::tile_size_y)
            && (T_src::tile_size_x == T_dst::tile_size_x);
};

template <typename T_dst, typename T_src>
struct is_same_elements {
    static constexpr bool value = (T_src::tile_elems == T_dst::tile_elems);
};

template <typename T_dst, typename T_src>
struct is_floating_to_integer {
    static constexpr bool value
            = is_floating_point<typename T_src::dtype>::value
            && is_integral<typename T_dst::dtype>::value;
};

template <typename tile_desc_, mem_space memory_space,
        mem_layout memory_layout = mem_layout::row_major>
struct msg_type_query {
    static constexpr msg_type value = memory_space == mem_space::global
            ? (((tile_desc_::tile_size_y == 1)
                       && (memory_layout == mem_layout::row_major))
                            ? msg_type::block_1d
                            : msg_type::block_2d)
            : (((tile_desc_::tile_size_y == 1)
                       && (memory_layout == mem_layout::row_major))
                            ? msg_type::block_1d
                            : msg_type::scatter);
};

template <typename tile_desc_, mem_space memory_space>
constexpr msg_type msg_type_v = msg_type_query<tile_desc_, memory_space>::value;

template <typename dtype, uint32_t tile_size_x, uint32_t tile_size_y,
        gpu_arch arch_tag, mem_layout mem_layout_ = mem_layout::row_major,
        reg_layout reg_layout_ = reg_layout::tiled>
struct get_load_block_size_auto {};

template <typename dtype, uint32_t tile_size_x, uint32_t tile_size_y>
struct get_load_block_size_auto<dtype, tile_size_x, tile_size_y, gpu_arch::Xe,
        mem_layout::row_major, reg_layout::tiled> {
private:
    using load_store_attr = arch_attr_t<gpu_arch::Xe>::template load_store_attr<
            msg_type::block_2d>;
    static constexpr uint32_t max_load_height_in_elem
            = load_store_attr::max_load_height_in_elem;
    static constexpr uint32_t max_load_width_in_bytes
            = load_store_attr::max_load_width_in_bytes;
    static constexpr uint32_t max_load_width_in_elem
            = max_load_width_in_bytes / sizeof(dtype);

public:
    // block_size_x should be power of 2 and tile_size_x should be divided by block_size_x
    static constexpr uint32_t block_size_x
            = detail::gcd<tile_size_x, max_load_width_in_elem>::value;
    static constexpr uint32_t block_size_y
            = max_load_height_in_elem > tile_size_y ? tile_size_y
                                                    : max_load_height_in_elem;
};

template <typename dtype, uint32_t tile_size_x, uint32_t tile_size_y,
        gpu_arch arch_tag, mem_layout mem_layout_ = mem_layout::row_major,
        reg_layout reg_layout_ = reg_layout::tiled>
struct get_store_block_size_auto {};

template <typename dtype, uint32_t tile_size_x, uint32_t tile_size_y>
struct get_store_block_size_auto<dtype, tile_size_x, tile_size_y, gpu_arch::Xe,
        mem_layout::row_major, reg_layout::tiled> {
private:
    using load_store_attr = arch_attr_t<gpu_arch::Xe>::template load_store_attr<
            msg_type::block_2d>;
    static constexpr uint32_t max_store_height_in_elem
            = load_store_attr::max_store_height_in_elem;
    static constexpr uint32_t max_store_width_in_bytes
            = load_store_attr::max_store_width_in_bytes;
    static constexpr uint32_t max_store_width_in_elem
            = max_store_width_in_bytes / sizeof(dtype);

public:
    // block_size_x should be power of 2 and tile_size_x should be divided by block_size_x
    static constexpr uint32_t block_size_x
            = detail::gcd<tile_size_x, max_store_width_in_elem>::value;
    static constexpr uint32_t block_size_y
            = max_store_height_in_elem > tile_size_y ? tile_size_y
                                                     : max_store_height_in_elem;
};

// This type tag represents "global atomic oob check on" behavior
struct global_atomic_oob_check_on_tag : std::true_type {};

// This type tag represents "global atomic oob check off" behavior
struct global_atomic_oob_check_off_tag : std::false_type {};

} // namespace gpu::xetla::subgroup
