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

#ifndef XETLA_NO_CM_INCLUDE
#include <cm/cmtl.h>
#endif

namespace gpu::xetla {
namespace detail {

///
///@brief Get the element size code object
///
///@param element_size
///@return constexpr uint32_t
///
template <uint32_t element_size>
constexpr uint32_t get_element_size_code() {
    static_assert(element_size == 1 || element_size == 2 || element_size == 4
                    || element_size == 8,
            "element_size not supported!");
    switch (element_size) {
        case 1: return 0;
        case 2: return 1;
        case 4: return 2;
        case 8: return 3;
    }
}

enum class lsc_action { prefetch, load, store, atomic };

template <lsc_action Action, cache_hint L1H, cache_hint L2H, gpu_arch arch_tag>
constexpr std::enable_if_t<arch_tag == gpu_arch::Xe, void>
check_lsc_cache_hint() {
    if constexpr (Action == lsc_action::prefetch) {
        static_assert(
                ((L2H == cache_hint::uncached || L2H == cache_hint::cached)
                        && (L1H == cache_hint::uncached
                                || L1H == cache_hint::cached
                                || L1H == cache_hint::streaming)),
                "cache hint type not supported!");
    } else if constexpr (Action == lsc_action::load) {
        static_assert((L1H == cache_hint::none && L2H == cache_hint::none)
                        || ((L2H == cache_hint::uncached)
                                && (L1H == cache_hint::uncached
                                        || L1H == cache_hint::cached
                                        || L1H == cache_hint::streaming))
                        || ((L2H == cache_hint::cached)
                                && (L1H == cache_hint::uncached
                                        || L1H == cache_hint::cached
                                        || L1H == cache_hint::streaming
                                        || L1H == cache_hint::read_invalidate)),
                "unsupported cache hint!");
    } else if constexpr (Action == lsc_action::store) {
        static_assert((L1H == cache_hint::none && L2H == cache_hint::none)
                        || ((L2H == cache_hint::uncached)
                                && (L1H == cache_hint::uncached
                                        || L1H == cache_hint::write_through
                                        || L1H == cache_hint::streaming))
                        || ((L2H == cache_hint::write_back)
                                && (L1H == cache_hint::uncached
                                        || L1H == cache_hint::write_through
                                        || L1H == cache_hint::streaming
                                        || L1H == cache_hint::write_back)),
                "unsupported cache hint!");
    } else if constexpr (Action == lsc_action::atomic) {
        static_assert((L1H == cache_hint::none && L2H == cache_hint::none)
                        || (L1H == cache_hint::uncached
                                && (L2H == cache_hint::uncached
                                        || L2H == cache_hint::write_back)),
                "unsupported cache hint!");
    }
}

template <cache_hint L1H, cache_hint L2H, gpu_arch arch_tag>
constexpr std::enable_if_t<arch_tag == gpu_arch::Xe, uint32_t>
get_load_cache_hint_code() {
    check_lsc_cache_hint<lsc_action::load, L1H, L2H, arch_tag>();
    if (L1H == cache_hint::none && L2H == cache_hint::none) {
        return 0;
    } else if (L2H == cache_hint::uncached) {
        if (L1H == cache_hint::uncached) { return 1; }
        if (L1H == cache_hint::cached) { return 3; }
        if (L1H == cache_hint::streaming) { return 5; }
    } else if (L2H == cache_hint::cached) {
        if (L1H == cache_hint::uncached) { return 2; }
        if (L1H == cache_hint::cached) { return 4; }
        if (L1H == cache_hint::streaming) { return 6; }
        if (L1H == cache_hint::read_invalidate) { return 7; }
    }
}

template <cache_hint L1H, cache_hint L2H, gpu_arch arch_tag>
constexpr std::enable_if_t<arch_tag == gpu_arch::Xe, uint32_t>
get_prefetch_cache_hint_code() {
    check_lsc_cache_hint<lsc_action::prefetch, L1H, L2H, arch_tag>();
    if (L2H == cache_hint::uncached) {
        if (L1H == cache_hint::uncached) { return 1; }
        if (L1H == cache_hint::cached) { return 3; }
        if (L1H == cache_hint::streaming) { return 5; }
    } else if (L2H == cache_hint::cached) {
        if (L1H == cache_hint::uncached) { return 2; }
        if (L1H == cache_hint::cached) { return 4; }
        if (L1H == cache_hint::streaming) { return 6; }
    }
}

template <cache_hint L1H, cache_hint L2H, gpu_arch arch_tag>
constexpr std::enable_if_t<arch_tag == gpu_arch::Xe, uint32_t>
get_store_cache_hint_code() {
    check_lsc_cache_hint<lsc_action::store, L1H, L2H, arch_tag>();
    if (L1H == cache_hint::none && L2H == cache_hint::none) {
        return 0;
    } else if (L2H == cache_hint::uncached) {
        if (L1H == cache_hint::uncached) { return 1; }
        if (L1H == cache_hint::write_through) { return 3; }
        if (L1H == cache_hint::streaming) { return 5; }
    } else if (L2H == cache_hint::write_back) {
        if (L1H == cache_hint::uncached) { return 2; }
        if (L1H == cache_hint::write_through) { return 4; }
        if (L1H == cache_hint::streaming) { return 6; }
        if (L1H == cache_hint::write_back) { return 7; }
    }
}

template <cache_hint L1H, cache_hint L2H, gpu_arch arch_tag>
constexpr std::enable_if_t<arch_tag == gpu_arch::Xe, uint32_t>
get_atomic_cache_hint_code() {
    check_lsc_cache_hint<lsc_action::atomic, L1H, L2H, arch_tag>();
    if (L1H == cache_hint::none && L2H == cache_hint::none) {
        return 0;
    } else if (L2H == cache_hint::uncached) {
        if (L1H == cache_hint::uncached) { return 1; }
        if (L1H == cache_hint::write_through) { return 3; }
        if (L1H == cache_hint::streaming) { return 5; }
    } else if (L2H == cache_hint::write_back) {
        if (L1H == cache_hint::uncached) { return 2; }
        if (L1H == cache_hint::write_through) { return 4; }
        if (L1H == cache_hint::streaming) { return 6; }
        if (L1H == cache_hint::write_back) { return 7; }
    }
}

template <uint32_t num_channel>
constexpr uint32_t get_execSize_code() {
    static_assert(num_channel == 1 || num_channel == 2 || num_channel == 4
                    || num_channel == 8 || num_channel == 16
                    || num_channel == 32,
            "num_channel not supported!");
    switch (num_channel) {
        case 1: return 0;
        case 2: return 1;
        case 4: return 2;
        case 8: return 3;
        case 16: return 4;
        case 32: return 5;
    }
}

template <atomic_op Op>
constexpr uint32_t get_atomic_opcode() {
    static_assert(Op == atomic_op::fadd || Op == atomic_op::fmax
                    || Op == atomic_op::iadd,
            "Other atomic op didn't added");
    switch (Op) {
        case atomic_op::fadd: return 19;
        case atomic_op::fmax: return 22;
        case atomic_op::iadd: return 12;
    }
}

} // namespace detail

///
///@brief tile layout in register
/// linear: linear layout with one tile
/// tiled: 2d block stacked in raster order
/// vnni_tiled: vnni pack with 2d block and 2d block stacked in raster order
/// for dword and qword, there is no impact
/// for word, two rows are interleaved, i.e.
/// a0 b0 c0 d0 ==> a0 a1 b0 b1 c0 c1 d0 d1
/// a1 b1 c1 d1
/// for byte, four rows are interleaved and formed one row, i.e.
/// a0 b0 c0 d0 ==> a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 c3 d0 d1 d2 d3
/// a1 b1 c1 d1
/// a2 b2 c2 d2
/// a3 b3 c3 d3
///
enum class reg_layout : uint8_t {
    linear = 0,
    tiled = 1,
    vnni_tiled = 2,
    transpose_tiled = 3,
    /// this is vnni tiled format, but for each block, they are stored in col
    /// major order
    vnni_tiled_col_major = 4
};
enum class store_op : uint8_t {
    normal = 0,
    atomic_fadd = 1,
    atomic_iadd = 2,
    scattered_transpose = 3,
    block_1d = 4
};
enum class mma_engine : uint8_t { xmx = 0, fpu = 1 };
// enum class trans_mode : uint8_t { none = 0, transpose = 1 };
enum class memory_op : uint8_t { load = 0, store = 1 };
enum class tdesc_update_dir : uint8_t { x_dir = 0, y_dir = 1 };
enum class post_kind : uint8_t {
    none = 0,
    relu = 1,
    gelu = 2,
    gelu_bwd_w = 3,
    sigmoid = 4,
    tanh = 5
};
enum class pre_kind : uint8_t { none = 0, bias_add = 1, res_add = 2 };
enum class offset_mode : uint8_t {
    const_offset = 0,
    cyclic_offset = 1,
    acyclic_offset = 2
};

} // namespace gpu::xetla
