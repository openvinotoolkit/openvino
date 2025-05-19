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

#include "subgroup/tile/tile.hpp"

namespace gpu::xetla::subgroup {

/// @brief  Helper to do the cooperative workgroups load.
/// @tparam matAcc_t Is the input mat type.
/// @tparam tile_shape Is the group-level tile shape.
/// @tparam mem_layout Is the memory layout of input.
/// @tparam num_cooperative_wg Is the number of workgroups to do the cooperation.
/// @tparam arch_tag Is the HW architecture.
template <typename matAcc_t_, mem_layout mem_layout_,
        uint32_t num_cooperative_wg, gpu_arch arch_tag_, class enable = void>
class cooperative_load_helper_t {};

/// @brief Workgroups to do the cooperative load. Specialized for and row_major and Xe architecture.
template <typename matAcc_t_, uint32_t num_cooperative_wg, gpu_arch arch_tag_>
class cooperative_load_helper_t<matAcc_t_, mem_layout::row_major,
        num_cooperative_wg, arch_tag_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)
                >> {
public:
    static constexpr gpu_arch arch_tag = arch_tag_;
    using matAcc_t = matAcc_t_;
    using dtype = typename matAcc_t::dtype;
    using tile_desc_t = typename matAcc_t::tile_desc;
    static constexpr mem_layout layout = mem_layout::row_major;

private:
    // cooperative split, y dir first
    static_assert((num_cooperative_wg & (num_cooperative_wg - 1)) == 0,
            "num_cooperative_wg should be power of 2");

public:
    static constexpr uint32_t src_block_size_x = tile_desc_t::block_size_x;
    static constexpr uint32_t src_block_size_y = tile_desc_t::block_size_y;
    static constexpr uint32_t src_tile_size_x = tile_desc_t::tile_size_x;
    static constexpr uint32_t src_tile_size_y = tile_desc_t::tile_size_y;

    static constexpr uint32_t coop_num_y
            = gpu::xetla::subgroup::detail::gcd<num_cooperative_wg,
                    src_tile_size_y>::value;
    static constexpr uint32_t coop_remain_num_x
            = num_cooperative_wg / coop_num_y;
    static constexpr uint32_t tile_size_y = src_tile_size_y / coop_num_y;
    static constexpr uint32_t tile_size_x = src_tile_size_x / coop_remain_num_x;
    static constexpr uint32_t coop_num_x = src_tile_size_x / tile_size_x;

    static_assert((tile_size_y * tile_size_x % 16) == 0,
            "cooperative tile size should be a multiply of simd-16 ");

public:
    static constexpr uint32_t block_size_x
            = gpu::xetla::subgroup::detail::gcd<tile_size_x,
                    src_block_size_x>::value;
    static constexpr uint32_t block_size_y
            = (tile_size_y > src_block_size_y) ? src_block_size_y : tile_size_y;

    using co_tile_desc_t = subgroup::tile_desc_t<tile_size_x, tile_size_y,
            block_size_x, block_size_y, reg_layout::tiled>;

public:
    inline cooperative_load_helper_t() = default;

    inline static int32_t get_offset_x(uint32_t coop_id) {
        return coop_id % coop_remain_num_x * tile_size_x;
    }

    inline static int32_t get_offset_y(uint32_t coop_id) {
        return coop_id / coop_remain_num_x * tile_size_y;
    }
};

/// @brief Workgroups to do the cooperative load. Specialized for and row_major and Xe architecture.
template <typename matAcc_t_, uint32_t num_cooperative_wg, gpu_arch arch_tag_>
class cooperative_load_helper_t<matAcc_t_, mem_layout::col_major,
        num_cooperative_wg, arch_tag_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)
                >> {
public:
    static constexpr gpu_arch arch_tag = arch_tag_;
    using matAcc_t = matAcc_t_;
    using dtype = typename matAcc_t::dtype;
    using tile_desc_t = typename matAcc_t::tile_desc;
    static constexpr mem_layout layout = mem_layout::col_major;

private:
    // cooperative split, y dir first
    static_assert((num_cooperative_wg & (num_cooperative_wg - 1)) == 0,
            "num_cooperative_wg should be power of 2");

public:
    static constexpr uint32_t src_block_size_x = tile_desc_t::block_size_x;
    static constexpr uint32_t src_block_size_y = tile_desc_t::block_size_y;
    static constexpr uint32_t src_tile_size_x = tile_desc_t::tile_size_x;
    static constexpr uint32_t src_tile_size_y = tile_desc_t::tile_size_y;

    static constexpr uint32_t coop_num_x
            = gpu::xetla::subgroup::detail::gcd<num_cooperative_wg,
                    src_tile_size_x>::value;
    static constexpr uint32_t coop_remain_num_y
            = num_cooperative_wg / coop_num_x;
    static constexpr uint32_t tile_size_x = src_tile_size_x / coop_num_x;
    static constexpr uint32_t tile_size_y = src_tile_size_y / coop_remain_num_y;
    static constexpr uint32_t coop_num_y = src_tile_size_y / tile_size_y;

    static_assert((tile_size_y * tile_size_x % 16) == 0,
            "cooperative tile size should be a multiply of simd-16 ");

public:
    static constexpr uint32_t block_size_y
            = gpu::xetla::subgroup::detail::gcd<tile_size_y,
                    src_block_size_y>::value;
    static constexpr uint32_t block_size_x
            = (tile_size_x > src_block_size_x) ? src_block_size_x : tile_size_x;

    using co_tile_desc_t = subgroup::tile_desc_t<tile_size_x, tile_size_y,
            block_size_x, block_size_y, reg_layout::tiled>;

public:
    inline cooperative_load_helper_t() = default;

    inline static int32_t get_offset_x(uint32_t coop_id) {
        return coop_id / coop_remain_num_y * tile_size_x;
    }

    inline static int32_t get_offset_y(uint32_t coop_id) {
        return coop_id % coop_remain_num_y * tile_size_y;
    }
};

} // namespace gpu::xetla::subgroup
