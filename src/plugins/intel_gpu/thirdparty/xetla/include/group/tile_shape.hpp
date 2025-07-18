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

namespace gpu::xetla::group {

template <auto... Args>
struct tile_shape_t;

/// @brief Workgroup level tile shape description.
/// Describes the task assignment and layout of subgroups in a group.
/// @tparam wg_tile_size_x_ Is the workgroup level tile size in x direction.
/// @tparam wg_tile_size_y_ Is the workgroup level tile size in y direction.
/// @tparam sg_tile_size_x_ Is the subgroup level tile size in x direction.
/// @tparam sg_tile_size_y_ Is the subgroup level tile size in y direction.
template <uint32_t wg_tile_size_x_, uint32_t wg_tile_size_y_,
        uint32_t sg_tile_size_x_, uint32_t sg_tile_size_y_>
struct tile_shape_t<wg_tile_size_x_, wg_tile_size_y_, sg_tile_size_x_,
        sg_tile_size_y_> {
    static constexpr uint32_t dim = 2;
    static constexpr uint32_t wg_tile_size_x = wg_tile_size_x_;
    static constexpr uint32_t wg_tile_size_y = wg_tile_size_y_;
    static constexpr uint32_t sg_tile_size_x = sg_tile_size_x_;
    static constexpr uint32_t sg_tile_size_y = sg_tile_size_y_;

    static constexpr uint32_t wg_size_x
            = (wg_tile_size_x + sg_tile_size_x - 1) / sg_tile_size_x;
    static constexpr uint32_t wg_size_y
            = (wg_tile_size_y + sg_tile_size_y - 1) / sg_tile_size_y;
    using work_group_t = work_group_t<wg_size_x * wg_size_y>;
};

/// @brief Workgroup level tile shape description.
/// Describes the task assignment and layout of subgroups in a group.
/// @tparam wg_tile_size_n_  Is the workgroup level tile size in n direction.
/// @tparam wg_tile_size_p_  Is the workgroup level tile size in p direction.
/// @tparam wg_tile_size_q_  Is the workgroup level tile size in q direction.
/// @tparam wg_tile_size_k_  Is the workgroup level tile size in k direction.
/// @tparam sg_tile_size_n_  Is the subgroup level tile size in n direction.
/// @tparam sg_tile_size_p_  Is the subgroup level tile size in p direction.
/// @tparam sg_tile_size_q_  Is the subgroup level tile size in q direction.
/// @tparam sg_tile_size_k_  Is the subgroup level tile size in k direction.
/// ------------------------------------------------------------------------
template <uint32_t wg_tile_size_n_, uint32_t wg_tile_size_p_,
        uint32_t wg_tile_size_q_, uint32_t wg_tile_size_k_,
        uint32_t sg_tile_size_n_, uint32_t sg_tile_size_p_,
        uint32_t sg_tile_size_q_, uint32_t sg_tile_size_k_>
struct tile_shape_t<wg_tile_size_n_, wg_tile_size_p_, wg_tile_size_q_,
        wg_tile_size_k_, sg_tile_size_n_, sg_tile_size_p_, sg_tile_size_q_,
        sg_tile_size_k_> {
    static constexpr uint32_t dim = 4;
    static constexpr uint32_t wg_tile_size_n = wg_tile_size_n_;
    static constexpr uint32_t wg_tile_size_p = wg_tile_size_p_;
    static constexpr uint32_t wg_tile_size_q = wg_tile_size_q_;
    static constexpr uint32_t wg_tile_size_k = wg_tile_size_k_;

    static constexpr uint32_t sg_tile_size_n = sg_tile_size_n_;
    static constexpr uint32_t sg_tile_size_p = sg_tile_size_p_;
    static constexpr uint32_t sg_tile_size_q = sg_tile_size_q_;
    static constexpr uint32_t sg_tile_size_k = sg_tile_size_k_;

    static_assert(wg_tile_size_n % sg_tile_size_n == 0,
            "wg_tile_size_n should be a multiple of sg_tile_size_n");
    static_assert(wg_tile_size_p % sg_tile_size_p == 0,
            "wg_tile_size_p should be a multiple of sg_tile_size_p");
    static_assert(wg_tile_size_q % sg_tile_size_q == 0,
            "wg_tile_size_q should be a multiple of sg_tile_size_q");
    static_assert(wg_tile_size_k % sg_tile_size_k == 0,
            "wg_tile_size_k should be a multiple of sg_tile_size_k");

    static constexpr uint32_t wg_size_n
            = (wg_tile_size_n + sg_tile_size_n - 1) / sg_tile_size_n;
    static constexpr uint32_t wg_size_p
            = (wg_tile_size_p + sg_tile_size_p - 1) / sg_tile_size_p;
    static constexpr uint32_t wg_size_q
            = (wg_tile_size_q + sg_tile_size_q - 1) / sg_tile_size_q;
    static constexpr uint32_t wg_size_k
            = (wg_tile_size_k + sg_tile_size_k - 1) / sg_tile_size_k;

    using work_group_t
            = work_group_t<wg_size_n * wg_size_p * wg_size_q * wg_size_k>;
};
} // namespace gpu::xetla::group
