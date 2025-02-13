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
} // namespace gpu::xetla::group
