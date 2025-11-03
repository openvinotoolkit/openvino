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

#include "group/epilogue/common.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_epilogue
/// @{

/// @brief Default epilogue policy for store C.
/// @tparam arch_tag_ Is the HW architecture.
template <gpu_arch arch_tag_>
struct epilogue_policy_default {
    static constexpr gpu_arch arch_tag = arch_tag_;
};

/// @brief Default epilogue policy for store C.
/// @tparam tile_op_t_ Is the tile_op functor to apply before groupnorm reduction.
/// @tparam gn_reduce_t_ Is the gropunorm reduction.
/// @tparam arch_tag_ Is the HW architecture.
template <typename tile_op_t_, typename gn_reduce_t_, gpu_arch arch_tag_>
struct epilogue_policy_gn_reduce {
    using tile_op_t = tile_op_t_;
    using gn_reduce_t = gn_reduce_t_;
    static constexpr gpu_arch arch_tag = arch_tag_;
};

/// @addtogroup xetla_epilogue
/// @{

/// @brief Epilogue policy for tile_op + store C fusion.
/// @tparam tile_op_t_ Is the tile_op functor.
/// @tparam arch_tag_ Is the HW architecture.
template <typename tile_op_t_, gpu_arch arch_tag_>
struct epilogue_policy_tile_op {
    using tile_op_t = tile_op_t_;
    static constexpr gpu_arch arch_tag = arch_tag_;
};

/// @} xetla_epilogue

} // namespace gpu::xetla::group
