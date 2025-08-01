/*******************************************************************************
* Copyright (c) 2022-2024 Intel Corporation
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

#include "group/brconv/common.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_brconv
/// @{

/// @brief brconv_fwd functor.
/// @tparam compute_policy Is the compute algorithm of brconv implementation.
/// @tparam tile_shape Is the workgroup-level tile shape.
/// @tparam brconv_filter_attr Is the convolution filter attributes.
/// @tparam mem_desc_src Is the memory descriptor of source data.
/// @tparam mem_desc_weight Is the memory descriptor of weights.
template <typename compute_policy, typename tile_shape,
        typename brconv_filter_attr, typename mem_desc_src,
        typename mem_desc_weight, class enable = void>
class brconv_fwd_t {};

/// @} xetla_brconv

} // namespace gpu::xetla::group
