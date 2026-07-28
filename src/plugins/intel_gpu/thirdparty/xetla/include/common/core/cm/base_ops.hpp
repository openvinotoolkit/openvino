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
#include "../../../common/core/cm/common.hpp"
#else
#include "common/core/cm/common.hpp"
#endif

namespace gpu::xetla {

/// @addtogroup xetla_core_base_ops
/// @{

/// @brief xetla format.
/// Alias to CM `.format<...>()`;
/// @note usage:
/// ```
/// [xetla_vector|xetla_vector_ref|xetla_matrix_ref].xetla_format<type>(): returns a reference to the calling object interpreted as a new xetla_vector_ref (1D).
///
/// [xetla_vector|xetla_vector_ref|xetla_matrix_ref].xetla_format<type, rows, columns>(): returns a reference to the calling object interpreted as a new xetla_matrix_ref (2D).
/// ```
///
#define xetla_format format

/// @brief xetla select.
/// Alias to CM `.select<...>()`;
/// @note usage:
/// ```
/// [xetla_vector|xetla_vector_ref].xetla_select<size, stride>(uint16_t offset=0): returns a reference to the sub-vector starting from the offset-th element.
///
/// [xetla_matrix_ref].xetla_select<size_y, stride_y, size_x, stride_x>(uint16_t offset_y=0, uint16_t offset_x=0): returns a reference to the sub-matrix starting from the (offset_y, offset_x)-th element.
/// ```
///
#define xetla_select select

/// @brief xetla merge.
/// Alias to `.merge(...)`. Replaces part of the underlying data with the one taken from the other object according to a mask.
/// @note usage:
/// ```
/// [xetla_vector|xetla_vector_ref].xetla_merge(xetla_vector<Ty, N>Val, xetla_mask<N>mask): only elements in lanes with non-zero mask predicate are assigned from corresponding Val elements.
///
/// [xetla_vector|xetla_vector_ref].xetla_merge(xetla_vector<Ty, N>Val1, xetla_vector<Ty, N>Val2, xetla_mask<N>mask): non-zero in a mask's lane tells to take corresponding element from Val1, zero - from Val2.
/// ```
///
#define xetla_merge merge

// TODO add replicate, iselect

/// @} xetla_core_base_ops

} // namespace gpu::xetla
