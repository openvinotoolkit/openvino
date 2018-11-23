// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef INDEX_SELECT_H
#define INDEX_SELECT_H

#include "cldnn.h"


/// @addtogroup c_api C API
/// @{
/// @addtogroup c_topology Network Topology
/// @{
/// @addtogroup c_primitives Primitives
/// @{

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Axis which index_select primitive will index.
typedef enum /*:int32_t*/
{
    cldnn_along_b,
    cldnn_along_f,
    cldnn_along_x,
    cldnn_along_y,
} cldnn_index_select_axis;

/// @brief Select index, which will be copied to the output..
///
/// @details Applies index selecting along specified dimension. The indices, which will be copied are specifed by 
///          by @c indices.
/// @n
/// @n Example:
/// @n      <tt>input_sizes  = (1, 2, 4, 2)</tt>
/// @n      <tt>input_values = (a, b, c, d)</tt>
/// @n      <tt>               (e, f, g, h)</tt>
/// @n      <tt>indices_sizes  = (1, 1, 6, 1)</tt>
/// @n      <tt>indices_values = {0, 0, 1, 1, 3, 3}</tt>                  
/// @n  For axis: along_x:
/// @n      <tt>output_sizes  = (1, 2, 6, 2)</tt>
/// @n      <tt>output_values = (a, a, b, b, d, d)</tt>
/// @n      <tt>                (e, e, f, f, h, h)</tt>
/// @n
/// @n The resulting output will have sizes equal to input_size with changed concrete tensor size to inidices x size.
/// @n
/// @n@b Requirements:
/// @n - @c input must be a valid primitive_id, which output's format is bfyx/yxfb;
/// @n - @c indices must be a valid primitive_id, which output's layout is: (bfyx/yxfb, i32, {1, 1, indicies_size, 1})
/// @n - @c axis - valid index_select_axis_name instance. 
/// @n Breaking any of this conditions will cause exeption throw.
CLDNN_BEGIN_PRIMITIVE_DESC(index_select)

/// @brief Axis of index selecting.
cldnn_index_select_axis axis;

CLDNN_END_PRIMITIVE_DESC(index_select)


CLDNN_DECLARE_PRIMITIVE_TYPE_ID(index_select);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif // INDEX_SELECT_H
