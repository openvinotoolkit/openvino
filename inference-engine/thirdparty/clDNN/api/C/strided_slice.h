/*
// Copyright (c) 2019 Intel Corporation
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
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef STRIDED_SLICE_H
#define STRIDED_SLICE_H

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

CLDNN_BEGIN_PRIMITIVE_DESC(strided_slice)
/// @brief Array of bits, that provide replace begin[i] to max possible range in that dimension.
cldnn_uint8_t_arr begin_mask;
/// @brief Array of bits, that provide replace end[i] to max possible range in that dimension.
cldnn_uint8_t_arr end_mask;
/// @brief Array of bits, that provide adding a new length 1 dimension at ith position in the output tensor.
cldnn_uint8_t_arr new_axis_mask;
/// @brief Array of bits, that provide shrinks the dimensionality by 1, taking on the value at index begin[i].
cldnn_uint8_t_arr shrink_axis_mask;
CLDNN_END_PRIMITIVE_DESC(strided_slice)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(strided_slice);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
#endif // STRIDED_SLICE_H
