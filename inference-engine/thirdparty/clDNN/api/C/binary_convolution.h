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
#pragma once
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

/// @brief Performs forward spatial binary_convolution with weight sharing.
/// @details Parameters are defined in context of "direct" binary_convolution, but actual algorithm is not implied.
CLDNN_BEGIN_PRIMITIVE_DESC(binary_convolution)
/// @brief Defines a shift, relative to (0,0) position of the input buffer, where (0,0) point of the binary_convolution window should start calculations.
cldnn_tensor input_offset;
/// @brief Defines shift in input buffer between adjacent calculations of output values.
cldnn_tensor stride;
/// @brief Defines gaps in the input - dilation rate k=1 is normal binary_convolution, k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
/// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
/// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
cldnn_tensor dilation;
/// @brief Weights groups count
uint32_t split;
/// @brief User-defined output data size of the primitive (w/o padding).
cldnn_tensor output_size;
/// @brief Number of feature groups (grouped convolution). If more than 1 then weights/bias count needs to be 1.
int groups;
/// @brief Logical value of padding. Can be one of 3 values: 1 - pad bits equal to 1; -1 -> pad bits equal to 0; 0 -> pad is not counted
float pad_value;
/// @brief Array of primitive ids containing weights data. Size of array should be equivalent to @p split.
cldnn_primitive_id_arr weights;

CLDNN_END_PRIMITIVE_DESC(binary_convolution)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(binary_convolution);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}

