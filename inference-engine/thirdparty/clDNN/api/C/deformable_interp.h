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

/// @brief Performs interpolation pass for deformable convolution. Output tensor has IC*KH*KW channels.
CLDNN_BEGIN_PRIMITIVE_DESC(deformable_interp)
/// @brief Defines a shift, relative to (0,0) position of the input buffer, where (0,0) point of the convolution window should start calculations.
cldnn_tensor input_offset;
/// @brief Defines shift in input buffer between adjacent calculations of output values.
cldnn_tensor stride;
/// @brief Defines gaps in the input - dilation rate k=1 is normal convolution, k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
/// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
/// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
cldnn_tensor dilation;
/// @brief On how many cards split the computation to.
uint32_t split;
/// @brief User-defined output data size of the primitive (w/o padding).
cldnn_tensor output_size;
/// @brief Size of the weights tensor.
cldnn_tensor kernel_size;
uint32_t groups;
/// @param deformable_groups Defines a number of deformable groups that splits trans input into several parts
/// by channel dimension.
uint32_t deformable_groups;
/// @param padding_above Defines a padding added to input image on left (x axis) and top (y axis).
cldnn_tensor padding_above;
/// @param padding_below Defines a padding added to input image on right (x axis) and bottom (y axis).
cldnn_tensor padding_below;

CLDNN_END_PRIMITIVE_DESC(deformable_interp)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(deformable_interp);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
