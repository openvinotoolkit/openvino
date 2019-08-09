/*
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
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/C/cldnn.h"
/// @addtogroup c_api C API
/// @{
/// @addtogroup c_topology Network Topology
/// @{
/// @addtogroup c_primitives Primitives
/// @{

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Primitives that fuses convolution, batch norm, scale and optionally Relu.
CLDNN_BEGIN_PRIMITIVE_DESC(fused_conv_bn_scale)
/// @brief Defines a shift, relative to (0,0) position of the input buffer, where (0,0) point of the convolution window should start calculations.
cldnn_tensor input_offset;
/// @brief Defines shift in input buffer between adjacent calculations of output values.
cldnn_tensor stride;
/// @brief Defines gaps in the input - dilation rate k=1 is normal convolution, k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
/// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
/// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
cldnn_tensor dilation;
/// @brief Enable Relu activation.
uint32_t with_activation;
/// @brief Relu activation slope.
float activation_negative_slope;
/// @brief On how many cards split the computation to.
uint32_t split;
/// @brief Array of primitive ids containing weights data. Size of array should be equivalent to @p split.
cldnn_primitive_id_arr weights;
/// @brief Array of primitive ids containing bias data. Size of array should be equivalent to @p split.
cldnn_primitive_id_arr bias;
/// @brief Primitive id containing scale bias data for fused convolution.
cldnn_primitive_id scale_bias;
/// @brief Primitive id containing inverted variance used in future gradient computing for fused convolution.
cldnn_primitive_id inv_variance;
/// @brief Epsilon for fused convolution.
float epsilon;
/// @brief Indicates that primitive is fused with batch norm and scale.
uint32_t fused_batch_norm_scale;
CLDNN_END_PRIMITIVE_DESC(fused_conv_bn_scale)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(fused_conv_bn_scale);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
