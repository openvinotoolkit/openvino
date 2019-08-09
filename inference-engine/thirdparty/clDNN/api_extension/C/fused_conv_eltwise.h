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

/// @brief Performs forward spatial convolution with weight sharing fused with eltwise.
/// Also supports built-in Relu @CLDNN_PRIMITIVE_DESC{activation} separate for convolution and for eltwise, available by setting it in arguments.
CLDNN_BEGIN_PRIMITIVE_DESC(fused_conv_eltwise)

struct conv_data {
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
    /// @brief Indicates that the primitive has user-defined output size (non-zero value).
    uint32_t with_output_size;
    /// @brief User-defined output data size of the primitive (w/o padding).
    cldnn_tensor output_size;
    /// @brief Array of primitive ids containing weights data. Size of array should be equivalent to @p split.
    cldnn_primitive_id_arr weights;
    /// @brief Array of primitive ids containing bias data. Size of array should be equivalent to @p split.
    cldnn_primitive_id_arr bias;
    /// @brief List of primitive ids containing weights quanitization factors per output feature map.
    cldnn_primitive_id_arr weights_quantization_factors;
    /// @brief List of primitive ids containing output calibration factors per output feature map.
    cldnn_primitive_id_arr output_calibration_factors;
    /// @brief Input quantization factor
    float input_quantization_factor;
    /// @brief Output quantization factor
    float output_quantization_factor;
} conv;

struct eltw_data {
    /// @brief Primitive id containing output quanitization factors per output feature map.
    cldnn_primitive_id output_calibration_factors;
    /// @brief Output quantization factor
    float output_quantization_factor;
    /// @brief Eltwise mode. See #cldnn_eltwise_mode.
    int32_t mode; /*cldnn_eltwise_mode*/
    /// @brief Blob-wise coefficient for SUM operation
    cldnn_float_arr coefficients;
    /// @brief Enables Relu activation.
    uint32_t with_activation;
    /// @brief Relu activation slope.
    float activation_negative_slope;
    /// @brief Defines shift in input buffers between adjacent calculations of output values.
    cldnn_tensor_arr stride;
} eltw;

// @brief Non-convolution output scaling factor. Might be used both to represent
// i8->float dynamic range conversion and dynamic range scaling without changing
// data precision (e.g. to align dynamic range with that of convolution result).
float non_conv_scale = 1.0f;

/// @brief Is optimization that output contains data from second input ON ?
bool second_input_in_output = false;

CLDNN_END_PRIMITIVE_DESC(fused_conv_eltwise)

CLDNN_DECLARE_PRIMITIVE_TYPE_ID(fused_conv_eltwise);

#ifdef __cplusplus
}
#endif

/// @}
/// @}
/// @}
