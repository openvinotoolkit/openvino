/*
// Copyright (c) 2016-2019 Intel Corporation
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
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Sample mode for the @ref resample layer.
enum class resample_type : int32_t {
    /// @brief nearest neighbor.
    nearest,
    /// @brief bilinear interpolation.
    bilinear,
    /// @brief caffe bilinear interpolation.
    caffe_bilinear
};

/// @brief Performs nearest neighbor/bilinear resample
/// Also supports built-in Relu @ref activation available by setting it in arguments.
struct resample : public primitive_base<resample> {
    CLDNN_DECLARE_PRIMITIVE(resample)

    /// @brief Constructs Resample primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param scale Resample scale.
    /// @param num_filter Input filter. Only used by bilinear sample_type.
    /// @param sample_type Resample method (nearest neighbor/bilinear/caffe bilinear).
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    resample(const primitive_id& id,
             const primitive_id& input,
             tensor output_size,
             uint32_t num_filter,
             resample_type operation_type = resample_type::nearest,
             bool with_activation = false,
             float activation_slp = 0.0f,
             const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          output_size(output_size),
          num_filter(num_filter),
          pad_begin(0),
          pad_end(0),
          align_corners(1),
          operation_type(operation_type),
          with_activation(with_activation),
          activation_negative_slope(activation_slp) {}

    /// @brief Constructs Resample primitive with Interp operation.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param pad_begin Optional begin padding for input.
    /// @param pad_end Optional end padding for input.
    /// @param align_corners Align corner pixels of the input and output tensors.
    /// @param resample_type Resample bilinear method.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    resample(const primitive_id& id,
             const primitive_id& input,
             tensor output_size,
             int32_t pad_begin = 0,
             int32_t pad_end = 0,
             int32_t align_corners = 1,
             resample_type operation_type = resample_type::bilinear,
             bool with_activation = false,
             float activation_slp = 0.0f,
             const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          output_size(output_size),
          num_filter(0),
          pad_begin(pad_begin),
          pad_end(pad_end),
          align_corners(align_corners),
          operation_type(operation_type),
          with_activation(with_activation),
          activation_negative_slope(activation_slp) {}

    /// @param scale Resample scale.
    tensor output_size;
    /// @param num_filter Input filter. Only used by bilinear sample_type.
    uint32_t num_filter;
    /// @param pad_begin Begin padding for input.
    int32_t pad_begin;
    /// @param pad_end End padding for input.
    int32_t pad_end;
    /// @param align_corners corner pixels of the input and output tensors
    int32_t align_corners;
    /// @param sample_type Resample method (nearest neighbor/bilinear/caffe bilinear).
    resample_type operation_type;
    /// @brief Enables Relu activation.
    bool with_activation;
    /// @brief Relu activation slope.
    float activation_negative_slope;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
