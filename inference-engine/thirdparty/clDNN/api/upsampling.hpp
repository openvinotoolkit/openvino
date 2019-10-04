/*
// Copyright (c) 2016 Intel Corporation
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

/// @brief Sample mode for the @ref upsampling layer.
enum class upsampling_sample_type : int32_t {
    /// @brief upsampling nearest neighbor.
    nearest,
    /// @brief upsampling bilinear.
    bilinear
};

/// @brief Performs nearest neighbor/bilinear upsampling
/// Also supports built-in Relu @ref activation available by setting it in arguments.
struct upsampling : public primitive_base<upsampling> {
    CLDNN_DECLARE_PRIMITIVE(upsampling)

    /// @brief Constructs upsampling primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param scale Upsampling scale.
    /// @param num_filter Input filter. Only used by bilinear sample_type.
    /// @param sample_type Upsampling method (nearest neighbor/bilinear).
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    upsampling(const primitive_id& id,
               const primitive_id& input,
               tensor output_size,
               uint32_t num_filter,
               upsampling_sample_type sample_type,
               bool with_activation = false,
               float activation_slp = 0.0f,
               const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          output_size(output_size),
          num_filter(num_filter),
          sample_type(sample_type),
          with_activation(with_activation),
          activation_negative_slope(activation_slp) {}

    /// @param scale Upsampling scale.
    tensor output_size;
    /// @param num_filter Input filter. Only used by bilinear sample_type.
    uint32_t num_filter;
    /// @param sample_type Upsampling method (nearest neighbor/bilinear).
    upsampling_sample_type sample_type;
    /// @brief Enables Relu activation.
    bool with_activation;
    /// @brief Relu activation slope.
    float activation_negative_slope;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
