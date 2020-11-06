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
#include "primitive.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs forward spatial binary_convolution with weight sharing.
struct binary_convolution : public primitive_base<binary_convolution> {
    CLDNN_DECLARE_PRIMITIVE(binary_convolution)

    /// @brief Constructs binary_convolution primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the binary_convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal binary_convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    /// @param groups Number of feature groups (grouped convolution). If more than 1 then weights/bias count needs to be 1.
    /// @param pad_value Logical value of padding. Can be one of 3 values: 1 - pad bits equal to 1; -1 -> pad bits equal to 0; 0 -> pad is not counted
    /// @param calc_precision Precision of intermediate accumulators
    binary_convolution(const primitive_id& id,
                       const primitive_id& input,
                       const std::vector<primitive_id>& weights,
                       tensor stride = {1, 1, 1, 1},
                       tensor input_offset = {0, 0, 0, 0},
                       tensor dilation = {1, 1, 1, 1},
                       tensor output_size = {0, 0, 0, 0},
                       int groups = 1,
                       float pad_value = 0.0f,
                       data_types calc_precision = data_types::f32,
                       const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding, optional_data_type {calc_precision}),
          input_offset(input_offset),
          stride(stride),
          dilation(dilation),
          output_size(output_size),
          groups(groups),
          pad_value(pad_value),
          weights(weights) {}

    /// @brief Defines a shift, relative to (0,0) position of the input buffer, where (0,0) point of the binary_convolution window should start calculations.
    tensor input_offset;
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    tensor stride;
    /// @brief Defines gaps in the input - dilation rate k=1 is normal binary_convolution, k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    tensor dilation;
    /// @brief User-defined output data size of the primitive (w/o padding).
    tensor output_size;
    /// @brief Number of feature groups (grouped convolution). If more than 1 then weights/bias count needs to be 1.
    int groups;
    /// @brief Logical value of padding. Can be one of 3 values: 1 - pad bits equal to 1; -1 -> pad bits equal to 0; 0 -> pad is not counted
    float pad_value;
    /// @brief List of primitive ids containing weights data.
    const primitive_id_arr weights;

    int32_t split() const { return static_cast<int32_t>(weights.size()); }

    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.reserve(weights.size());
        for (auto& w : weights) ret.push_back(std::ref(w));
        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
