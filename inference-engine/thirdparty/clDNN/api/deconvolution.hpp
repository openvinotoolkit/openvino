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
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs transposed convolution.
/// Also supports built-in Relu @ref activation available by setting it in arguments.
/// @details Deconvolution is similar to convolution layer with the weights flipped on the axis
/// and stride and input padding parameters used in opposite sense as in convolution.
struct deconvolution : public primitive_base<deconvolution> {
    CLDNN_DECLARE_PRIMITIVE(deconvolution)
    /// @brief Constructs deconvolution primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data. Provide empty vector if using next parameters without bias.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the deconvolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id>& weights,
                  const std::vector<primitive_id>& bias,
                  tensor stride = {1, 1, 1, 1},
                  tensor input_offset = {0, 0, 0, 0},
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          with_output_size(false),
          groups(1),
          grouped_weights_shape(false),
          weights(weights),
          bias(bias) {}
    /// @brief Constructs deconvolution primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param groups Number of filter groups.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data. Provide empty vector if using next parameters without bias.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the deconvolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id>& weights,
                  const std::vector<primitive_id>& bias,
                  uint32_t groups,
                  tensor stride = {1, 1, 1, 1},
                  tensor input_offset = {0, 0, 0, 0},
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          with_output_size(false),
          groups(groups),
          grouped_weights_shape(false),
          weights(weights),
          bias(bias) {}

    /// @brief Constructs deconvolution primitive (w/o bias).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the deconvolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id>& weights,
                  tensor stride = {1, 1, 1, 1},
                  tensor input_offset = {0, 0, 0, 0},
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          with_output_size(false),
          groups(1),
          grouped_weights_shape(false),
          weights(weights),
          bias(std::vector<primitive_id>(0)) {}

    /// @brief Constructs deconvolution primitive (w/o bias).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param groups Number of filter groups.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the deconvolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id>& weights,
                  uint32_t groups,
                  tensor stride = {1, 1, 1, 1},
                  tensor input_offset = {0, 0, 0, 0},
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          with_output_size(false),
          groups(groups),
          grouped_weights_shape(false),
          weights(weights),
          bias(std::vector<primitive_id>(0)) {}

    /// @brief Constructs deconvolution primitive (computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data. Provide empty vector if using next parameters without bias.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the deconvolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id>& weights,
                  const std::vector<primitive_id>& bias,
                  tensor stride,
                  tensor input_offset,
                  tensor output_size,
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          with_output_size(true),
          output_size(output_size),
          groups(1),
          grouped_weights_shape(false),
          weights(weights),
          bias(bias) {}

    /// @brief Constructs deconvolution primitive (computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data. Provide empty vector if using next parameters without bias.
    /// @param groups Number of filter groups.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the deconvolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id>& weights,
                  const std::vector<primitive_id>& bias,
                  uint32_t groups,
                  tensor stride,
                  tensor input_offset,
                  tensor output_size,
                  bool grouped_weights_shape,
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          with_output_size(true),
          output_size(output_size),
          groups(groups),
          grouped_weights_shape(grouped_weights_shape),
          weights(weights),
          bias(bias) {}

    /// @brief Constructs deconvolution primitive (w/o bias, computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the deconvolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id>& weights,
                  tensor stride,
                  tensor input_offset,
                  tensor output_size,
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          with_output_size(true),
          output_size(output_size),
          groups(1),
          grouped_weights_shape(false),
          weights(weights),
          bias(std::vector<primitive_id>(0)) {}

    /// @brief Constructs deconvolution primitive (computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data. Provide empty vector if using next parameters without bias.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the deconvolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    /// @return Deconvolution primitive with specified settings.
    static deconvolution create_with_output_size(const primitive_id& id,
                                                 const primitive_id& input,
                                                 const std::vector<primitive_id>& weights,
                                                 const std::vector<primitive_id>& bias,
                                                 tensor output_size,
                                                 tensor stride = {1, 1, 1, 1},
                                                 tensor input_offset = {0, 0, 0, 0},
                                                 const padding& output_padding = padding()) {
        return deconvolution(id,
                             input,
                             weights,
                             bias,
                             stride,
                             input_offset,
                             output_size,
                             output_padding);
    }

    /// @brief Constructs deconvolution primitive (w/o bias; computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the deconvolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    /// @return Deconvolution primitive with specified settings.
    static deconvolution create_with_output_size(const primitive_id& id,
                                                 const primitive_id& input,
                                                 const std::vector<primitive_id>& weights,
                                                 tensor output_size,
                                                 tensor stride = {1, 1, 1, 1},
                                                 tensor input_offset = {0, 0, 0, 0},
                                                 const padding& output_padding = padding()) {
        return deconvolution(id,
                             input,
                             weights,
                             stride,
                             input_offset,
                             output_size,
                             output_padding);
    }

    /// @brief Defines a shift, relative to (0,0) position of the input buffer, where (0,0) point of the deconvolution window should start calculations.
    tensor input_offset;
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    tensor stride;
    /// @brief Indicates that the primitive has user-defined output size (non-zero value).
    bool with_output_size;
    /// @brief User-defined output data size of the primitive (w/o padding).
    tensor output_size;
    /// @brief Number of feature groups (grouped convolution). If more than 1 then weights/bias count needs to be 1.
    uint32_t groups;
    /// @param grouped_weights_shape Defines if weights tensor has explicit group dimension.
    bool grouped_weights_shape;
    /// @brief List of primitive ids containing weights data.
    const primitive_id_arr weights;
    /// @brief List of primitive ids containing bias data.
    const primitive_id_arr bias;

    /// @brief On how many cards split the computation to.
    int32_t split() const { return static_cast<int32_t>(weights.size()); }

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.reserve(weights.size() + bias.size());
        for (auto& w : weights) ret.push_back(std::ref(w));
        for (auto& b : bias) ret.push_back(std::ref(b));

        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
