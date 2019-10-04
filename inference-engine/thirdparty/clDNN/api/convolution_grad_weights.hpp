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
#include "primitive.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs backward convolution operation for weights and biases.
/// @details convolution_grad_weights updates weights and bias mutable data for training purposes.
/// @details Please note that this primitive was not heavily tested and currently only batch=1 is enabled for this primitive.
struct convolution_grad_weights
    : public primitive_base<convolution_grad_weights> {
    CLDNN_DECLARE_PRIMITIVE(convolution_grad_weights)

    /// @brief Constructs convolution_grad_weights primitive.
    /// @param id This primitive id.
    /// @param input Input gradient primitive id.
    /// @param input Input primitive id from convolution forward pass.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data. Provide empty vector if using next parameters without bias.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution_grad_weights window should start calculations.
    /// @param dilation Defines dilation size.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param conv_grad Id of primitive which uses weights and biases updated in this primitive.
    /// This is for correct order of calculating. Leave empty if primitive is last in backward pass.
    convolution_grad_weights(const primitive_id& id,
                             const primitive_id& input_grad,
                             const primitive_id& input,
                             const std::vector<primitive_id>& weights,
                             const std::vector<primitive_id>& bias,
                             tensor stride = {1, 1, 1, 1},
                             tensor input_offset = {0, 0, 0, 0},
                             tensor dilation = {1, 1, 1, 1},
                             const primitive_id& conv_grad = "",
                             const padding& output_padding = padding())
        : primitive_base(id, {input_grad, input}, output_padding),
          conv_grad(conv_grad),
          stride(stride),
          input_offset(input_offset),
          dilation(dilation),
          output_grad_w(false),
          weights(weights),
          bias(bias),
          prev_weights_grad(std::vector<primitive_id>(0)),
          prev_bias_grad(std::vector<primitive_id>(0)) {}

    /// @brief Constructs convolution_grad_weights primitive (w/o bias).
    /// @param id This primitive id.
    /// @param input Input gradient primitive id.
    /// @param input Input primitive id from convolution forward pass.
    /// @param weights List of primitive ids containing weights data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution_grad_weights window should start calculations.
    /// @param dilation Defines dilation size.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param Should primitive give weights gradient (delta) as an output
    /// @param conv_grad Id of primitive which uses weights and biases updated in this primitive.
    /// This is for correct order of calculating. Leave empty if primitive is last in backward pass.
    convolution_grad_weights(const primitive_id& id,
                             const primitive_id& input_grad,
                             const primitive_id& input,
                             const std::vector<primitive_id>& weights,
                             tensor stride = {1, 1, 1, 1},
                             tensor input_offset = {0, 0, 0, 0},
                             tensor dilation = {1, 1, 1, 1},
                             bool output_grad_w = false,
                             const primitive_id& conv_grad = "",
                             const padding& output_padding = padding())
        : primitive_base(id, {input_grad, input}, output_padding),
          conv_grad(conv_grad),
          stride(stride),
          input_offset(input_offset),
          dilation(dilation),
          output_grad_w(output_grad_w),
          weights(weights),
          bias(std::vector<primitive_id>(0)),
          prev_weights_grad(std::vector<primitive_id>(0)),
          prev_bias_grad(std::vector<primitive_id>(0)) {}

    /// @brief Constructs convolution_grad_weights primitive (w/o bias).
    /// @param id This primitive id.
    /// @param input Input gradient primitive id.
    /// @param input Input primitive id from convolution forward pass.
    /// @param weights List of primitive ids containing weights data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution_grad_weights window should start calculations.
    /// @param dilation Defines dilation size.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param conv_grad Id of primitive which uses weights and biases updated in this primitive.
    /// This is for correct order of calculating. Leave empty if primitive is last in backward pass.
    convolution_grad_weights(const primitive_id& id,
                             const primitive_id& input_grad,
                             const primitive_id& input,
                             const std::vector<primitive_id>& weights,
                             tensor stride,
                             tensor input_offset,
                             tensor dilation,
                             const primitive_id& conv_grad = "",
                             const padding& output_padding = padding())
        : primitive_base(id, {input_grad, input}, output_padding),
          conv_grad(conv_grad),
          stride(stride),
          input_offset(input_offset),
          dilation(dilation),
          output_grad_w(false),
          weights(weights),
          bias(std::vector<primitive_id>(0)),
          prev_weights_grad(std::vector<primitive_id>(0)),
          prev_bias_grad(std::vector<primitive_id>(0)) {}

    /// @brief Constructs convolution_grad_weights primitive with momentum optimizer.
    /// @param id This primitive id.
    /// @param input Input gradient primitive id.
    /// @param input Input primitive id from convolution forward pass.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data. Provide empty vector if using next parameters without bias.
    /// @param prev_weights_grad List of primitive ids which contains weights gradient data calculated in previous iteration. Used in momentum optimizer.
    /// @param prev_bias_grad List of primitive ids which contains bias gradient data calculated in previous iteration. Used in momentum optimizer.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution_grad_weights window should start calculations.
    /// @param dilation Defines dilation size.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param conv_grad Id of primitive which uses weights and biases updated in this primitive.
    /// This is for correct order of calculating. Leave empty if primitive is last in backward pass.
    convolution_grad_weights(const primitive_id& id,
                             const primitive_id& input_grad,
                             const primitive_id& input,
                             const std::vector<primitive_id>& weights,
                             const std::vector<primitive_id>& bias,
                             const std::vector<primitive_id>& prev_weights_grad,
                             const std::vector<primitive_id>& prev_bias_grad,
                             tensor stride = {1, 1, 1, 1},
                             tensor input_offset = {0, 0, 0, 0},
                             tensor dilation = {1, 1, 1, 1},
                             const primitive_id& conv_grad = "",
                             const padding& output_padding = padding())
        : primitive_base(id, {input_grad, input}, output_padding),
          conv_grad(conv_grad),
          stride(stride),
          input_offset(input_offset),
          dilation(dilation),
          output_grad_w(false),
          weights(weights),
          bias(bias),
          prev_weights_grad(prev_weights_grad),
          prev_bias_grad(prev_bias_grad) {}

    /// @brief Primitive id containing convolution gradient data.
    primitive_id conv_grad;
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    tensor stride;
    /// @brief Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution_grad_weights window should start calculations.
    tensor input_offset;
    /// @brief Defines gaps in the input - dilation rate k=1 is normal convolution, k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    tensor dilation;
    /// @brief Should primitive give weights gradient (delta) as an output
    bool output_grad_w;
    /// @brief List of primitive ids containing weights data.
    const primitive_id_arr weights;
    /// @brief List of primitive ids containing bias data.
    const primitive_id_arr bias;
    /// @brief Array of primitive ids containing weights gradient data calculated in previous iteration.
    /// Amount of primitives and their memory sizes should be same as weights.
    const primitive_id_arr prev_weights_grad;
    /// @brief Array of primitive ids containing bias gradient data calculated in previous iteration.
    /// Amount of primitives and their memory sizes should be same as biases.
    const primitive_id_arr prev_bias_grad;

    /// @brief On how many cards split the computation to.
    int32_t split() const { return static_cast<int32_t>(weights.size()); }

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.reserve(weights.size() + bias.size() + !conv_grad.empty() + prev_weights_grad.size() +
                    prev_bias_grad.size());
        for (auto& w : weights) ret.push_back(std::ref(w));
        for (auto& b : bias) ret.push_back(std::ref(b));

        for (auto& g : prev_weights_grad) ret.push_back(std::ref(g));
        for (auto& g : prev_bias_grad) ret.push_back(std::ref(g));
        if (!conv_grad.empty())
            ret.push_back(conv_grad);

        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
