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
#include "../C/fused_conv_bn_scale.h"
#include "api/CPP/primitive.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Primitives that fuses convolution, batch norm, scale and optionally Relu.
struct fused_conv_bn_scale : public primitive_base<fused_conv_bn_scale, CLDNN_PRIMITIVE_DESC(fused_conv_bn_scale)> {
    CLDNN_DECLARE_PRIMITIVE(fused_conv_bn_scale)

    /// @brief Constructs convolution primitive fused with batch norm and scale.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data.
    /// @param epsilon Small number to protect from 0 dividing.
    /// @param scale_input Scale input primitive id with values needed for product computation. Used in fused scale part.
    /// @param scale_bias Primitive id containing bias data for fused scale part.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param inv_variance Primitive id containing inverted variance calculated in this primitive. Used in fused batch norm part.
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    fused_conv_bn_scale(const primitive_id& id,
                        const primitive_id& input,
                        const std::vector<primitive_id>& weights,
                        const std::vector<primitive_id>& bias,
                        float epsilon,
                        const primitive_id& scale_input,
                        const primitive_id& scale_bias = "",
                        tensor stride = {1, 1, 1, 1},
                        tensor dilation = {1, 1, 1, 1},
                        tensor input_offset = {0, 0, 0, 0},
                        const primitive_id& inv_variance = "",
                        bool with_activation = false,
                        float activation_slp = 0.0f,
                        const padding& output_padding = padding())
        : primitive_base(id, {input, scale_input}, output_padding),
          weights(_weights.cpp_ids),
          bias(_bias.cpp_ids),
          input_offset(input_offset),
          stride(stride),
          dilation(dilation),
          with_activation(with_activation),
          activation_negative_slope(activation_slp),
          with_output_size(false),
          scale_bias(scale_bias),
          inv_variance(inv_variance),
          epsilon(epsilon),
          _weights(weights),
          _bias(bias) {
        if ((bias.size() != 0) && (weights.size() != bias.size()))
            throw std::runtime_error("convolution's weights/bias count does not match");
    }

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{fused_conv_bn_scale}
    fused_conv_bn_scale(const dto* dto)
        : primitive_base(dto),
          weights(_weights.cpp_ids),
          bias(_bias.cpp_ids),
          input_offset(dto->input_offset),
          stride(dto->stride),
          dilation(dto->dilation),
          with_activation(dto->with_activation != 0),
          activation_negative_slope(dto->activation_negative_slope),
          scale_bias(dto->scale_bias),
          inv_variance(dto->inv_variance),
          epsilon(dto->epsilon),
          _weights(dto->weights),
          _bias(dto->bias) {
        if (!dto->split || (weights.size() != bias.size() && bias.size() != 0) || dto->split != weights.size())
            throw std::invalid_argument("Invalid convolution dto: bad split value");
    }

    /// @brief List of primitive ids containing weights data.
    fixed_size_vector_ref weights;
    /// @brief List of primitive ids containing bias data.
    fixed_size_vector_ref bias;
    /// @brief Defines a shift, relative to (0,0) position of the input buffer, where (0,0) point of the convolution window should start calculations.
    tensor input_offset;
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    tensor stride;
    /// @brief Defines gaps in the input - dilation rate k=1 is normal convolution, k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    tensor dilation;
    /// @brief Enable Relu activation.
    bool with_activation;
    /// @brief Relu activation slope.
    float activation_negative_slope;
    /// @brief Indicates that the primitive has user-defined output size (non-zero value).
    bool with_output_size;
    /// @brief User-defined output data size of the primitive (w/o padding).
    tensor output_size;
    /// @brief Primitive id containing scale bias data for fused convolution.
    primitive_id scale_bias;
    /// @brief Primitive id containing inverted variance used in future gradient computing for fused convolution.
    primitive_id inv_variance;
    /// @brief Epsilon for fused convolution.
    float epsilon;
    /// @brief On how many cards split the computation to.
    int32_t split() const { return static_cast<int32_t>(weights.size()); }

protected:
    primitive_id_arr _weights;
    primitive_id_arr _bias;

    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.reserve(weights.size() + bias.size() + !scale_bias.empty() + !inv_variance.empty());
        for (auto& w : weights) ret.push_back(w);
        for (auto& b : bias) ret.push_back(b);
        if (!scale_bias.empty())
            ret.push_back(scale_bias);
        if (!inv_variance.empty())
            ret.push_back(inv_variance);
        return ret;
    }

    void update_dto(dto& dto) const override {
        dto.weights = _weights.ref();
        dto.bias = _bias.ref();
        dto.input_offset = input_offset;
        dto.stride = stride;
        dto.dilation = dilation;
        dto.split = split();
        dto.with_activation = with_activation;
        dto.activation_negative_slope = activation_negative_slope;
        dto.epsilon = epsilon;
        dto.inv_variance = inv_variance.c_str();
        dto.scale_bias = scale_bias.c_str();
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn