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
#include "../C/fused_conv_eltwise.h"
#include "api/CPP/primitive.hpp"
#include "api/CPP/eltwise.hpp"

namespace cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs forward spatial convolution with fused eltwise and optionally Relu.
struct fused_conv_eltwise : public primitive_base<fused_conv_eltwise, CLDNN_PRIMITIVE_DESC(fused_conv_eltwise)>
{
    CLDNN_DECLARE_PRIMITIVE(fused_conv_eltwise)

    /// @brief Constructs fused_conv_eltwise primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data.
    /// @param w_quantization_factor List of primitive ids containing weights quanitization factors per output feature map.
    /// @param output_calibration_factors List of primitive ids output containing calibration factors per output feature map.
    /// @param i_quantization_factor Input quantization factor
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer, where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution, k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    fused_conv_eltwise(
        const primitive_id& id,
        const primitive_id& input,
        const primitive_id& input2,
        eltwise_mode mode,
        const std::vector<primitive_id>& weights,
        const std::vector<primitive_id>& bias,
        const std::vector<primitive_id>& conv_w_quantization_factor,
        const std::vector<primitive_id>& conv_output_calibration_factors,
        const float conv_i_quantization_factor,
        const primitive_id& eltw_output_calibration_factors,
        const std::vector<tensor>& eltw_stride,
        tensor stride = { 1, 1, 1, 1 },
        tensor input_offset = { 0,0,0,0 },
        tensor dilation = { 1, 1, 1, 1 },
        bool conv_with_activation = false,
        float conv_activation_slp = 0.0f,
        bool eltw_with_activation = false,
        float eltw_activation_slp = 0.0f,
        const padding& output_padding = padding()
    )
        :primitive_base(id, { input, input2 }, output_padding)
        , conv(_conv_weights.cpp_ids, _conv_bias.cpp_ids, _conv_weights_quantization_factors.cpp_ids, _conv_output_calibration_factors.cpp_ids)
        , eltw(eltw_output_calibration_factors)
        , _conv_weights(weights)
        , _conv_bias(bias)
        , _conv_weights_quantization_factors(conv_w_quantization_factor)
        , _conv_output_calibration_factors(conv_output_calibration_factors)
    {

        conv.input_quantization_factor = conv_i_quantization_factor;
        conv.output_quantization_factor = 1.0f;

        conv.input_offset = input_offset;
        conv.stride = stride;
        conv.dilation = dilation;
        conv.with_activation = conv_with_activation;
        conv.activation_negative_slope = conv_activation_slp;
        conv.with_output_size = false;

        eltw.mode = mode;
        eltw.with_activation = eltw_with_activation;
        eltw.activation_negative_slope = eltw_activation_slp;
        eltw.stride = eltw_stride;

        if ((bias.size() != 0) && (weights.size() != bias.size()))
            throw std::runtime_error("convolution's weights/bias count does not match");
        if (conv.output_calibration_factors.size())
        {
            if ((weights.size() != 0) && (weights.size() != conv.weights_quantization_factors.size()))
                throw std::runtime_error("convolution's weights count does not match quantization factors count");
        }
    }

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{convolution}
    fused_conv_eltwise(const dto* dto)
        :primitive_base(dto)
        , conv(_conv_weights.cpp_ids, _conv_bias.cpp_ids, _conv_weights_quantization_factors.cpp_ids, _conv_output_calibration_factors.cpp_ids)
        , eltw(dto->eltw.output_calibration_factors)
        , _conv_weights(dto->conv.weights)
        , _conv_bias(dto->conv.bias)
        , _conv_weights_quantization_factors(dto->conv.weights_quantization_factors)
        , _conv_output_calibration_factors(dto->conv.output_calibration_factors)
        , _eltw_stride(tensor_vector_to_cldnn_vector(eltw.stride))
    {
        conv.input_quantization_factor = dto->conv.input_quantization_factor;
        conv.output_quantization_factor = dto->conv.output_quantization_factor;
        conv.input_offset = dto->conv.input_offset;
        conv.stride = dto->conv.stride;
        conv.dilation = dto->conv.dilation;
        conv.with_activation = dto->conv.with_activation != 0;
        conv.activation_negative_slope = dto->conv.activation_negative_slope;
        conv.with_output_size = dto->conv.with_output_size != 0;
        conv.output_size = dto->conv.output_size;

        second_input_in_output = dto->second_input_in_output;

        if (!dto->conv.split || (conv.weights.size() != conv.bias.size() && conv.bias.size() != 0) || dto->conv.split != conv.weights.size())
            throw std::invalid_argument("Invalid convolution dto: bad split value");
    }

    struct conv_data
    {
        /// @brief List of primitive ids containing weights data.
        fixed_size_vector_ref weights;
        /// @brief List of primitive ids containing bias data.
        fixed_size_vector_ref bias;
        /// @brief List of primitive ids containing weights quanitization factors per output feature map.
        fixed_size_vector_ref weights_quantization_factors;
        /// @brief List of primitive ids containing output quanitization factors per output feature map for convolution.
        fixed_size_vector_ref output_calibration_factors;
        /// @brief Input quantization factor for convolution
        float input_quantization_factor;
        /// @brief Output quantization factor for convolution
        float output_quantization_factor;
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

        conv_data(const fixed_size_vector_ref& weights,
            const fixed_size_vector_ref& bias,
            const fixed_size_vector_ref& weights_quantization_factors,
            const fixed_size_vector_ref& output_calibration_factors
        ) : weights(weights),
            bias(bias),
            weights_quantization_factors(weights_quantization_factors),
            output_calibration_factors(output_calibration_factors)
        {}
    } conv;

    struct eltw_data
    {
        /// @brief Primitive id containing output quanitization factors per output feature map.
        primitive_id output_calibration_factors;
        /// @brief Output quantization factor for eltwise
        float output_quantization_factor;
        /// @param mode Eltwise mode.
        eltwise_mode mode;
        /// @brief Enable Relu activation.
        bool with_activation;
        /// @brief Relu activation slope.
        float activation_negative_slope;
        /// @brief Defines shift in input buffers between adjacent calculations of output values.
        std::vector<tensor> stride;

        eltw_data(const primitive_id& output_calibration_factors)
            : output_calibration_factors(output_calibration_factors)
        {}
    } eltw;

    /// @brief On how many cards split the computation to.
    int32_t split() const { return static_cast<int32_t>(conv.weights.size()); }

    /// @brief Is optimization that output contains data from second input ON ?
    bool second_input_in_output = false;
protected:
    primitive_id_arr _conv_weights;
    primitive_id_arr _conv_bias;
    primitive_id_arr _conv_weights_quantization_factors;
    primitive_id_arr _conv_output_calibration_factors;

    std::vector<cldnn_tensor> _eltw_stride;

    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override
    {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.reserve(conv.weights.size()
            + conv.bias.size()
            + conv.weights_quantization_factors.size()
            + conv.output_calibration_factors.size()
            + (eltw.output_calibration_factors.empty() ? 0 : 1));

        for (auto& w : conv.weights)
            ret.push_back(w);
        for (auto& b : conv.bias)
            ret.push_back(b);
        for (auto& q : conv.weights_quantization_factors)
            ret.push_back(q);
        for (auto& q : conv.output_calibration_factors)
            ret.push_back(q);

        if (!eltw.output_calibration_factors.empty())
            ret.push_back(eltw.output_calibration_factors);

        return ret;
    }

    void update_dto(dto& dto) const override
    {
        dto.conv.weights = _conv_weights.ref();
        dto.conv.bias = _conv_bias.ref();
        dto.conv.weights_quantization_factors = _conv_weights_quantization_factors.ref();
        dto.conv.output_calibration_factors = _conv_output_calibration_factors.ref();
        dto.conv.input_quantization_factor = conv.input_quantization_factor;
        dto.conv.output_quantization_factor = conv.output_quantization_factor;
        dto.conv.input_offset = conv.input_offset;
        dto.conv.stride = conv.stride;
        dto.conv.split = split();
        dto.conv.with_activation = conv.with_activation;
        dto.conv.activation_negative_slope = conv.activation_negative_slope;
        dto.conv.dilation = conv.dilation;
        dto.conv.with_output_size = conv.with_output_size;
        dto.conv.output_size = conv.output_size;

        dto.eltw.output_calibration_factors = eltw.output_calibration_factors.c_str();
        dto.eltw.output_quantization_factor = eltw.output_quantization_factor;
        dto.eltw.mode = static_cast<cldnn_eltwise_mode>(eltw.mode);
        dto.eltw.with_activation = eltw.with_activation;
        dto.eltw.activation_negative_slope = eltw.activation_negative_slope;
        dto.eltw.stride = tensor_vector_to_arr(_eltw_stride);

        dto.second_input_in_output = second_input_in_output;
    }
};
/// @}
/// @}
/// @}
}