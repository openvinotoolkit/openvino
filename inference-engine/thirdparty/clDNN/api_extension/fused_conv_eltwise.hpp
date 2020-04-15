/*
// Copyright (c) 2018-2019 Intel Corporation
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
#include "api/primitive.hpp"
#include "api/eltwise.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs forward spatial convolution with fused eltwise and optionally Relu.
struct fused_conv_eltwise : public primitive_base<fused_conv_eltwise> {
    CLDNN_DECLARE_PRIMITIVE(fused_conv_eltwise)

    /// @brief Constructs fused_conv_eltwise primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data.
    /// @param w_quantization_factor List of primitive ids containing weights quanitization factors per output feature map.
    /// @param output_calibration_factors List of primitive ids output containing calibration factors per output feature map.
    /// @param i_quantization_factor Input quantization factor
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution, k=2 means skipping one pixel per input,
    /// k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_data_type Precision of the output after eltwise activaiton. Might be less precise than internal computations
    /// (e.g. i8 input, i32 accumulator/activation, u8 output).
    fused_conv_eltwise(const primitive_id& id,
                       const primitive_id& input,
                       const primitive_id& input2,
                       eltwise_mode mode,
                       const std::vector<primitive_id>& weights,
                       const std::vector<primitive_id>& bias,
                       const std::vector<primitive_id>& conv_w_quantization_factor,
                       const std::vector<primitive_id>& conv_output_calibration_factors,
                       const float conv_i_quantization_factor,
                       const float non_conv_scale,
                       const primitive_id& eltw_output_calibration_factors,
                       const std::vector<tensor>& eltw_stride,
                       tensor stride = {1, 1, 1, 1},
                       tensor input_offset = {0, 0, 0, 0},
                       tensor dilation = {1, 1, 1, 1},
                       bool conv_with_activation = false,
                       float conv_activation_slp = 0.0f,
                       bool eltw_with_activation = false,
                       float eltw_activation_slp = 0.0f,
                       const padding& output_padding = padding(),
                       optional_data_type output_data_type = {})
        : primitive_base(id, {input, input2}, output_padding, output_data_type),
          conv((primitive_id_arr)weights,
              (primitive_id_arr)bias,
              (primitive_id_arr)conv_w_quantization_factor,
              (primitive_id_arr)conv_output_calibration_factors),
          eltw(eltw_output_calibration_factors),
          non_conv_scale(non_conv_scale),
          conv_weights(weights),
          conv_bias(bias),
          conv_weights_quantization_factors(conv_w_quantization_factor),
          conv_output_calibration_factors(conv_output_calibration_factors) {
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
        if (conv.output_calibration_factors.size()) {
            if ((weights.size() != 0) && (weights.size() != conv.weights_quantization_factors.size()))
                throw std::runtime_error("convolution's weights count does not match quantization factors count");
        }
    }

    struct conv_data {
        /// @brief List of primitive ids containing weights data.
        const primitive_id_arr weights;
        /// @brief List of primitive ids containing bias data.
        const primitive_id_arr bias;
        /// @brief List of primitive ids containing weights quanitization factors per output feature map.
        const primitive_id_arr weights_quantization_factors;
        /// @brief List of primitive ids containing output quanitization factors per output feature map for convolution.
        const primitive_id_arr output_calibration_factors;
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

        conv_data(const primitive_id_arr& weights,
                  const primitive_id_arr& bias,
                  const primitive_id_arr& weights_quantization_factors,
                  const primitive_id_arr& output_calibration_factors)
            : weights(weights),
              bias(bias),
              weights_quantization_factors(weights_quantization_factors),
              output_calibration_factors(output_calibration_factors) {}
    } conv;

    struct eltw_data {
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
        explicit eltw_data(const primitive_id& output_calibration_factors)
            : output_calibration_factors(output_calibration_factors) {}
    } eltw;

    /// @brief On how many cards split the computation to.
    int32_t split() const { return static_cast<int32_t>(conv.weights.size()); }

    // FIXME: In fact, that should be needed for any EltWise primitive, not
    // only the fused one. What's more important, these scales should be
    // separate for different inputs and probably per-channel, not per
    // primitive.
    //
    // I'm only needing a scalar for my particular task, so let's hack like
    // this in the meantime. The final design is still to be investigated.
    float non_conv_scale = 1.0f;

    /// @brief Is optimization that output contains data from second input ON ?
    bool second_input_in_output = false;
    bool depth_to_space_already_fused = false;

protected:
    const primitive_id_arr conv_weights;
    const primitive_id_arr conv_bias;
    const primitive_id_arr conv_weights_quantization_factors;
    const primitive_id_arr conv_output_calibration_factors;

    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.reserve(conv.weights.size() + conv.bias.size() + conv.weights_quantization_factors.size() +
                    conv.output_calibration_factors.size() + (eltw.output_calibration_factors.empty() ? 0 : 1));

        for (auto& w : conv.weights) ret.push_back(std::ref(w));
        for (auto& b : conv.bias) ret.push_back(std::ref(b));
        for (auto& q : conv.weights_quantization_factors) ret.push_back(std::ref(q));
        for (auto& q : conv.output_calibration_factors) ret.push_back(std::ref(q));

        if (!eltw.output_calibration_factors.empty())
            ret.push_back(eltw.output_calibration_factors);

        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
