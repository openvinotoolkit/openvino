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
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs forward spatial convolution with weight sharing.
/// Also supports built-in Relu @CLDNN_PRIMITIVE_DESC{activation} available by setting it in arguments.
/// @details Parameters are defined in context of "direct" convolution, but actual algorithm is not implied.
struct convolution : public primitive_base<convolution> {
    CLDNN_DECLARE_PRIMITIVE(convolution)

    /// @brief Constructs convolution primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param groups Number of filter groups.
    /// @param bias List of primitive ids containing bias data.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    /// @param output_type User-defined output data type of the primitive.
    /// @param grouped_weights_shape Defines if weights tensor has explicit group dimension.
    /// This parameter affects how bfzyx and bfwzyx format on weights is converted:
    /// bfzyx -> oizyx (grouped_weights_shape=false) or goiyx (grouped_weights_shape=true)
    /// bfwzyx -> error (grouped_weights_shape=false) or goizyx (grouped_weights_shape=true)
    /// If weights already have (g)oi(z)yx format, then this flag has no effect
    convolution(const primitive_id& id,
                const primitive_id& input,
                const std::vector<primitive_id>& weights,
                const std::vector<primitive_id>& bias,
                uint32_t groups,
                tensor stride,
                tensor input_offset,
                tensor dilation,
                tensor output_size,
                data_types output_type,
                bool grouped_weights_shape,
                const padding& output_padding = padding())
            : primitive_base(id, {input}, output_padding, optional_data_type{output_type}),
              input_offset(input_offset),
              stride(stride),
              dilation(dilation),
              with_output_size(true),
              output_size(output_size),
              groups(groups),
              deformable_groups(1),
              padding_above(tensor(0)),
              padding_below(tensor(0)),
              deformable_mode(false),
              grouped_weights_shape(grouped_weights_shape),
              weights(weights),
              bias(bias),
              weights_zero_points(std::vector<primitive_id>(0)),
              activations_zero_points(std::vector<primitive_id>(0)),
              compensation(std::vector<primitive_id>(0))  {
        if ((bias.size() != 0) && (weights.size() != bias.size()))
            throw std::runtime_error("convolution's weights/bias count does not match");
    }

    /// @brief Constructs convolution primitive (computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data.
    /// @param w_zero_point List of primitive ids containing weights zero points.
    /// @param a_zero_point List of primitive ids containing activations zero points.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    convolution(const primitive_id& id,
                const primitive_id& input,
                const std::vector<primitive_id>& weights,
                const std::vector<primitive_id>& bias,
                const std::vector<primitive_id>& w_zero_point,
                const std::vector<primitive_id>& a_zero_point,
                uint32_t groups,
                data_types output_data_type,
                tensor stride,
                tensor input_offset,
                tensor dilation,
                tensor output_size,
                bool grouped_weights_shape,
                const padding& output_padding = padding())
            : primitive_base(id, {input}, output_padding, optional_data_type{output_data_type}),
              input_offset(input_offset),
              stride(stride),
              dilation(dilation),
              with_output_size(true),
              output_size(output_size),
              groups(groups),
              deformable_groups(1),
              padding_above(tensor(0)),
              padding_below(tensor(0)),
              deformable_mode(false),
              grouped_weights_shape(grouped_weights_shape),
              weights(weights),
              bias(bias),
              weights_zero_points(w_zero_point),
              activations_zero_points(a_zero_point),
              compensation(std::vector<primitive_id>(0)) {
        if ((!bias.empty()) && (weights.size() != bias.size()))
            throw std::runtime_error("convolution's weights/bias count does not match");
        if ((!w_zero_point.empty()) && (weights.size() != w_zero_point.size()))
            throw std::runtime_error("convolution's weights/w_zero_points count does not match");
        if ((!a_zero_point.empty()) && (weights.size() != a_zero_point.size()))
            throw std::runtime_error("convolution's weights/a_zero_points count does not match");
    }

    /// @brief Constructs convolution primitive (computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data.
    /// @param w_zero_point List of primitive ids containing weights zero points.
    /// @param a_zero_point List of primitive ids containing activations zero points.
    /// @param compensation List of primitive ids containing activations precalculated compensations for optimized asymmetric quantization.
    /// It works as bias, but can be skipped by the kernel if it performs direct zero-points subtraction
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    convolution(const primitive_id& id,
                const primitive_id& input,
                const std::vector<primitive_id>& weights,
                const std::vector<primitive_id>& bias,
                const std::vector<primitive_id>& w_zero_point,
                const std::vector<primitive_id>& a_zero_point,
                const std::vector<primitive_id>& compensation,
                uint32_t groups,
                data_types output_data_type,
                tensor stride,
                tensor input_offset,
                tensor dilation,
                tensor output_size,
                bool grouped_weights_shape,
                const padding& output_padding = padding())
            : primitive_base(id, {input}, output_padding, optional_data_type{output_data_type}),
              input_offset(input_offset),
              stride(stride),
              dilation(dilation),
              with_output_size(true),
              output_size(output_size),
              groups(groups),
              deformable_groups(1),
              padding_above(tensor(0)),
              padding_below(tensor(0)),
              deformable_mode(false),
              grouped_weights_shape(grouped_weights_shape),
              weights(weights),
              bias(bias),
              weights_zero_points(w_zero_point),
              activations_zero_points(a_zero_point),
              compensation(compensation) {
        if ((!bias.empty()) && (weights.size() != bias.size()))
            throw std::runtime_error("convolution's weights/bias count does not match");
        if ((!w_zero_point.empty()) && (weights.size() != w_zero_point.size()))
            throw std::runtime_error("convolution's weights/w_zero_points count does not match");
        if ((!a_zero_point.empty()) && (weights.size() != a_zero_point.size()))
            throw std::runtime_error("convolution's weights/a_zero_points count does not match");
    }

    /// @brief Constructs convolution primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    convolution(const primitive_id& id,
                const primitive_id& input,
                const std::vector<primitive_id>& weights,
                const std::vector<primitive_id>& bias,
                tensor stride = {1, 1, 1, 1},
                tensor input_offset = tensor(0),
                tensor dilation = {1, 1, 1, 1},
                const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          dilation(dilation),
          with_output_size(false),
          groups(1),
          deformable_groups(1),
          padding_above(tensor(0)),
          padding_below(tensor(0)),
          deformable_mode(false),
          grouped_weights_shape(false),
          weights(weights),
          bias(bias),
          weights_zero_points(std::vector<primitive_id>(0)),
          activations_zero_points(std::vector<primitive_id>(0)),
          compensation(std::vector<primitive_id>(0)) {
        if ((bias.size() != 0) && (weights.size() != bias.size()))
            throw std::runtime_error("convolution's weights/bias count does not match");
    }

    /// @brief Constructs convolution primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param padding_above Defines a padding added to input image on left (x axis) and top (y axis).
    /// @param padding_below Defines a padding added to input image on right (x axis) and bottom (y axis).
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    convolution(const primitive_id& id,
                const primitive_id& input,
                const std::vector<primitive_id>& weights,
                const std::vector<primitive_id>& bias,
                tensor stride,
                tensor input_offset,
                tensor dilation,
                tensor padding_above,
                tensor padding_below,
                const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          dilation(dilation),
          with_output_size(false),
          groups(1),
          deformable_groups(1),
          padding_above(padding_above),
          padding_below(padding_below),
          deformable_mode(false),
          grouped_weights_shape(false),
          weights(weights),
          bias(bias),
          weights_zero_points(std::vector<primitive_id>(0)),
          activations_zero_points(std::vector<primitive_id>(0)),
          compensation(std::vector<primitive_id>(0)) {
        if ((bias.size() != 0) && (weights.size() != bias.size()))
            throw std::runtime_error("convolution's weights/bias count does not match");
    }

    /// @brief Constructs convolution primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param groups Number of filter groups.
    /// @param bias List of primitive ids containing bias data.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param padding_above Defines a padding added to input image on left (x axis) and top (y axis).
    /// @param padding_below Defines a padding added to input image on right (x axis) and bottom (y axis).
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    convolution(const primitive_id& id,
                const primitive_id& input,
                const std::vector<primitive_id>& weights,
                const std::vector<primitive_id>& bias,
                uint32_t groups,
                tensor stride,
                tensor input_offset,
                tensor dilation,
                tensor padding_above,
                tensor padding_below,
                const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          dilation(dilation),
          with_output_size(false),
          groups(groups),
          deformable_groups(1),
          padding_above(padding_above),
          padding_below(padding_below),
          deformable_mode(false),
          grouped_weights_shape(false),
          weights(weights),
          bias(bias),
          weights_zero_points(std::vector<primitive_id>(0)),
          activations_zero_points(std::vector<primitive_id>(0)),
          compensation(std::vector<primitive_id>(0)) {
        if ((bias.size() != 0) && (weights.size() != bias.size()))
            throw std::runtime_error("convolution's weights/bias count does not match");
    }

    /// @brief Constructs convolution primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param groups Number of filter groups.
    /// @param bias List of primitive ids containing bias data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    convolution(const primitive_id& id,
                const primitive_id& input,
                const std::vector<primitive_id>& weights,
                const std::vector<primitive_id>& bias,
                uint32_t groups,
                tensor stride = {1, 1, 1, 1},
                tensor input_offset = tensor(0),
                tensor dilation = {1, 1, 1, 1},
                bool grouped_weights_shape = false,
                const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          dilation(dilation),
          with_output_size(false),
          groups(groups),
          deformable_groups(1),
          padding_above(tensor(0)),
          padding_below(tensor(0)),
          deformable_mode(false),
          grouped_weights_shape(grouped_weights_shape),
          weights(weights),
          bias(bias),
          weights_zero_points(std::vector<primitive_id>(0)),
          activations_zero_points(std::vector<primitive_id>(0)),
          compensation(std::vector<primitive_id>(0)) {
        if ((bias.size() != 0) && (weights.size() != bias.size()))
            throw std::runtime_error("convolution's weights/bias count does not match");
        if ((groups > 1) && ((weights.size() != 1) || ((bias.size() != 0) && (bias.size() != 1))))
            throw std::runtime_error("grouped convolution's weights/bias count must be 1");
    }

    /// @brief Constructs convolution primitive (w/o bias).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    convolution(const primitive_id& id,
                const primitive_id& input,
                const std::vector<primitive_id>& weights,
                tensor stride = {1, 1, 1, 1},
                tensor input_offset = tensor(0),
                tensor dilation = {1, 1, 1, 1},
                bool grouped_weights_shape = false,
                const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          dilation(dilation),
          with_output_size(false),
          groups(1),
          deformable_groups(1),
          padding_above(tensor(0)),
          padding_below(tensor(0)),
          deformable_mode(false),
          grouped_weights_shape(grouped_weights_shape),
          weights(weights),
          bias(std::vector<primitive_id>(0)),
          weights_zero_points(std::vector<primitive_id>(0)),
          activations_zero_points(std::vector<primitive_id>(0)),
          compensation(std::vector<primitive_id>(0)) {}

    /// @brief Constructs convolution primitive (w/o bias).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param padding_above Defines a padding added to input image on left (x axis) and top (y axis).
    /// @param padding_below Defines a padding added to input image on right (x axis) and bottom (y axis).
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    convolution(const primitive_id& id,
                const primitive_id& input,
                const std::vector<primitive_id>& weights,
                tensor stride,
                tensor input_offset,
                tensor dilation,
                tensor padding_above,
                tensor padding_below,
                const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          dilation(dilation),
          with_output_size(false),
          groups(1),
          deformable_groups(1),
          padding_above(padding_above),
          padding_below(padding_below),
          deformable_mode(false),
          grouped_weights_shape(false),
          weights(weights),
          bias(std::vector<primitive_id>(0)),
          weights_zero_points(std::vector<primitive_id>(0)),
          activations_zero_points(std::vector<primitive_id>(0)),
          compensation(std::vector<primitive_id>(0)) {}

    /// @brief Constructs convolution primitive (w/o bias).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param groups Number of filter groups.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param padding_above Defines a padding added to input image on left (x axis) and top (y axis).
    /// @param padding_below Defines a padding added to input image on right (x axis) and bottom (y axis).
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    convolution(const primitive_id& id,
                const primitive_id& input,
                const std::vector<primitive_id>& weights,
                uint32_t groups,
                tensor stride,
                tensor input_offset,
                tensor dilation,
                tensor padding_above,
                tensor padding_below,
                const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          dilation(dilation),
          with_output_size(false),
          groups(groups),
          deformable_groups(1),
          padding_above(padding_above),
          padding_below(padding_below),
          deformable_mode(false),
          grouped_weights_shape(false),
          weights(weights),
          bias(std::vector<primitive_id>(0)),
          weights_zero_points(std::vector<primitive_id>(0)),
          activations_zero_points(std::vector<primitive_id>(0)),
          compensation(std::vector<primitive_id>(0)) {}

    /// @brief Constructs convolution primitive (w/o bias).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param groups Number of filter groups.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    convolution(const primitive_id& id,
                const primitive_id& input,
                const std::vector<primitive_id>& weights,
                uint32_t groups,
                tensor stride = {1, 1, 1, 1},
                tensor input_offset = tensor(0),
                tensor dilation = {1, 1, 1, 1},
                bool grouped_weights_shape = false,
                const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          dilation(dilation),
          with_output_size(false),
          groups(groups),
          deformable_groups(1),
          padding_above(tensor(0)),
          padding_below(tensor(0)),
          deformable_mode(false),
          grouped_weights_shape(grouped_weights_shape),
          weights(weights),
          bias(std::vector<primitive_id>(0)),
          weights_zero_points(std::vector<primitive_id>(0)),
          activations_zero_points(std::vector<primitive_id>(0)),
          compensation(std::vector<primitive_id>(0)) {}

    /// @brief Constructs convolution primitive (computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    convolution(const primitive_id& id,
                const primitive_id& input,
                const std::vector<primitive_id>& weights,
                const std::vector<primitive_id>& bias,
                tensor stride,
                tensor input_offset,
                tensor dilation,
                tensor output_size,
                const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          dilation(dilation),
          with_output_size(true),
          output_size(output_size),
          groups(1),
          deformable_groups(1),
          padding_above(tensor(0)),
          padding_below(tensor(0)),
          deformable_mode(false),
          grouped_weights_shape(false),
          weights(weights),
          bias(bias),
          weights_zero_points(std::vector<primitive_id>(0)),
          activations_zero_points(std::vector<primitive_id>(0)),
          compensation(std::vector<primitive_id>(0)) {
        if ((bias.size() != 0) && (weights.size() != bias.size()))
            throw std::runtime_error("convolution's weights/bias count does not match");
    }

    /// @brief Constructs convolution primitive (w/o bias; computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    convolution(const primitive_id& id,
                const primitive_id& input,
                const std::vector<primitive_id>& weights,
                tensor stride,
                tensor input_offset,
                tensor dilation,
                tensor output_size,
                const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          input_offset(input_offset),
          stride(stride),
          dilation(dilation),
          with_output_size(true),
          output_size(output_size),
          groups(1),
          deformable_groups(1),
          padding_above(tensor(0)),
          padding_below(tensor(0)),
          deformable_mode(false),
          grouped_weights_shape(false),
          weights(weights),
          bias(std::vector<primitive_id>(0)),
          weights_zero_points(std::vector<primitive_id>(0)),
          activations_zero_points(std::vector<primitive_id>(0)),
          compensation(std::vector<primitive_id>(0)) {}

    /*
    /// @brief Constructs convolution primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param groups Number of filter groups.
    /// @param bias List of primitive ids containing bias data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param deformable_groups Defines a number of deformable groups that splits trans input into several parts
    /// by channel dimension.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
*/
    convolution(const primitive_id& id,
                const primitive_id& input,
                const primitive_id& trans,
                const std::vector<primitive_id>& weights,
                const std::vector<primitive_id>& bias,
                uint32_t groups,
                uint32_t deformable_groups,
                tensor stride,
                tensor input_offset,
                tensor dilation,
                tensor output_size,
                const padding& output_padding = padding())
        : primitive_base(id, {input, trans}, output_padding),
          input_offset(input_offset),
          stride(stride),
          dilation(dilation),
          with_output_size(true),
          output_size(output_size),
          groups(groups),
          deformable_groups(deformable_groups),
          padding_above(tensor(0)),
          padding_below(tensor(0)),
          deformable_mode(true),
          grouped_weights_shape(false),
          weights(weights),
          bias(bias),
          weights_zero_points(std::vector<primitive_id>(0)),
          activations_zero_points(std::vector<primitive_id>(0)),
          compensation(std::vector<primitive_id>(0)) {
        if ((bias.size() != 0) && (weights.size() != bias.size()))
            throw std::runtime_error("convolution's weights/bias count does not match");
        if ((groups > 1) && ((weights.size() != 1) || ((bias.size() != 0) && (bias.size() != 1))))
            throw std::runtime_error("grouped convolution's weights/bias count must be 1");
    }

    /// @brief Constructs convolution primitive (computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    /// @return Convolution primitive with specified settings.
    static convolution create_with_output_size(const primitive_id& id,
                                               const primitive_id& input,
                                               const std::vector<primitive_id>& weights,
                                               const std::vector<primitive_id>& bias,
                                               tensor output_size,
                                               tensor stride = {1, 1, 1, 1},
                                               tensor input_offset = tensor(0),
                                               tensor dilation = {1, 1, 1, 1},
                                               const padding& output_padding = padding()) {
        return convolution(id,
                           input,
                           weights,
                           bias,
                           stride,
                           input_offset,
                           dilation,
                           output_size,
                           output_padding);
    }

    /// @brief Constructs convolution primitive (w/o bias; computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param input_offset Defines a shift, relative to (0,0) position of the input buffer,
    /// where (0,0) point of the convolution window should start calculations.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param with_activation Enable Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    /// @return Convolution primitive with specified settings.
    static convolution create_with_output_size(const primitive_id& id,
                                               const primitive_id& input,
                                               const std::vector<primitive_id>& weights,
                                               tensor output_size,
                                               tensor stride = {1, 1, 1, 1},
                                               tensor input_offset = tensor(0),
                                               tensor dilation = {1, 1, 1, 1},
                                               const padding& output_padding = padding()) {
        return convolution(id,
                           input,
                           weights,
                           stride,
                           input_offset,
                           dilation,
                           output_size,
                           output_padding);
    }

    /// @brief Defines a shift, relative to (0,0) position of the input buffer, where (0,0) point of the convolution window should start calculations.
    tensor input_offset;
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    tensor stride;
    /// @brief Defines gaps in the input - dilation rate k=1 is normal convolution, k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    tensor dilation;
    /// @brief Indicates that the primitive has user-defined output size (non-zero value).
    bool with_output_size;
    /// @brief User-defined output data size of the primitive (w/o padding).
    tensor output_size;
    /// @brief Number of feature groups (grouped convolution). If more than 1 then weights/bias count needs to be 1.
    uint32_t groups;
    /// @param deformable_groups Defines a number of deformable groups that splits trans input into several parts
    /// by channel dimension.
    uint32_t deformable_groups;
    /// @param padding_above Defines a padding added to input image on left (x axis) and top (y axis).
    tensor padding_above;
    /// @param padding_below Defines a padding added to input image on right (x axis) and bottom (y axis).
    tensor padding_below;
    /// @param deformable_mode.
    bool deformable_mode;
    /// @param grouped_weights_shape Defines if weights tensor has explicit group dimension.
    bool grouped_weights_shape;
    /// @brief List of primitive ids containing weights data.
    const primitive_id_arr weights;
    /// @brief List of primitive ids containing bias data.
    const primitive_id_arr bias;
    /// @brief List of primitive ids containing weights zero points.
    primitive_id_arr weights_zero_points;
    /// @brief List of primitive ids containing activations zero points.
    primitive_id_arr activations_zero_points;
    /// @brief List of primitive ids containing compensation.
    primitive_id_arr compensation;


    /// @brief On how many cards split the computation to.
    int32_t split() const { return static_cast<int32_t>(weights.size()); }

    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.reserve(weights.size() + bias.size() + weights_zero_points.size() +
                    activations_zero_points.size() + compensation.size());
        for (auto& w : weights) ret.push_back(std::ref(w));
        for (auto& b : bias) ret.push_back(std::ref(b));
        for (auto& q : weights_zero_points) ret.push_back(std::ref(q));
        for (auto& q : activations_zero_points) ret.push_back(std::ref(q));
        for (auto& q : compensation) ret.push_back(std::ref(q));
        return ret;
    }
};

struct deformable_interp : public primitive_base<deformable_interp> {
    CLDNN_DECLARE_PRIMITIVE(deformable_interp)

    deformable_interp(const primitive_id& id,
                      const primitive_id& input,
                      const primitive_id& trans,
                      uint32_t groups,
                      uint32_t deformable_groups,
                      tensor stride,
                      tensor input_offset,
                      tensor dilation,
                      tensor output_size,
                      tensor kernel_size,
                      const padding& output_padding = padding())
            : primitive_base(id, {input, trans}, output_padding),
              input_offset(input_offset),
              stride(stride),
              dilation(dilation),
              output_size(output_size),
              kernel_size(kernel_size),
              groups(groups),
              deformable_groups(deformable_groups),
              padding_above(tensor(0)),
              padding_below(tensor(0)) {}

    /// @brief Defines a shift, relative to (0,0) position of the input buffer, where (0,0) point of the convolution window should start calculations.
    tensor input_offset;
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    tensor stride;
    /// @brief Defines gaps in the input - dilation rate k=1 is normal convolution, k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    tensor dilation;
    /// @brief Size of output tensor.
    tensor output_size;
    /// @brief Size of weights tensor.
    tensor kernel_size;
    /// @brief Number of feature groups (grouped convolution). If more than 1 then weights/bias count needs to be 1.
    uint32_t groups;
    /// @param deformable_groups Defines a number of deformable groups that splits trans input into several parts
    /// by channel dimension.
    uint32_t deformable_groups;
    /// @param padding_above Defines a padding added to input image on left (x axis) and top (y axis).
    tensor padding_above;
    /// @param padding_below Defines a padding added to input image on right (x axis) and bottom (y axis).
    tensor padding_below;
};

struct deformable_conv : public primitive_base<deformable_conv> {
    CLDNN_DECLARE_PRIMITIVE(deformable_conv)

    deformable_conv(const primitive_id& id,
                    const primitive_id& input,
                    const std::vector<primitive_id>& weights,
                    const std::vector<primitive_id>& biases,
                    uint32_t groups,
                    tensor output_size,
                    const padding& output_padding = padding())
            : primitive_base(id, {input}, output_padding),
              output_size(output_size),
              groups(groups),
              weights(weights),
              bias(biases) {}

    /// @brief User-defined output data size of the primitive (w/o padding).
    tensor output_size;
    /// @brief Number of feature groups (grouped convolution). If more than 1 then weights/bias count needs to be 1.
    uint32_t groups;
    /// @brief List of primitive ids containing weights data.
    const primitive_id_arr weights;
    /// @brief List of primitive ids containing bias data.
    const primitive_id_arr bias;

    /// @brief On how many cards split the computation to.
    int32_t split() const { return static_cast<int32_t>(weights.size()); }

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
