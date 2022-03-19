// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"

#include "openvino/core/strides.hpp"
#include "openvino/core/coordinate_diff.hpp"
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
    /// @param pad Defines logical pad value added to input tensor
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id>& weights,
                  const std::vector<primitive_id>& bias,
                  ov::Strides stride = {1, 1},
                  ov::CoordinateDiff pad = {0, 0},
                  const primitive_id& ext_prim_id = "",
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          pad(pad),
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
    /// @param pad Defines logical pad value added to input tensor
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id>& weights,
                  const std::vector<primitive_id>& bias,
                  uint32_t groups,
                  ov::Strides stride = {1, 1},
                  ov::CoordinateDiff pad = {0, 0},
                  const primitive_id& ext_prim_id = "",
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          pad(pad),
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
    /// @param pad Defines logical pad value added to input tensor
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id>& weights,
                  ov::Strides stride = {1, 1},
                  ov::CoordinateDiff pad = {0, 0},
                  const primitive_id& ext_prim_id = "",
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          pad(pad),
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
    /// @param pad Defines logical pad value added to input tensor
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id> &weights,
                  uint32_t groups,
                  ov::Strides stride = {1, 1},
                  ov::CoordinateDiff pad = {0, 0},
                  const primitive_id& ext_prim_id = "",
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          pad(pad),
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
    /// @param pad Defines logical pad value added to input tensor
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id>& weights,
                  const std::vector<primitive_id>& bias,
                  ov::Strides stride,
                  ov::CoordinateDiff pad,
                  tensor output_size,
                  const primitive_id& ext_prim_id = "",
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          pad(pad),
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
    /// @param pad Defines logical pad value added to input tensor
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id>& weights,
                  const std::vector<primitive_id>& bias,
                  uint32_t groups,
                  ov::Strides stride,
                  ov::CoordinateDiff pad,
                  tensor output_size,
                  bool grouped_weights_shape,
                  const primitive_id& ext_prim_id = "",
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          pad(pad),
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
    /// @param pad Defines logical pad value added to input tensor
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    deconvolution(const primitive_id& id,
                  const primitive_id& input,
                  const std::vector<primitive_id>& weights,
                  ov::Strides stride,
                  ov::CoordinateDiff pad,
                  tensor output_size,
                  const primitive_id& ext_prim_id = "",
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          pad(pad),
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
    /// @param pad Defines logical pad value added to input tensor
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
                                                 ov::Strides stride = {1, 1},
                                                 ov::CoordinateDiff pad = {0, 0},
                                                 const primitive_id& ext_prim_id = "",
                                                 const padding& output_padding = padding()) {
        return deconvolution(id,
                             input,
                             weights,
                             bias,
                             stride,
                             pad,
                             output_size,
                             ext_prim_id,
                             output_padding);
    }

    /// @brief Constructs deconvolution primitive (w/o bias; computes input paddings to match output size).
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param pad Defines logical pad value added to input tensor
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    /// @return Deconvolution primitive with specified settings.
    static deconvolution create_with_output_size(const primitive_id& id,
                                                 const primitive_id& input,
                                                 const std::vector<primitive_id>& weights,
                                                 tensor output_size,
                                                 ov::Strides stride = {1, 1},
                                                 ov::CoordinateDiff pad = {0, 0},
                                                 const primitive_id& ext_prim_id = "",
                                                 const padding& output_padding = padding())     {
        return deconvolution(id,
                             input,
                             weights,
                             stride,
                             pad,
                             output_size,
                             ext_prim_id,
                             output_padding);
    }

    /// @brief Defines logical pad value added to input tensor.
    ov::CoordinateDiff pad;
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    ov::Strides stride;
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
