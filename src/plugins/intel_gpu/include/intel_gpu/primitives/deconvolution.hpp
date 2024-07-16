// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

#include "openvino/core/strides.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include <vector>

namespace cldnn {

/// @brief Performs transposed convolution.
/// Also supports built-in Relu @ref activation available by setting it in arguments.
/// @details Deconvolution is similar to convolution layer with the weights flipped on the axis
/// and stride and input padding parameters used in opposite sense as in convolution.
struct deconvolution : public primitive_base<deconvolution> {
    CLDNN_DECLARE_PRIMITIVE(deconvolution)

    deconvolution() : primitive_base("", {}) {}

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
                  const input_info& input,
                  const std::vector<primitive_id>& weights,
                  const std::vector<primitive_id>& bias,
                  ov::Strides stride = {1, 1},
                  ov::CoordinateDiff pad = {0, 0},
                  ov::Strides dilations = {1, 1})
        : primitive_base(id, {input}),
          pad(pad),
          stride(stride),
          dilations(dilations),
          with_output_size(false),
          groups(1),
          pads_begin(pad.size(), 0),
          pads_end(pad.size(), 0),
          out_padding(pad.size(), 0),
          grouped_weights_shape(false),
          output_partial_shape({}),
          output_shape_id(""),
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
                  const input_info& input,
                  const std::vector<primitive_id>& weights,
                  const std::vector<primitive_id>& bias,
                  uint32_t groups,
                  ov::Strides stride = {1, 1},
                  ov::CoordinateDiff pad = {0, 0},
                  ov::Strides dilations = {1, 1})
        : primitive_base(id, {input}),
          pad(pad),
          stride(stride),
          dilations(dilations),
          with_output_size(false),
          groups(groups),
          pads_begin(pad.size(), 0),
          pads_end(pad.size(), 0),
          out_padding(pad.size(), 0),
          grouped_weights_shape(false),
          output_partial_shape({}),
          output_shape_id(""),
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
                  const input_info& input,
                  const std::vector<primitive_id>& weights,
                  ov::Strides stride = {1, 1},
                  ov::CoordinateDiff pad = {0, 0},
                  ov::Strides dilations = {1, 1})
        : primitive_base(id, {input}),
          pad(pad),
          stride(stride),
          dilations(dilations),
          with_output_size(false),
          groups(1),
          pads_begin(pad.size(), 0),
          pads_end(pad.size(), 0),
          out_padding(pad.size(), 0),
          grouped_weights_shape(false),
          output_partial_shape({}),
          output_shape_id(""),
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
                  const input_info& input,
                  const std::vector<primitive_id> &weights,
                  uint32_t groups,
                  ov::Strides stride = {1, 1},
                  ov::CoordinateDiff pad = {0, 0},
                  ov::Strides dilations = {1, 1})
        : primitive_base(id, {input}),
          pad(pad),
          stride(stride),
          dilations(dilations),
          with_output_size(false),
          groups(groups),
          pads_begin(pad.size(), 0),
          pads_end(pad.size(), 0),
          out_padding(pad.size(), 0),
          grouped_weights_shape(false),
          output_partial_shape({}),
          output_shape_id(""),
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
                  const input_info& input,
                  const std::vector<primitive_id>& weights,
                  const std::vector<primitive_id>& bias,
                  ov::Strides stride,
                  ov::CoordinateDiff pad,
                  ov::Strides dilations,
                  tensor output_size)
        : primitive_base(id, {input}),
          pad(pad),
          stride(stride),
          dilations(dilations),
          with_output_size(true),
          output_size(output_size),
          groups(1),
          pads_begin(pad.size(), 0),
          pads_end(pad.size(), 0),
          out_padding(pad.size(), 0),
          grouped_weights_shape(false),
          output_partial_shape({}),
          output_shape_id(""),
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
                  const input_info& input,
                  const std::vector<primitive_id>& weights,
                  const std::vector<primitive_id>& bias,
                  uint32_t groups,
                  ov::Strides stride,
                  ov::CoordinateDiff pad,
                  ov::Strides dilations,
                  tensor output_size,
                  bool grouped_weights_shape)
        : primitive_base(id, {input}),
          pad(pad),
          stride(stride),
          dilations(dilations),
          with_output_size(true),
          output_size(output_size),
          groups(groups),
          pads_begin(pad.size(), 0),
          pads_end(pad.size(), 0),
          out_padding(pad.size(), 0),
          grouped_weights_shape(grouped_weights_shape),
          output_partial_shape({}),
          output_shape_id(""),
          weights(weights),
          bias(bias) {}

    /// @brief Constructs deconvolution primitive with dynamic shape.
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
                  const input_info& input,
                  const std::vector<primitive_id>& weights,
                  const std::vector<primitive_id>& bias,
                  uint32_t groups,
                  ov::Strides stride,
                  ov::CoordinateDiff pad,
                  ov::Strides dilations,
                  ov::CoordinateDiff pads_begin,
                  ov::CoordinateDiff pads_end,
                  ov::CoordinateDiff out_padding,
                  bool grouped_weights_shape)
        : primitive_base(id, {input}),
          pad(pad),
          stride(stride),
          dilations(dilations),
          with_output_size(false),
          groups(groups),
          pads_begin(pads_begin),
          pads_end(pads_end),
          out_padding(out_padding),
          grouped_weights_shape(grouped_weights_shape),
          output_partial_shape({}),
          output_shape_id(""),
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
                  const input_info& input,
                  const std::vector<primitive_id>& weights,
                  ov::Strides stride,
                  ov::CoordinateDiff pad,
                  ov::Strides dilations,
                  tensor output_size)
        : primitive_base(id, {input}),
          pad(pad),
          stride(stride),
          dilations(dilations),
          with_output_size(true),
          output_size(output_size),
          groups(1),
          pads_begin(pad.size(), 0),
          pads_end(pad.size(), 0),
          out_padding(pad.size(), 0),
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
                                                 const input_info& input,
                                                 const std::vector<primitive_id>& weights,
                                                 const std::vector<primitive_id>& bias,
                                                 tensor output_size,
                                                 ov::Strides stride = {1, 1},
                                                 ov::CoordinateDiff pad = {0, 0},
                                                 ov::Strides dilations = {1, 1}) {
        return deconvolution(id,
                             input,
                             weights,
                             bias,
                             stride,
                             pad,
                             dilations,
                             output_size);
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
                                                 const input_info& input,
                                                 const std::vector<primitive_id>& weights,
                                                 tensor output_size,
                                                 ov::Strides stride = {1, 1},
                                                 ov::CoordinateDiff pad = {0, 0},
                                                 ov::Strides dilations = {1, 1})     {
        return deconvolution(id,
                             input,
                             weights,
                             stride,
                             pad,
                             dilations,
                             output_size);
    }

    /// @brief Defines logical pad value added to input tensor.
    ov::CoordinateDiff pad;
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    ov::Strides stride;
    /// @brief Defines the distance in width and height between elements in the filter.
    ov::Strides dilations;
    /// @brief Indicates that the primitive has user-defined output size (non-zero value).
    bool with_output_size = true;
    /// @brief User-defined output data size of the primitive (w/o padding).
    tensor output_size;
    /// @brief Number of feature groups (grouped convolution). If more than 1 then weights/bias count needs to be 1.
    uint32_t groups = 1;
    /// @brief Defines a padding added to input image on left (x axis) and top (y axis).
    ov::CoordinateDiff pads_begin;
    /// @brief Defines a padding added to input image on right (x axis) and bottom (y axis).
    ov::CoordinateDiff pads_end;
    /// @brief Defines additional amount of paddings per each spatial axis added to output tensor.
    ov::CoordinateDiff out_padding;
    /// @param grouped_weights_shape Defines if weights tensor has explicit group dimension.
    bool grouped_weights_shape = false;
    /// @brief Defines spatial shape of the output.
    ov::PartialShape output_partial_shape;
    /// @brief Data primitive id containing spatial shape of the output.
    primitive_id output_shape_id;
    /// @brief List of primitive ids containing weights data.
    const primitive_id_arr weights;
    /// @brief List of primitive ids containing bias data.
    const primitive_id_arr bias;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, pad.begin(), pad.end());
        seed = hash_range(seed, stride.begin(), stride.end());
        seed = hash_combine(seed, groups);
        seed = hash_combine(seed, grouped_weights_shape);
        seed = hash_combine(seed, weights.size());
        seed = hash_combine(seed, bias.size());
        seed = hash_combine(seed, output_shape_id.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const deconvolution>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(pad) &&
               cmp_fields(stride) &&
               cmp_fields(dilations) &&
               cmp_fields(groups) &&
               cmp_fields(pads_begin) &&
               cmp_fields(pads_end) &&
               cmp_fields(out_padding) &&
               cmp_fields(grouped_weights_shape) &&
               cmp_fields(weights.size()) &&
               cmp_fields(bias.size()) &&
               cmp_fields(output_shape_id.empty());
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<deconvolution>::save(ob);
        ob << pad;
        ob << stride;
        ob << dilations;
        ob << with_output_size;
        ob << output_size;
        ob << groups;
        ob << pads_begin;
        ob << pads_end;
        ob << out_padding;
        ob << grouped_weights_shape;
        ob << output_partial_shape;
        ob << output_shape_id;
        ob << weights;
        ob << bias;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<deconvolution>::load(ib);
        ib >> pad;
        ib >> stride;
        ib >> dilations;
        ib >> with_output_size;
        ib >> output_size;
        ib >> groups;
        ib >> pads_begin;
        ib >> pads_end;
        ib >> out_padding;
        ib >> grouped_weights_shape;
        ib >> output_partial_shape;
        ib >> output_shape_id;
        ib >> *const_cast<primitive_id_arr*>(&weights);
        ib >> *const_cast<primitive_id_arr*>(&bias);
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret;
        ret.reserve(weights.size() + bias.size() + (output_shape_id.empty() ? 0 : 1));
        for (auto& w : weights) ret.push_back(w);
        for (auto& b : bias) ret.push_back(b);
        if (!output_shape_id.empty()) ret.push_back(output_shape_id);

        return ret;
    }
};
}  // namespace cldnn
