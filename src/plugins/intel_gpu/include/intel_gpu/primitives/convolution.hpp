// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include <vector>

namespace cldnn {

struct convolution : public primitive_base<convolution> {
    CLDNN_DECLARE_PRIMITIVE(convolution)

    convolution() : primitive_base("", {}) {}

    /// @brief Constructs convolution primitive
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights Primitive id containing weights data.
    /// @param bias Primitive id containing bias data.
    /// @param w_zero_point Primitive id containing weights zero points.
    /// @param a_zero_point Primitive id containing activations zero points.
    /// @param compensation Primitive id containing activations precalculated compensations for optimized asymmetric quantization.
    /// It works as bias, but can be skipped by the kernel if it performs direct zero-points subtraction
    /// @param groups Number of filter groups.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param padding_begin Defines a padding added to input image on left (x axis) and top (y axis).
    /// @param padding_end Defines a padding added to input image on right (x axis) and bottom (y axis).
    /// @param grouped_weights_shape True if weights shape is [G, O, I, ...], and false if it's [O, I, ...] or [G*O, I, ...]
    /// @param audo_pad The pad type for automatically computing padding sizes
    convolution(const primitive_id& id,
                const input_info& input,
                const primitive_id& weights,
                const primitive_id& bias,
                const primitive_id& w_zero_point,
                const primitive_id& a_zero_point,
                const primitive_id& compensation,
                uint32_t groups,
                ov::Strides stride,
                ov::Strides dilation,
                ov::CoordinateDiff padding_begin,
                ov::CoordinateDiff padding_end,
                bool grouped_weights_shape,
                data_types output_data_type,
                const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT)
            : primitive_base(id, {input}, 1, {optional_data_type{output_data_type}}),
              groups(groups),
              stride(stride),
              dilation(dilation),
              padding_begin(padding_begin),
              padding_end(padding_end),
              auto_pad(auto_pad),
              grouped_weights_shape(grouped_weights_shape),
              weights(weights),
              bias(bias),
              weights_zero_points(w_zero_point),
              activations_zero_points(a_zero_point),
              compensation(compensation) {
    }

    /// @brief Constructs convolution primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights Primitive id containing weights data.
    /// @param bias Primitive id containing bias data.
    /// @param groups Number of filter groups.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param padding_begin Defines a padding added to input image on left (x axis) and top (y axis).
    /// @param padding_end Defines a padding added to input image on right (x axis) and bottom (y axis).
    /// @param grouped_weights_shape True if weights shape is [G, O, I, ...], and false if it's [O, I, ...] or [G*O, I, ...]
    /// @param audo_pad The pad type for automatically computing padding sizes
    convolution(const primitive_id& id,
                const input_info& input,
                const primitive_id& weights,
                const primitive_id& bias,
                uint32_t groups,
                ov::Strides stride,
                ov::Strides dilation,
                ov::CoordinateDiff padding_begin,
                ov::CoordinateDiff padding_end,
                bool grouped_weights_shape,
                const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT)
        : primitive_base(id, {input}),
          groups(groups),
          stride(stride),
          dilation(dilation),
          padding_begin(padding_begin),
          padding_end(padding_end),
          auto_pad(auto_pad),
          grouped_weights_shape(grouped_weights_shape),
          weights(weights),
          bias(bias),
          weights_zero_points(""),
          activations_zero_points(""),
          compensation("") {
    }

    /// @brief Constructs convolution primitive.
    /// @param id This primitive id.
    /// @param inputs Array of input primitive ids.
    /// @param weights Primitive id containing weights data.
    /// @param bias Primitive id containing bias data.
    /// @param deformable_mode Defines a mode for convolution
    /// @param groups Number of filter groups.
    /// @param deformable_groups Defines a number of deformable groups that splits trans input into several parts
    /// by channel dimension.
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param dilation Defines gaps in the input - dilation rate k=1 is normal convolution,
    /// k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following:
    /// w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    /// @param padding_begin Defines a padding added to input image on left (x axis) and top (y axis).
    /// @param padding_end Defines a padding added to input image on right (x axis) and bottom (y axis).
    /// @param bilinear_interpolation_pad If bilinear_interpolation_pad is true and the sampling location is within
    /// one pixel outside of the feature map boundary, then bilinear interpolation is performed on the zero padded feature map.
    convolution(const primitive_id& id,
                const std::vector<input_info>& inputs,
                const primitive_id& weights,
                const primitive_id& bias,
                bool deformable_mode,
                uint32_t groups,
                uint32_t deformable_groups,
                ov::Strides stride,
                ov::Strides dilation,
                ov::CoordinateDiff padding_begin,
                ov::CoordinateDiff padding_end,
                bool bilinear_interpolation_pad = false)
    : primitive_base(id, inputs),
      groups(groups),
      stride(stride),
      dilation(dilation),
      padding_begin(padding_begin),
      padding_end(padding_end),
      auto_pad(ov::op::PadType::EXPLICIT),
      deformable_mode(deformable_mode),
      deformable_groups(deformable_groups),
      bilinear_interpolation_pad(bilinear_interpolation_pad),
      grouped_weights_shape(false),
      weights(weights),
      bias(bias),
      weights_zero_points(""),
      activations_zero_points(""),
      compensation("") {
    }

    /// @brief Number of feature groups (grouped convolution). If more than 1 then weights/bias count needs to be 1.
    uint32_t groups {1};
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    ov::Strides stride;
    /// @brief Defines gaps in the input - dilation rate k=1 is normal convolution, k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    ov::Strides dilation;
    /// @param padding_begin Defines a padding added to input image on left (x axis) and top (y axis).
    ov::CoordinateDiff padding_begin;
    /// @param padding_end Defines a padding added to input image on right (x axis) and bottom (y axis).
    ov::CoordinateDiff padding_end;
    /// @param audo_pad The pad type for automatically computing padding sizes
    ov::op::PadType auto_pad = ov::op::PadType::NOTSET;

    /// @param deformable_mode.
    bool deformable_mode {false};
    /// @param deformable_groups Defines a number of deformable groups that splits trans input into several parts
    /// by channel dimension.
    uint32_t deformable_groups {1};
    /// if bilinear_interpolation_pad is true and the sampling location is within one pixel outside of the feature map boundary,
    /// then bilinear interpolation is performed on the zero padded feature map.
    /// If bilinear_interpolation_pad is false and the sampling location is within one pixel outside of the feature map boundary,
    /// then the sampling location shifts to the inner boundary of the feature map.
    bool bilinear_interpolation_pad {false};

    bool transposed {false};

    /// @param grouped_weights_shape Defines if weights tensor has explicit group dimension.
    bool grouped_weights_shape {false};
    /// @brief Primitive id containing weights data.
    const primitive_id weights;
    /// @brief Primitive id containing bias data.
    const primitive_id bias;
    /// @brief Primitive id containing weights zero points.
    const primitive_id weights_zero_points;
    /// @brief Primitive id containing activations zero points.
    const primitive_id activations_zero_points;
    /// @brief Primitive id containing compensation.
    const primitive_id compensation;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, padding_end.begin(), padding_end.end());
        seed = hash_range(seed, padding_begin.begin(), padding_begin.end());
        seed = hash_range(seed, stride.begin(), stride.end());
        seed = hash_range(seed, dilation.begin(), dilation.end());
        seed = hash_combine(seed, auto_pad);
        seed = hash_combine(seed, groups);
        seed = hash_combine(seed, deformable_groups);
        seed = hash_combine(seed, deformable_mode);
        seed = hash_combine(seed, bilinear_interpolation_pad);
        seed = hash_combine(seed, transposed);
        seed = hash_combine(seed, grouped_weights_shape);
        seed = hash_combine(seed, !weights.empty());
        seed = hash_combine(seed, !bias.empty());
        seed = hash_combine(seed, !weights_zero_points.empty());
        seed = hash_combine(seed, !activations_zero_points.empty());
        seed = hash_combine(seed, !compensation.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const convolution>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(stride) &&
               cmp_fields(dilation) &&
               cmp_fields(groups) &&
               cmp_fields(deformable_groups) &&
               cmp_fields(padding_begin) &&
               cmp_fields(padding_end) &&
               cmp_fields(auto_pad) &&
               cmp_fields(deformable_mode) &&
               cmp_fields(bilinear_interpolation_pad) &&
               cmp_fields(transposed) &&
               cmp_fields(grouped_weights_shape) &&
               cmp_fields(weights.empty()) &&
               cmp_fields(bias.empty()) &&
               cmp_fields(weights_zero_points.empty()) &&
               cmp_fields(activations_zero_points.empty()) &&
               cmp_fields(compensation.empty());
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<convolution>::save(ob);
        ob << groups;
        ob << stride;
        ob << dilation;
        ob << padding_begin;
        ob << padding_end;
        ob << make_data(&auto_pad, sizeof(ov::op::PadType));
        ob << deformable_mode;
        ob << deformable_groups;
        ob << bilinear_interpolation_pad;
        ob << transposed;
        ob << grouped_weights_shape;
        ob << weights;
        ob << bias;
        ob << weights_zero_points;
        ob << activations_zero_points;
        ob << compensation;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<convolution>::load(ib);
        ib >> groups;
        ib >> stride;
        ib >> dilation;
        ib >> padding_begin;
        ib >> padding_end;
        ib >> make_data(&auto_pad, sizeof(ov::op::PadType));
        ib >> deformable_mode;
        ib >> deformable_groups;
        ib >> bilinear_interpolation_pad;
        ib >> transposed;
        ib >> grouped_weights_shape;
        ib >> *const_cast<primitive_id*>(&weights);
        ib >> *const_cast<primitive_id*>(&bias);
        ib >> *const_cast<primitive_id*>(&weights_zero_points);
        ib >> *const_cast<primitive_id*>(&activations_zero_points);
        ib >> *const_cast<primitive_id*>(&compensation);
    }

    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret = {weights};
        if (!bias.empty()) {
            ret.push_back(bias);
        }
        if (!weights_zero_points.empty()) {
            ret.push_back(weights_zero_points);
        }
        if (!activations_zero_points.empty()) {
            ret.push_back(activations_zero_points);
        }
        if (!compensation.empty()) {
            ret.push_back(compensation);
        }

        return ret;
    }
};

}  // namespace cldnn
