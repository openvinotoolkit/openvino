// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"
#include <vector>

namespace cldnn {

/// @brief Performs forward spatial binary_convolution with weight sharing.
struct binary_convolution : public primitive_base<binary_convolution> {
    CLDNN_DECLARE_PRIMITIVE(binary_convolution)

    binary_convolution() : primitive_base("", {}) {}

    /// @brief Constructs binary_convolution primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param pad Defines logical pad value added to input tensor
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
                       const input_info& input,
                       const std::vector<primitive_id>& weights,
                       ov::Strides stride = {1, 1},
                       ov::CoordinateDiff pad = {0, 0},
                       ov::Strides dilation = {1, 1},
                       tensor output_size = {0, 0, 0, 0},
                       int groups = 1,
                       float pad_value = 0.0f,
                       data_types calc_precision = data_types::f32,
                       const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}, {optional_data_type {calc_precision}}),
          pad(pad),
          stride(stride),
          dilation(dilation),
          output_size(output_size),
          groups(groups),
          pad_value(pad_value),
          weights(weights) {}

    /// @brief Defines logical pad value added to input tensor
    ov::CoordinateDiff pad;
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    ov::Strides stride;
    /// @brief Defines gaps in the input - dilation rate k=1 is normal binary_convolution, k=2 means skipping one pixel per input, k=4 means skipping 3 pixels.
    /// As an example in one dimension, a filter w of size 3 would compute over input x the following: w[0]*x[0] + w[1]*x[1] + w[2]*x[2] for dilation of 1.
    /// For dilation 2 the filter would instead compute w[0]*x[0] + w[1]*x[2] + w[2]*x[4].
    ov::Strides dilation;
    /// @brief User-defined output data size of the primitive (w/o padding).
    tensor output_size;
    /// @brief Number of feature groups (grouped convolution). If more than 1 then weights/bias count needs to be 1.
    int groups = 1;
    /// @brief Logical value of padding. Can be one of 3 values: 1 - pad bits equal to 1; -1 -> pad bits equal to 0; 0 -> pad is not counted
    float pad_value = 0.0f;
    /// @brief List of primitive ids containing weights data.
    const primitive_id_arr weights;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, pad.begin(), pad.end());
        seed = hash_range(seed, stride.begin(), stride.end());
        seed = hash_range(seed, dilation.begin(), dilation.end());
        seed = hash_combine(seed, groups);
        seed = hash_combine(seed, pad_value);
        seed = hash_combine(seed, weights.size());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const binary_convolution>(rhs);

        return pad == rhs_casted.pad &&
               stride == rhs_casted.stride &&
               dilation == rhs_casted.dilation &&
               groups == rhs_casted.groups &&
               pad_value == rhs_casted.pad_value &&
               weights.size() == rhs_casted.weights.size();
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<binary_convolution>::save(ob);
        ob << pad;
        ob << stride;
        ob << dilation;
        ob << output_size;
        ob << groups;
        ob << pad_value;
        ob << weights;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<binary_convolution>::load(ib);
        ib >> pad;
        ib >> stride;
        ib >> dilation;
        ib >> output_size;
        ib >> groups;
        ib >> pad_value;
        ib >> *const_cast<primitive_id_arr*>(&weights);
    }

    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.reserve(weights.size());
        for (auto& w : weights) ret.push_back(std::ref(w));
        return ret;
    }
};
}  // namespace cldnn
