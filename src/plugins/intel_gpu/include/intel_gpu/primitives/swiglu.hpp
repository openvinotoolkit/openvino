// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "ov_ops/glu.hpp"
#include "primitive.hpp"

namespace cldnn {

/// @brief Swish Gated Linear Unit Activation primitive
/// @details Performs gated linear unit activation that combines swish or gelu activation function
struct swiglu : public primitive_base<swiglu> {
    CLDNN_DECLARE_PRIMITIVE(swiglu);

    swiglu() : primitive_base("", {}) {}

    /// @brief Constructs swiglu primitive
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param axis The index of an axis in data along which to perform the split
    /// @param split_lengths A list containing the sizes of each output tensor along the split axis
    /// @param output_size Output data size of the primitive
    swiglu(const primitive_id& id,
           const input_info& input,
           const int64_t& axis,
           const int64_t& split_lengths,
           const ov::op::internal::GLU::GluType glu_type,
           const size_t split_to_glu_idx,
           const tensor output_size)
           : primitive_base(id, {input}),
             axis(axis),
             split_lengths(split_lengths),
             glu_type(glu_type),
             split_to_glu_idx(split_to_glu_idx),
             output_size(output_size) {}

    int64_t axis = 0;
    int64_t split_lengths = 0;
    ov::op::internal::GLU::GluType glu_type = ov::op::internal::GLU::GluType::Swish;
    size_t split_to_glu_idx = 0;
    tensor output_size;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, axis);
        seed = hash_combine(seed, split_lengths);
        seed = hash_combine(seed, glu_type);
        seed = hash_combine(seed, split_to_glu_idx);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const swiglu>(rhs);
        return axis == rhs_casted.axis && split_lengths == rhs_casted.split_lengths &&
               glu_type == rhs_casted.glu_type && split_to_glu_idx == rhs_casted.split_to_glu_idx;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<swiglu>::save(ob);
        ob << axis;
        ob << split_lengths;
        ob << output_size;
        ob << make_data(&glu_type, sizeof(glu_type));
        ob << split_to_glu_idx;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<swiglu>::load(ib);
        ib >> axis;
        ib >> split_lengths;
        ib >> output_size;
        ib >> make_data(&glu_type, sizeof(glu_type));
        ib >> split_to_glu_idx;
    }
};
}  // namespace cldnn
