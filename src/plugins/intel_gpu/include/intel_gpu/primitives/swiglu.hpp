// Copyright (C) 2018-2026 Intel Corporation
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
    /// @param glu_stride A list containing the sizes of each output tensor along the split axis
    /// @param output_size Output data size of the primitive
    swiglu(const primitive_id& id,
           const input_info& input,
           const int64_t& axis,
           const int64_t& glu_stride,
           const ov::op::internal::GLU::GluType glu_type,
           const size_t gate_idx,
           const float clamp_min,
           const float clamp_max,
           const float swish_beta,
           const float up_add_val,
           const tensor output_size)
           : primitive_base(id, {input}),
             axis(axis),
             glu_stride(glu_stride),
             glu_type(glu_type),
             gate_idx(gate_idx),
             clamp_min(clamp_min),
             clamp_max(clamp_max),
             swish_beta(swish_beta),
             up_add_val(up_add_val),
             output_size(output_size) {}

    swiglu(const primitive_id& id,
           const input_info& input,
           const int64_t& axis,
           const int64_t& glu_stride,
           const ov::op::internal::GLU::GluType glu_type,
           const size_t gate_idx,
           const tensor output_size)
        : swiglu(id,
                 input,
                 axis,
                 glu_stride,
                 glu_type,
                 gate_idx,
                 std::numeric_limits<float>::lowest(),
                 std::numeric_limits<float>::max(),
                 1.0f,
                 0.0f,
                 output_size) {}

    int64_t axis = 0;
    int64_t glu_stride = 0;
    ov::op::internal::GLU::GluType glu_type = ov::op::internal::GLU::GluType::Swish;
    size_t gate_idx = 0;
    float clamp_min = std::numeric_limits<float>::lowest();
    float clamp_max = std::numeric_limits<float>::max();;
    float swish_beta = 1.0f;
    float up_add_val = 0.0f;
    tensor output_size;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, axis);
        seed = hash_combine(seed, glu_stride);
        seed = hash_combine(seed, glu_type);
        seed = hash_combine(seed, gate_idx);
        seed = hash_combine(seed, clamp_min);
        seed = hash_combine(seed, clamp_max);
        seed = hash_combine(seed, swish_beta);
        seed = hash_combine(seed, up_add_val);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const swiglu>(rhs);
        return axis == rhs_casted.axis && glu_stride == rhs_casted.glu_stride &&
               glu_type == rhs_casted.glu_type && gate_idx == rhs_casted.gate_idx &&
               clamp_min == rhs_casted.clamp_min && clamp_max == rhs_casted.clamp_max &&
               swish_beta == rhs_casted.swish_beta && up_add_val == rhs_casted.up_add_val;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<swiglu>::save(ob);
        ob << axis;
        ob << glu_stride;
        ob << output_size;
        ob << make_data(&glu_type, sizeof(glu_type));
        ob << gate_idx;
        ob << clamp_min;
        ob << clamp_max;
        ob << swish_beta;
        ob << up_add_val;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<swiglu>::load(ib);
        ib >> axis;
        ib >> glu_stride;
        ib >> output_size;
        ib >> make_data(&glu_type, sizeof(glu_type));
        ib >> gate_idx;
        ib >> clamp_min;
        ib >> clamp_max;
        ib >> swish_beta;
        ib >> up_add_val;
    }
};
}  // namespace cldnn
