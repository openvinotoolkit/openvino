// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <optional>

#include "primitive.hpp"

namespace cldnn {

/// @brief Grouped matmul for MoE forward pass (2D x 3D case).
/// @details
/// Computes multiple matrix multiplications where each group processes a subset
/// of the input rows. Matches ov::op::v17::GroupedMatMul 2D x 3D semantics.
/// 3D x 3D case is handled by vanilla matmul, instead of grouped matmul
/// because the behaviors are identical.
///
///   mat_a   : [total_tokens, K]        - rows partitioned by @p offsets
///   mat_b   : [G, N, K]                - per-group weights, stored transposed (B^T)
///   offsets : [G] (s32)                - cumulative end-offsets partitioning mat_a rows
///   output  : [total_tokens, N]
///
/// The primitive also supports a compressed weights variant, in which mat_b is a
/// low-precision (u4/i4/u8/i8) constant and additional inputs carry the per-group
/// dequantization scale (and optional zero-point). At runtime, the oneDNN impl
/// consumes the compressed weight directly via primitive attribute scales / zero-points.
struct grouped_matmul : public primitive_base<grouped_matmul> {
    CLDNN_DECLARE_PRIMITIVE(grouped_matmul)

    enum GroupedMatmulInputIdx {
        INPUT = 0,     // mat_a  : [total_tokens, K]
        WEIGHT = 1,    // mat_b  : [G, N, K]
        OFFSETS = 2,   // offsets: [G] cumulative end-offsets (s32)
    };

    grouped_matmul() : primitive_base("", {}) {}

    /// @brief Constructs a non-compressed grouped matmul.
    grouped_matmul(const primitive_id& id,
                   const std::vector<input_info>& inputs,
                   const data_types output_data_type)
        : primitive_base(id, inputs, 1, {optional_data_type{output_data_type}}) {}

    /// @brief Constructs a compressed-weights grouped matmul.
    /// @param inputs Base inputs [mat_a, mat_b, offsets]. Scale/zp are attached
    ///        separately via @p decompression_scale / @p decompression_zero_point
    ///        so that shape-inference dependencies remain limited to mat_a/mat_b.
    grouped_matmul(const primitive_id& id,
                   const std::vector<input_info>& inputs,
                   const input_info& decompression_scale,
                   const input_info& decompression_zero_point,
                   const data_types output_data_type)
        : primitive_base(id, inputs, 1, {optional_data_type{output_data_type}}),
          compressed_weights(true),
          decompression_scale(decompression_scale),
          decompression_zero_point(decompression_zero_point) {
        OPENVINO_ASSERT(decompression_scale.is_valid(),
                        "[GPU] Compressed grouped_matmul requires a decompression scale input");
    }

    bool compressed_weights = false;
    input_info decompression_scale = {};
    input_info decompression_zero_point = {};
    std::optional<float> decompression_zero_point_scalar = std::optional<float>();

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, compressed_weights);
        seed = hash_combine(seed, decompression_scale.is_valid());
        seed = hash_combine(seed, decompression_zero_point.is_valid());
        seed = hash_combine(seed, decompression_zero_point_scalar.has_value());
        seed = hash_combine(seed, decompression_zero_point_scalar.value_or(0.0f));
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const grouped_matmul>(rhs);
        return compressed_weights == rhs_casted.compressed_weights &&
               decompression_scale.is_valid() == rhs_casted.decompression_scale.is_valid() &&
               decompression_zero_point.is_valid() == rhs_casted.decompression_zero_point.is_valid() &&
               decompression_zero_point_scalar.value_or(0.0f) ==
                   rhs_casted.decompression_zero_point_scalar.value_or(0.0f);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<grouped_matmul>::save(ob);
        ob << compressed_weights;
        ob << decompression_scale;
        ob << decompression_zero_point;
        if (decompression_zero_point_scalar.has_value()) {
            ob << true;
            ob << make_data(&decompression_zero_point_scalar.value(), sizeof(float));
        } else {
            ob << false;
        }
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<grouped_matmul>::load(ib);
        ib >> compressed_weights;
        ib >> decompression_scale;
        ib >> decompression_zero_point;
        bool has_value = false;
        ib >> has_value;
        if (has_value) {
            float v = 0.f;
            ib >> make_data(&v, sizeof(float));
            decompression_zero_point_scalar = v;
        } else {
            decompression_zero_point_scalar = std::optional<float>();
        }
    }

protected:
    std::map<size_t, const input_info*> get_dependencies_map() const override {
        auto ret = std::map<size_t, const input_info*>{};
        auto idx = input.size();
        if (decompression_scale.is_valid())
            ret[idx++] = &decompression_scale;
        if (decompression_zero_point.is_valid())
            ret[idx++] = &decompression_zero_point;
        return ret;
    }
};
}  // namespace cldnn
