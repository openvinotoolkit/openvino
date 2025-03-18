// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

#include "openvino/core/shape.hpp"

namespace cldnn {

/// @brief
/// @details
struct gather : public primitive_base<gather> {
    CLDNN_DECLARE_PRIMITIVE(gather)

    gather() : primitive_base("", {}) {}

    /// @brief Constructs gather primitive.
    /// @param id This primitive id.
    /// @param dict Input dictionary primitive id.
    /// @param idx Input indexes primitive id.
    /// @param axis Gathering axis.
    /// @param input_rank Input rank.
    /// @param output_shape Output shape.
    /// @param batch_dim Batch_dim
    /// @param support_neg_ind Support negative indexes
    gather(const primitive_id& id,
           const input_info& dict,
           const input_info& idx,
           const int64_t axis,
           const int64_t input_rank,
           const ov::Shape& output_shape,
           const int64_t batch_dim = 0,
           const bool support_neg_ind = false)
        : primitive_base(id, {dict, idx})
        , axis(axis)
        , input_rank(input_rank)
        , output_shape(output_shape)
        , batch_dim(batch_dim)
        , support_neg_ind(support_neg_ind) {}

    /// @brief Constructs gather compressed primitive.
    /// @param id This primitive id.
    /// @param dict Input dictionary primitive id.
    /// @param idx Input indexes primitive id.
    /// @param axis Gathering axis.
    /// @param decompression_scale Input decompression scale factors primitive id.
    /// @param decompression_zero_point Input decompression zero point primitive id.
    /// @param input_rank Input rank.
    /// @param output_shape Output shape.
    /// @param batch_dim Batch_dim
    /// @param support_neg_ind Support negative indexes
    gather(const primitive_id& id,
           const input_info& dict,
           const input_info& idx,
           const int64_t axis,
           const input_info& decompression_scale,
           const input_info& decompression_zero_point,
           const ov::element::Type decompressed_type,
           const int64_t input_rank,
           const ov::Shape& output_shape,
           const int64_t batch_dim = 0,
           const bool support_neg_ind = false)
        : primitive_base(id, {dict, idx})
        , axis(axis)
        , input_rank(input_rank)
        , output_shape(output_shape)
        , batch_dim(batch_dim)
        , support_neg_ind(support_neg_ind)
        , compressed_weights(true)
        , decompressed_type(decompressed_type)
        , decompression_scale(decompression_scale)
        , decompression_zero_point(decompression_zero_point) {
            OPENVINO_ASSERT(decompression_scale.is_valid(), "[GPU] Compressed gather requires at least decompression scale input");
        }

    /// @brief Gathering axis
    int64_t axis = 0;
    /// @brief Gather input rank
    int64_t input_rank;
    /// @brief Gather output shape
    ov::Shape output_shape;
    /// @brief Gathering batch_dim
    int64_t batch_dim = 0;
    /// @brief Support negative indexes
    bool support_neg_ind = false;

    bool compressed_weights = false;
    ov::element::Type decompressed_type;
    input_info decompression_scale;
    input_info decompression_zero_point;
    std::optional<float> decompression_zero_point_scalar = std::optional<float>();

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, axis);
        seed = hash_combine(seed, batch_dim);
        seed = hash_combine(seed, support_neg_ind);
        seed = hash_combine(seed, compressed_weights);
        seed = hash_combine(seed, decompressed_type.get_type_name());
        seed = hash_combine(seed, decompression_scale.is_valid());
        seed = hash_combine(seed, decompression_zero_point.is_valid());
        seed = hash_combine(seed, decompression_zero_point_scalar.has_value());
        seed = hash_combine(seed, decompression_zero_point_scalar.value_or(0.0f));
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const gather>(rhs);

        return axis == rhs_casted.axis &&
               batch_dim == rhs_casted.batch_dim &&
               support_neg_ind == rhs_casted.support_neg_ind &&
               compressed_weights == rhs_casted.compressed_weights &&
               decompression_scale.is_valid() == rhs_casted.decompression_scale.is_valid() &&
               decompression_zero_point.is_valid() == rhs_casted.decompression_zero_point.is_valid() &&
               decompression_zero_point_scalar.value_or(0.0f) == rhs_casted.decompression_zero_point_scalar.value_or(0.0f);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<gather>::save(ob);
        ob << axis;
        ob << input_rank;
        ob << output_shape;
        ob << batch_dim;
        ob << support_neg_ind;
        ob << compressed_weights;
        ob << decompressed_type.get_type_name();
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
        primitive_base<gather>::load(ib);
        ib >> axis;
        ib >> input_rank;
        ib >> output_shape;
        ib >> batch_dim;
        ib >> support_neg_ind;
        ib >> compressed_weights;
        std::string decompressed_type_name;
        ib >> decompressed_type_name;
        decompressed_type = ov::element::Type(decompressed_type_name);
        ib >> decompression_scale;
        ib >> decompression_zero_point;

        bool has_value;
        ib >> has_value;
        if (has_value) {
            float decompression_zero_point_value = 0.f;
            ib >> make_data(&decompression_zero_point_value, sizeof(float));
            decompression_zero_point_scalar = decompression_zero_point_value;
        } else {
            decompression_zero_point_scalar = std::optional<float>();
        }
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret;

        if (decompression_scale.is_valid())
            ret.push_back(decompression_scale.pid);

        if (decompression_zero_point.is_valid())
            ret.push_back(decompression_zero_point.pid);

        return ret;
    }
};
}  // namespace cldnn
