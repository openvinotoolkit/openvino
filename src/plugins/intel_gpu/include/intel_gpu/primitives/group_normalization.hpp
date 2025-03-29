// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Performs the following transformation of the input tensor:
/// y = scale * (x - mean) / sqrt(variance + epsilon) + bias
/// The operation is applied per batch, per group of channels.
struct group_normalization : public primitive_base<group_normalization> {
    CLDNN_DECLARE_PRIMITIVE(group_normalization)

    group_normalization() : primitive_base("", {}) {}

    /// @brief Constructs group_normalization primitive.
    /// @param id This primitive id.
    /// @param data The input tensor to be normalized.
    /// @param scale Scale values tensor.
    /// @param bias Bias values.
    /// @param num_groups Number of groups the channel dimension will be divided into.
    /// @param epsilon A value added to the variance which ensures that division by zero.
    /// does not occur for any normalized element.
    group_normalization(const primitive_id& id,
                        const input_info& data,
                        const input_info& scale,
                        const input_info& bias,
                        std::int64_t num_groups,
                        double epsilon)
    : primitive_base(id, {data, scale, bias}), num_groups{num_groups}, epsilon{epsilon} {}

    /// @brief Number of groups the channel dimension will be divided into
    /// @details
    /// Specifies the number of groups G that the channel dimension will be divided into.
    std::int64_t num_groups{};

    /// @brief A value added to the variance which ensures that division by zero.
    /// @details
    /// A very small value added to the variance for numerical stability.
    /// Ensures that division by zero does not occur for any normalized element.
    double epsilon{};

    std::size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_groups);
        return hash_combine(seed, epsilon);
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        const auto& rhs_casted = downcast<const group_normalization>(rhs);

        return num_groups == rhs_casted.num_groups && epsilon == rhs_casted.epsilon;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<group_normalization>::save(ob);
        ob << num_groups;
        ob << epsilon;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<group_normalization>::load(ib);
        ib >> num_groups;
        ib >> epsilon;
    }
};

} // namespace cldnn
