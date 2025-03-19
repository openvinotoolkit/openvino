// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Mean Variance Normalization primitive.
/// @details Normalizes the input to have 0-mean and/or unit (1) variance.
struct mvn : public primitive_base<mvn> {
    CLDNN_DECLARE_PRIMITIVE(mvn)

    mvn() : primitive_base("", {}) {}

    /// @brief Constructs mvn primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param reduction_axes Determines axes set for normalization.
    /// @param normalize_variance Determines if normalize variance is applied. Default is true.
    /// @param epsilon Epsilon for not dividing by zero while normalizing.
    /// @param eps_inside_sqrt The mode of applying epsilon.
    mvn(const primitive_id& id,
        const input_info& input,
        const bool normalize_variance,
        const float epsilon,
        const bool eps_inside_sqrt,
        const std::vector<int64_t>& reduction_axes)
        : primitive_base(id, {input}),
          normalize_variance(normalize_variance),
          epsilon(epsilon),
          eps_inside_sqrt(eps_inside_sqrt),
          reduction_axes(reduction_axes) {}

    /// @brief Determines if normalize variance is applied.
    bool normalize_variance;
    /// @brief Epsilon for not dividing by zero while normalizing.
    float epsilon;
    /// @brief The mode of applying epsilon.
    bool eps_inside_sqrt = false;
    /// @brief Determines axes set for normalization.
    std::vector<int64_t> reduction_axes;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, normalize_variance);
        seed = hash_combine(seed, epsilon);
        seed = hash_combine(seed, eps_inside_sqrt);
        seed = hash_range(seed, reduction_axes.begin(), reduction_axes.end());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const mvn>(rhs);

        return normalize_variance == rhs_casted.normalize_variance &&
               epsilon == rhs_casted.epsilon &&
               eps_inside_sqrt == rhs_casted.eps_inside_sqrt &&
               reduction_axes == rhs_casted.reduction_axes;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<mvn>::save(ob);
        ob << normalize_variance;
        ob << epsilon;
        ob << eps_inside_sqrt;
        ob << reduction_axes;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<mvn>::load(ib);
        ib >> normalize_variance;
        ib >> epsilon;
        ib >> eps_inside_sqrt;
        ib >> reduction_axes;
    }

    bool across_channels() const {
        int64_t channel_axis = 1;
        if (std::find(reduction_axes.begin(), reduction_axes.end(), channel_axis) != reduction_axes.end()) {
            return true;
        } else {
            return false;
        }
    }

    bool requires_alignment(const ov::PartialShape& shape) const {
        auto rank = static_cast<int64_t>(shape.size());
        auto axes = reduction_axes;
        std::for_each(axes.begin(), axes.end(), [rank](int64_t& v) { v = (v < 0) ? v + rank : v; });

        // If all axes from 2 to rank-1 is a part of reduction scope,
        // then it's mapped to the old MVN case and don't require alignment
        for (int64_t i = 2; i < rank; i++) {
            if (std::find_if(axes.begin(), axes.end(), [i, &shape](const int64_t& v){ return v == i || shape[i].get_max_length() == 1; }) == axes.end())
                return true;
        }

        return false;
    }
};
}  // namespace cldnn
