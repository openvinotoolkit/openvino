// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/serialization/utils.hpp"

namespace cldnn {

/// @brief SegmentMax computes the maximum values along segments of a tensor.
/// @details
///   Given a data tensor and a 1-D segment_ids tensor, produces an output tensor
///   where each segment contains the elementwise maximum of the corresponding
///   data rows.
struct segment_max : public primitive_base<segment_max> {
    CLDNN_DECLARE_PRIMITIVE(segment_max)

    segment_max() : primitive_base("", {}) {}

    /// @brief Constructs segment_max primitive.
    /// @param id This primitive id.
    /// @param data Data input.
    /// @param segment_ids Segment indices input (1-D, non-decreasing).
    /// @param fill_mode 0 = ZERO, 1 = LOWEST. Value for empty segments.
    segment_max(const primitive_id& id,
                const input_info& data,
                const input_info& segment_ids,
                int fill_mode)
        : primitive_base(id, {data, segment_ids}),
          fill_mode(fill_mode) {}

    /// @brief Fill mode for empty segments. 0 = ZERO (default), 1 = LOWEST.
    int fill_mode = 0;

    /// @brief Stored segment_ids values for compile-time shape inference (when constant).
    std::vector<int64_t> segment_ids_data;

    /// @brief Stored num_segments value for compile-time shape inference (when constant). -1 = not set.
    int64_t num_segments_val = -1;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, fill_mode);
        seed = hash_range(seed, segment_ids_data.begin(), segment_ids_data.end());
        seed = hash_combine(seed, num_segments_val);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const segment_max>(rhs);

        return fill_mode == rhs_casted.fill_mode &&
               segment_ids_data == rhs_casted.segment_ids_data &&
               num_segments_val == rhs_casted.num_segments_val;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<segment_max>::save(ob);
        ob << fill_mode;
        ob << segment_ids_data;
        ob << num_segments_val;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<segment_max>::load(ib);
        ib >> fill_mode;
        ib >> segment_ids_data;
        ib >> num_segments_val;
    }
};
}  // namespace cldnn
