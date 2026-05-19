// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/serialization/utils.hpp"
#include "openvino/op/util/attr_types.hpp"

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
    /// @param fill_mode Fill mode for empty segments.
    segment_max(const primitive_id& id,
                const input_info& data,
                const input_info& segment_ids,
                ov::op::FillMode fill_mode)
        : primitive_base(id, {data, segment_ids}),
          fill_mode(fill_mode) {}

    /// @brief Fill mode for empty segments.
    ov::op::FillMode fill_mode = ov::op::FillMode::ZERO;

    /// @brief Max segment_id value for compile-time shape inference (when constant). -1 = not set.
    int64_t max_segment_id = -1;

    /// @brief Stored num_segments value for compile-time shape inference (when constant). -1 = not set.
    int64_t num_segments_val = -1;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, static_cast<int>(fill_mode));
        seed = hash_combine(seed, max_segment_id);
        seed = hash_combine(seed, num_segments_val);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const segment_max>(rhs);

        return fill_mode == rhs_casted.fill_mode &&
               max_segment_id == rhs_casted.max_segment_id &&
               num_segments_val == rhs_casted.num_segments_val;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<segment_max>::save(ob);
        ob << static_cast<int>(fill_mode);
        ob << max_segment_id;
        ob << num_segments_val;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<segment_max>::load(ib);
        int fm = 0;
        ib >> fm;
        fill_mode = static_cast<ov::op::FillMode>(fm);
        ib >> max_segment_id;
        ib >> num_segments_val;
    }
};
}  // namespace cldnn
