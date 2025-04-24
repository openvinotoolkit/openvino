// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Changes information about inputs's layout effectively creating new memory which share underlaying buffer
/// but is interpreted in a different way (different shape).
/// @note reshape primitive is supposed only to reinterpret shape of the memory therefore it's not possible to change
/// neither data type nor format of the input buffer and total number of elements in input and output (excluding paddings) must match.
/// Please note that there is no guarantee that underlying data will be in proper format if primitive was explicitly added to output list.
struct reshape : public primitive_base<reshape> {
    CLDNN_DECLARE_PRIMITIVE(reshape)

    reshape() : primitive_base("", {}) {}

    enum reshape_mode : uint32_t {
        base,
        squeeze,
        unsqueeze
    };

    /// @brief Constructs reshape primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_shape Requested memory shape (excluding padding).
    /// A dimension could be 0, in this case,  the value is taken from the input tensor.
    /// At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions.
    reshape(const primitive_id& id,
            const input_info& input,
            const tensor& output_shape,
            reshape_mode mode = reshape_mode::base)
        : primitive_base(id, {input})
        , output_shape(output_shape)
        , output_pattern({})
        , output_partial_shape({})
        , mode(mode) {}

    /// @brief reshape with dynamic pattern
    reshape(const primitive_id& id,
            const input_info& input,
            const input_info& pattern_id,
            bool special_zero,
            const ov::PartialShape& output_partial_shape,
            reshape_mode mode = reshape_mode::base)
        : primitive_base(id, {input, pattern_id})
        , output_shape(tensor())
        , special_zero(special_zero)
        , output_pattern({})
        , output_partial_shape(output_partial_shape)
        , mode(mode) {}

    /// @brief reshape with static pattern
    reshape(const primitive_id& id,
            const input_info& input,
            bool special_zero,
            const std::vector<int64_t>& output_pattern,
            const ov::PartialShape& output_partial_shape,
            reshape_mode mode = reshape_mode::base)
        : primitive_base(id, {input})
        , output_shape(tensor())
        , special_zero(special_zero)
        , output_pattern(output_pattern)
        , output_partial_shape(output_partial_shape)
        , mode(mode) {}

    /// @brief Requested memory shape.
    tensor output_shape;

    bool special_zero = false;

    std::vector<int64_t> output_pattern;

    ov::PartialShape output_partial_shape;

    reshape_mode mode = reshape_mode::base;

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const reshape>(rhs);

        return special_zero == rhs_casted.special_zero &&
               mode == rhs_casted.mode;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<reshape>::save(ob);
        ob << output_shape;
        ob << special_zero;
        ob << output_pattern;
        ob << output_partial_shape;
        ob << make_data(&mode, sizeof(reshape_mode));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<reshape>::load(ib);
        ib >> output_shape;
        ib >> special_zero;
        ib >> output_pattern;
        ib >> output_partial_shape;
        ib >> make_data(&mode, sizeof(reshape_mode));
    }
};

}  // namespace cldnn
