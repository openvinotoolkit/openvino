// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "openvino/core/coordinate_diff.hpp"

namespace cldnn {

/// @brief Adds border around input.
///
/// @details Applies border of specified type around input data. The size of output data is increased or decreased
///          by @c pads_begin and by @c pads_end.
/// @n
/// @n@b Requirements:
/// @n - @c pads_begin and @c pads_end must be compatible
///      with size of input (describe the same dimensions).
/// @n - For @c PadMode equal to @c SYMMETRIC, @c pads_begin and @c pads_end
///      must be lower than or equal to size of input on corresponding dimension (for all dimensions)
/// @n - For @c PadMode equal to @c REFLECT, @c pads_begin and @c pads_end
///      must be lower than size of input on corresponding dimension (for all dimensions)
/// @n Breaking any of this conditions will cause exception throw.
struct border : public primitive_base<border> {
    CLDNN_DECLARE_PRIMITIVE(border)

    border() : primitive_base("", {}) {}

    /// @brief whether the input is const or not
    enum PAD_NON_CONST_INPUT {
        BEGIN = 0x1,
        END = (0x1 << 1),
        VALUE = (0x1 << 2)
    };

    /// @brief Constructs border primitive / layer
    ///
    /// @param id                       An identifier of new primitive.
    /// @param inputs                   An identifier list of primitives which are not constant input.
    /// @param non_constant_input_mask  Bit mask whether inputs are non-constant or not
    /// @param pads_begin               Sizes of border that needs to be added (or removed) from left
    ///                                 (in X dimension) and from top (in Y dimension).
    /// @param pads_end                 Sizes of border that needs to be added (or removed) from right
    ///                                 (in X dimension) and from bottom (in Y dimension).
    /// @param pad_mode                 Value of elements which is used for paddings
    /// @param pad_value                Pad's value in case of PadMode::CONSTANT
    /// @param allow_negative_pad       Allow negative values in pads_begin and pad_end to remove borders
    border(const primitive_id& id,
           const std::vector<input_info>& inputs,
           int32_t non_constant_input_mask = 0,
           const ov::CoordinateDiff& pads_begin = {},
           const ov::CoordinateDiff& pads_end = {},
           const ov::op::PadMode pad_mode = ov::op::PadMode::CONSTANT,
           const float pad_value = 0.0f,
           const bool allow_negative_pad = false)
        : primitive_base(id, inputs),
          pads_begin(pads_begin),
          pads_end(pads_end),
          pad_mode(pad_mode),
          pad_value(pad_value),
          allow_negative_pad(allow_negative_pad),
          non_constant_input_mask(non_constant_input_mask) {}

    /// @brief Sizes of border that needs to be added from left (in X dimension) and from top (in Y dimension).
    ov::CoordinateDiff pads_begin;
    /// @brief Sizes of border that needs to be added from right (in X dimension) and from bottom (in Y dimension).
    ov::CoordinateDiff pads_end;
    /// @brief Type of border that needs to be added to the input.
    ov::op::PadMode pad_mode = ov::op::PadMode::CONSTANT;
    /// @brief Border value that is used in constant mode.
    float pad_value{0.0};
    /// @brief Allow negative values in pads_begin and pad_end.
    bool allow_negative_pad{false};
    /// @brief Bit mask whether input is non-constant or not. Position is defined at PAD_NON_CONST_INPUT.
    int32_t non_constant_input_mask = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, pads_begin.begin(), pads_begin.end());
        seed = hash_range(seed, pads_end.begin(), pads_end.end());
        seed = hash_combine(seed, pad_mode);
        seed = hash_combine(seed, pad_value);
        seed = hash_combine(seed, allow_negative_pad);
        seed = hash_combine(seed, non_constant_input_mask);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const border>(rhs);

        return pads_begin == rhs_casted.pads_begin &&
               pads_end == rhs_casted.pads_end &&
               pad_mode == rhs_casted.pad_mode &&
               pad_value == rhs_casted.pad_value &&
               allow_negative_pad == rhs_casted.allow_negative_pad;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<border>::save(ob);
        ob << pads_begin;
        ob << pads_end;
        ob << make_data(&pad_mode, sizeof(ov::op::PadMode));
        ob << pad_value;
        ob << non_constant_input_mask;
        ob << allow_negative_pad;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<border>::load(ib);
        ib >> pads_begin;
        ib >> pads_end;
        ib >> make_data(&pad_mode, sizeof(ov::op::PadMode));
        ib >> pad_value;
        ib >> non_constant_input_mask;
        ib >> allow_negative_pad;
    }
};
}  // namespace cldnn
