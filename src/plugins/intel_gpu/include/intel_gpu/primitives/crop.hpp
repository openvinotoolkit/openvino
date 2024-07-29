// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Select original ngraph op mode for the @ref crop layer.
enum class crop_ngraph_op_mode : int32_t {
    none,
    /// @brief ngraph split op.
    split,
    /// @brief ngraph variadic split op.
    variadic_split
};

/// @brief Marker type indicating that instead of reference input size left, top,
///        right and bottom borders (to cut out) should be specified.
///
/// @details Used to differentiate constructors.
struct crop_borders_t {};

/// @brief Marker indicating that instead of reference input size left, top,
///        right and bottom borders (to cut out) should be specified.
constexpr auto crop_borders = crop_borders_t{};

/// @brief Performs crop operation on input.
/// @details Crops the input to the shape of reference_input across all dimensions taking into account specified input offsets.
/// @n       Borders variant calculated output shape from input shape minus the specified borders.
/// @n\b Requirements (reference size variant)
/// @n - Input size cannot be greater than reference size in any dimension
/// @n - All sizes have to have positive numbers
/// @n - Reference size plus offset cannot exceed input size
/// @n
/// @n\b Requirements (borders variant)
/// @n - Borders support batch, feature and spatial dimensions (rest of dimensions ignored).
/// @n - Input size cannot be greater than reference size in any dimension
/// @n - All sizes specified in borders have to have non-negative values (positive or @c 0).
/// @n - Sum of sizes of opposite borders must be lower than input size (on all non-ignored dimensions).
/// @n
/// @n Breaking any of this conditions will cause exception throw.
struct crop : public primitive_base<crop> {
    CLDNN_DECLARE_PRIMITIVE(crop)

    crop() : primitive_base("", {}) {}

    /// @brief Constructs crop primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param reference_input Reference input tensor with the required dimensions.
    /// @param offsets Input offsets.
    crop(const primitive_id& id,
         const input_info& input,
         const tensor& reference_input,
         const tensor& offsets)
        : primitive_base(id, {input}), reference_input(reference_input),
            offsets(offsets), op_mode(crop_ngraph_op_mode::none) {}

    /// @brief Constructs crop primitive (borders variant).
    ///
    /// @details Allows to specify borders from each side that should be cut out
    ///          by the primitive.
    /// @n       NOTE: Borders variant supports only up to four dimensions.
    ///
    /// @param id         Identifier of newly created primitive.
    /// @param input      Identifier of input primitive which dimensions will be cropped.
    /// @param lt_borders Border sizes (spatial dimensions define left (X) and top (Y)
    ///                   borders, non-spatial dimensions - lower borders)
    /// @param rb_borders Border sizes (spatial dimensions define right (X) and bottom (Y)
    ///                   borders, non-spatial dimensions - upper borders)
    crop(const primitive_id& id,
         const input_info& input,
         const tensor& lt_borders,
         const tensor& rb_borders,
         const crop_borders_t)
        : primitive_base(id, {input}), reference_input(rb_borders.negate()),
            offsets(lt_borders), op_mode(crop_ngraph_op_mode::none) {}

    /// @brief Constructs crop primitive (symmetric borders variant).
    ///
    /// @details Allows to specify borders from each side that should be cut out
    ///          by the primitive.
    /// @n       NOTE: Borders variant supports only up to four dimensions.
    ///
    /// @param id         Identifier of newly created primitive.
    /// @param input      Identifier of input primitive which dimensions will be cropped.
    /// @param xy_borders Border sizes (symmetric; spatial dimensions define left/right (X)
    ///                   and top/bottom (Y) borders, non-spatial dimensions - lower/upper borders).
    crop(const primitive_id& id,
         const input_info& input,
         const tensor& xy_borders,
         const crop_borders_t)
        : primitive_base(id, {input}), reference_input(xy_borders.negate()),
            offsets(xy_borders), op_mode(crop_ngraph_op_mode::none) {}

    /// @brief Constructs crop primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitive id vector.
    /// @param reference_input Reference input tensor with the required dimensions.
    /// @param offsets Input offsets.
    /// @param output_idx Output data index of splited output.
    /// @param axis Axis along data to split.
    /// @param num_splits The number of pieces that the data tensor should be split into.
    crop(const primitive_id& id,
         const std::vector<input_info>& inputs,
         const tensor& reference_input,
         const tensor& offsets,
         const crop_ngraph_op_mode op_mode,
         const int output_idx,
         const int64_t axis = -1,
         const size_t num_splits = 1)
        : primitive_base(id, inputs), reference_input(reference_input), offsets(offsets),
            output_idx(output_idx), axis(axis), num_splits(num_splits), op_mode(op_mode) {}

    /// @brief Reference input tensor with the required dimensions.
    tensor reference_input;
    /// @brief Input offsets.
    tensor offsets;
    /// @brief data index of splited output.
    int output_idx = 0;
    /// @brief axis along data to split
    int64_t axis = -1;
    /// @brief num_splits which Split has number of split as property
    size_t num_splits = 1;
    /// @brief original ngraph operation type
    crop_ngraph_op_mode op_mode = crop_ngraph_op_mode::none;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, reference_input.hash());
        seed = hash_combine(seed, offsets.hash());
        seed = hash_combine(seed, output_idx);
        seed = hash_combine(seed, axis);
        seed = hash_combine(seed, num_splits);
        seed = hash_combine(seed, op_mode);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const crop>(rhs);

        return reference_input == rhs_casted.reference_input &&
               offsets == rhs_casted.offsets &&
               output_idx == rhs_casted.output_idx &&
               axis == rhs_casted.axis &&
               num_splits == rhs_casted.num_splits &&
               op_mode == rhs_casted.op_mode;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<crop>::save(ob);
        ob << reference_input;
        ob << offsets;
        ob << output_idx;
        ob << axis;
        ob << num_splits;
        ob << make_data(&op_mode, sizeof(crop_ngraph_op_mode));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<crop>::load(ib);
        ib >> reference_input;
        ib >> offsets;
        ib >> output_idx;
        ib >> axis;
        ib >> num_splits;
        ib >> make_data(&op_mode, sizeof(crop_ngraph_op_mode));
    }
};
}  // namespace cldnn
