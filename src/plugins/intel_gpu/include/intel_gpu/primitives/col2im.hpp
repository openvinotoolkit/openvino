// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct col2im : public primitive_base<col2im> {
    CLDNN_DECLARE_PRIMITIVE(col2im)

    col2im() : primitive_base("", {}) {}

    /// @brief Constructs col2im primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param stride Defines shift in input buffer
    /// @param dilation Defines gaps in the input
    /// @param padding_begin Defines a padding added to input image on left (x axis) and top (y axis).
    /// @param padding_end Defines a padding added to input image on right (x axis) and bottom (y axis).
    /// @param output_shape Defines the output tensor the output image
    /// @param kernel_shape Defines size of the sliding blocks
    col2im(const primitive_id& id,
                   const input_info& input,
                   ov::Strides stride,
                   ov::Strides dilation,
                   ov::CoordinateDiff padding_begin,
                   ov::CoordinateDiff padding_end,
                   ov::Shape output_shape,
                   ov::Shape kernel_shape)
        : primitive_base(id, {input})
        , stride(stride)
        , dilation(dilation)
        , padding_begin(padding_begin)
        , padding_end(padding_end)
        , output_shape(output_shape)
        , kernel_shape(kernel_shape) {}

    /// @brief Defines shift in input buffer
    ov::Strides stride;
    // @brief Defines gaps in the input
    ov::Strides dilation;
    /// @param padding_begin Defines a padding added to input image on left (x axis) and top (y axis).
    ov::CoordinateDiff padding_begin;
    /// @param padding_end Defines a padding added to input image on right (x axis) and bottom (y axis).
    ov::CoordinateDiff padding_end;
    ov::Shape output_shape;
    ov::Shape kernel_shape;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, padding_end.begin(), padding_end.end());
        seed = hash_range(seed, padding_begin.begin(), padding_begin.end());
        seed = hash_range(seed, dilation.begin(), dilation.end());
        seed = hash_range(seed, stride.begin(), stride.end());
        seed = hash_range(seed, output_shape.begin(), output_shape.end());
        seed = hash_range(seed, kernel_shape.begin(), kernel_shape.end());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const col2im>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(stride) &&
               cmp_fields(dilation) &&
               cmp_fields(padding_begin) &&
               cmp_fields(padding_end) &&
               cmp_fields(output_shape) &&
               cmp_fields(kernel_shape);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<col2im>::save(ob);
        ob << stride;
        ob << dilation;
        ob << padding_begin;
        ob << padding_end;
        ob << output_shape;
        ob << kernel_shape;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<col2im>::load(ib);
        ib >> stride;
        ib >> dilation;
        ib >> padding_begin;
        ib >> padding_end;
        ib >> output_shape;
        ib >> kernel_shape;
    }
};
}  // namespace cldnn
