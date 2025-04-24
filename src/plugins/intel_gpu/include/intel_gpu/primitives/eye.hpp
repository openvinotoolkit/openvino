// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details Construct identity matrix or batch fo them
struct eye : public primitive_base<eye> {
    CLDNN_DECLARE_PRIMITIVE(eye)

    eye() : primitive_base("", {}) {}

    /// @brief Constructs eye primitive.
    /// @param id This primitive id.
    /// @param inputs List of primitive ids.
    /// @param output_shape Tensor output shape
    /// @param ext_prim_id Primitive extra id (friendly name)
    /// @param shift Eye diagonal
    /// @param output_type Tensor output type
    eye(const primitive_id& id,
        const std::vector<input_info>& inputs,
        const tensor& output_shape,
        const int32_t shift,
        const cldnn::data_types output_type)
        : primitive_base{id, inputs, 1, {optional_data_type(output_type)}},
          output_shape{output_shape},
          shift{shift} {}

    tensor output_shape;
    int32_t shift = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, shift);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const eye>(rhs);

        return shift == rhs_casted.shift;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<eye>::save(ob);
        ob << output_shape;
        ob << shift;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<eye>::load(ib);
        ib >> output_shape;
        ib >> shift;
    }
};
}  // namespace cldnn
