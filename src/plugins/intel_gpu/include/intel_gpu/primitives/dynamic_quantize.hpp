// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Dynamic Quantize primitive
/// @details Performs dynamic quantization
struct dynamic_quantize : public primitive_base<dynamic_quantize> {
    CLDNN_DECLARE_PRIMITIVE(dynamic_quantize);

    dynamic_quantize() : primitive_base("", {}), group_size(0) {}

    // [TODO] should fix size of outputs
    /// @brief Constructs dynamic_quantize primitive
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param output_size Output data size of the primitive
    dynamic_quantize(const primitive_id& id,
           const input_info& input,
           const size_t group_size,
           const padding& output_padding = padding())
           : primitive_base(id, {input}, {output_padding})
	   , group_size(group_size) {}

    /// @brief Constructs dynamic_quantize primitive
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param data_type Output data type of quantized
    /// @param output_padding Output data size of the primitive
    dynamic_quantize(const primitive_id& id,
           const input_info& input,
           const size_t group_size,
           const data_types data_type,
           const padding& output_padding = padding())
           : primitive_base(id, {input}, {output_padding}, {optional_data_type{data_type}}, 2)
	   , group_size(group_size) {}

    size_t group_size = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, group_size);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const dynamic_quantize>(rhs);

        return group_size == rhs_casted.group_size;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<dynamic_quantize>::save(ob);
        ob << group_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<dynamic_quantize>::load(ib);
        ib >> group_size;
    }
};
}  // namespace cldnn
