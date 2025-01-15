// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief FakeConvert performs element-wise quantization of input values
///        into a set of values corresponding to a target low-precision type.
struct fake_convert : public primitive_base<fake_convert> {
    CLDNN_DECLARE_PRIMITIVE(fake_convert)

    fake_convert() : primitive_base("", {}) {}

    /// @brief Constructs fake_convert primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param scale Scale primitive id.
    /// @param shift Shift primitive id.
    /// @param destination_type The low precision type to be emulated.
    fake_convert(const primitive_id& id,
             const input_info& input,
             const input_info& scale,
             const input_info& shift,
             ov::element::Type destination_type = ov::element::Type_t::f8e4m3)
        : primitive_base(id, {input, scale, shift}, 1), destination_type(destination_type) {}

    /// @brief Constructs fake_convert primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param scale Scale primitive id.
    /// @param shift Shift primitive id.
    /// @param destination_type The low precision type to be emulated.
    fake_convert(const primitive_id& id,
             const input_info& input,
             const input_info& scale,
             ov::element::Type destination_type = ov::element::Type_t::f8e4m3)
        : primitive_base(id, {input, scale}, 1), destination_type(destination_type) {}

    ov::element::Type destination_type;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, destination_type.get_type_name());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const fake_convert>(rhs);
        return (destination_type == rhs_casted.destination_type);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<fake_convert>::save(ob);
        ob << make_data(&destination_type, sizeof(destination_type));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<fake_convert>::load(ib);
        ib >> make_data(&destination_type, sizeof(destination_type));
    }
};
}  // namespace cldnn
