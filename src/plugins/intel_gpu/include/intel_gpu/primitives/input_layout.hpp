// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/runtime/layout.hpp"
#include "primitive.hpp"
#include "intel_gpu/runtime/memory.hpp"

namespace cldnn {

/// @brief Provides input layout for a data to be passed later to network.
/// @details This primitive allows to define the layout for input data
/// which will be passed to network before execution.
/// For example, network input images.
/// @note User should call network::set_input_data() for every @p input_layout primitive before network execution.
/// @note @p output_padding property of @p input_layout is ignored - its output layout is always equal to input layout defined during object creation.
/// @sa network::set_input_data(), cldnn::data
struct input_layout : public primitive_base<input_layout> {
    CLDNN_DECLARE_PRIMITIVE(input_layout)

    input_layout() : primitive_base("", {}) {}

    /// @brief Constructs input layout primitive.
    /// @param id This primitive id.
    /// @param layout Defines layout for the data will be passed to network.
    input_layout(const primitive_id& id, const layout& layout)
        : primitive_base(id, {}, 1, {optional_data_type()}, {layout.data_padding}), layout(layout) {}

    /// @brief Defines layout for the data will be passed to network.
    mutable cldnn::layout layout;

    void change_layout(const cldnn::layout& new_layout) {
        layout = new_layout;
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, id);
        return seed;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<input_layout>::save(ob);
        ob << layout;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<input_layout>::load(ib);
        ib >> layout;
    }
};
}  // namespace cldnn
