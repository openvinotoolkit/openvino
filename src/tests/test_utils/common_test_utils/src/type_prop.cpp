// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"

#include "openvino/core/dimension.hpp"
#include "sequence_generator.hpp"

ov::TensorSymbol get_shape_symbols(const ov::PartialShape& p_shape) {
    ov::TensorSymbol symbols;
    transform(p_shape.cbegin(), p_shape.cend(), back_inserter(symbols), [](const ov::Dimension& dim) {
        return dim.get_symbol();
    });
    return symbols;
}

ov::TensorSymbol set_shape_symbols(ov::PartialShape& p_shape) {
    ov::TensorSymbol symbols;
    for (auto& dim : p_shape) {
        if (!dim.has_symbol()) {
            auto new_symbol = std::make_shared<ov::Symbol>();
            dim.set_symbol(new_symbol);
        }
        symbols.push_back(dim.get_symbol());
    }
    return symbols;
}

void set_shape_symbols(ov::PartialShape& p_shape, const ov::TensorSymbol& symbols) {
    ASSERT_EQ(symbols.size(), p_shape.size());
    for (size_t i = 0; i < p_shape.size(); ++i)
        p_shape[i].set_symbol(symbols[i]);
}
