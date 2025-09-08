// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/utils.hpp"

#include "openvino/core/node.hpp"
#include "transformations/utils/utils.hpp"

bool ov::symbol::util::get_symbols(const ov::PartialShape& shape, ov::TensorSymbol& symbols) {
    if (shape.rank().is_dynamic())
        return false;
    symbols.clear();
    symbols.reserve(shape.size());
    for (const auto& d : shape)
        symbols.push_back((d.is_dynamic() ? d.get_symbol() : nullptr));
    return true;
}

bool ov::symbol::util::get_symbols(const ov::Output<ov::Node>& output, ov::TensorSymbol& symbols) {
    const auto& tensor = output.get_tensor();
    symbols = tensor.get_value_symbol();
    return !symbols.empty();
}

bool ov::symbol::util::are_unique_and_equal_symbols(const ov::TensorSymbol& lhs, const ov::TensorSymbol& rhs) {
    if (rhs.size() != lhs.size() || rhs.empty())
        return false;
    for (size_t i = 0; i < lhs.size(); ++i)
        if (!symbol::are_equal(lhs[i], rhs[i]))
            return false;
    return true;
}

bool ov::symbol::util::dims_are_equal(const ov::Dimension& lhs, const ov::Dimension& rhs) {
    if (lhs.is_static() && lhs == rhs)
        return true;
    return symbol::are_equal(lhs.get_symbol(), rhs.get_symbol());
}
