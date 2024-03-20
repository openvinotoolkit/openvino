// Copyright (C) 2018-2023 Intel Corporation
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
        if (lhs[i] == nullptr || rhs[i] == nullptr || !lhs[i]->is_equal_to(rhs[i]))
            return false;
    return true;
}

bool ov::symbol::util::dims_are_equal(const ov::Dimension& lhs, const ov::Dimension& rhs) {
    if (lhs.is_static() && lhs == rhs)
        return true;
    auto lhs_symbol = lhs.get_symbol();
    auto rhs_symbol = rhs.get_symbol();
    if (lhs_symbol == nullptr || rhs_symbol == nullptr)
        return false;
    return lhs_symbol->is_equal_to(rhs_symbol);
}
