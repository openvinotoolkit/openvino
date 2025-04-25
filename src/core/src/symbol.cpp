// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/symbol.hpp"

std::shared_ptr<ov::Symbol> ov::symbol::ancestor_of(const std::shared_ptr<Symbol>& symbol) {
    auto x = symbol;
    while (x->m_parent) {
        if (x->m_parent->m_parent)
            x->m_parent = x->m_parent->m_parent;
        x = x->m_parent;
    }
    return x;
}

bool ov::symbol::are_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return false;
    return ov::symbol::ancestor_of(lhs).get() == ov::symbol::ancestor_of(rhs).get();
}

void ov::symbol::set_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return;
    auto lhs_root = ov::symbol::ancestor_of(lhs), rhs_root = ov::symbol::ancestor_of(rhs);
    if (lhs_root.get() == rhs_root.get())
        return;  // already are equal
    lhs_root->m_parent = std::move(rhs_root);
}
