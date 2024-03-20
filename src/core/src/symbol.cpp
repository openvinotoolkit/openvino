// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/symbol.hpp"

#include <algorithm>

ov::Symbol::Symbol(ov::Symbol& s) : Symbol() {
    parent = s.get_parent();
}

void ov::Symbol::set_equal(const std::shared_ptr<Symbol>& other) {
    auto x = root(), y = other->root();
    if (x == y)
        return;  // already set as equal
    if (x->rank() < y->rank())
        std::swap(x, y);
    y->parent = x;
}

bool ov::Symbol::is_equal_to(const std::shared_ptr<Symbol>& other) {
    return root() == other->root();
}

size_t ov::Symbol::rank() {
    size_t rank = 0;
    std::shared_ptr<ov::Symbol> x = shared_from_this();
    while (x->get_parent().get() != x.get()) {
        x = x->get_parent();
        ++rank;
    }
    return rank;
}

std::shared_ptr<ov::Symbol> ov::Symbol::get_parent() {
    if (!parent)
        parent = shared_from_this();
    return parent;
}

std::shared_ptr<ov::Symbol> ov::Symbol::root() {
    ov::Symbol* x = this;
    while (x->get_parent().get() != x) {
        x->parent = x->get_parent()->get_parent();
        x = x->get_parent().get();
    }
    return x->shared_from_this();
}

bool ov::Symbol::are_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return false;
    return lhs->is_equal_to(rhs);
}

bool ov::Symbol::set_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return false;
    lhs->set_equal(rhs);
    return true;
}
