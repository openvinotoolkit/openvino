// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/symbol.hpp"

ov::SymbolKind ov::Symbol::get_kind() const {
    return m_kind;
}

bool ov::Symbol::is_leaf() const {
    return m_kind == SymbolKind::LEAF;
}

bool ov::Symbol::is_compound() const {
    return m_kind != SymbolKind::LEAF;
}

const std::shared_ptr<ov::Symbol>& ov::Symbol::get_lhs() const {
    return m_lhs;
}

const std::shared_ptr<ov::Symbol>& ov::Symbol::get_rhs() const {
    return m_rhs;
}

std::shared_ptr<ov::Symbol> ov::symbol::ancestor_of(const std::shared_ptr<Symbol>& symbol) {
    if (symbol->is_compound())
        return symbol;  // compound symbols have no parent chain
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
    if (lhs->is_compound() || rhs->is_compound())
        return structurally_equal(lhs, rhs);
    return ov::symbol::ancestor_of(lhs).get() == ov::symbol::ancestor_of(rhs).get();
}

void ov::symbol::set_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return;
    // union-find is only for leaf symbols
    if (lhs->is_compound() || rhs->is_compound())
        return;
    auto lhs_root = ov::symbol::ancestor_of(lhs), rhs_root = ov::symbol::ancestor_of(rhs);
    if (lhs_root.get() == rhs_root.get())
        return;  // already are equal
    lhs_root->m_parent = std::move(rhs_root);
}

std::shared_ptr<ov::Symbol> ov::symbol::add(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return nullptr;
    auto result = std::make_shared<Symbol>();
    result->m_kind = SymbolKind::ADD;
    result->m_lhs = lhs;
    result->m_rhs = rhs;
    return result;
}

std::shared_ptr<ov::Symbol> ov::symbol::mul(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return nullptr;
    auto result = std::make_shared<Symbol>();
    result->m_kind = SymbolKind::MUL;
    result->m_lhs = lhs;
    result->m_rhs = rhs;
    return result;
}

bool ov::symbol::structurally_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return false;
    if (lhs.get() == rhs.get())
        return true;

    // both leaves: use union-find
    if (lhs->is_leaf() && rhs->is_leaf())
        return ancestor_of(lhs).get() == ancestor_of(rhs).get();

    // kind mismatch
    if (lhs->get_kind() != rhs->get_kind())
        return false;

    // both compound with same kind: check operands
    // ADD and MUL are commutative: (A op B) == (A op B) || (A op B) == (B op A)
    if (lhs->get_kind() == SymbolKind::ADD || lhs->get_kind() == SymbolKind::MUL) {
        if (structurally_equal(lhs->get_lhs(), rhs->get_lhs()) &&
            structurally_equal(lhs->get_rhs(), rhs->get_rhs()))
            return true;
        if (structurally_equal(lhs->get_lhs(), rhs->get_rhs()) &&
            structurally_equal(lhs->get_rhs(), rhs->get_lhs()))
            return true;
        return false;
    }

    // @todo handle other compound kinds (SUB) when added
    return false;
}
