// Copyright (C) 2018-2024 Intel Corporation
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
    lhs_root->m_parent = rhs_root;
}

std::shared_ptr<ov::Symbol> ov::symbol::operator+(const std::shared_ptr<ov::Symbol> &l, const std::shared_ptr<ov::Symbol> &r) {
    if (l == nullptr || r == nullptr)
        return nullptr; // should we create a new shared_ptr to a symbol here?
    auto lhs = ancestor_of(l);
    auto rhs = ancestor_of(r);

    auto shared_map = std::make_shared<std::unordered_map<ov::symbol::WeakSymbolVector, ov::symbol::WeakSymbol>>();
    if (lhs->m_add && rhs->m_add && lhs->m_add.get() == rhs->m_add.get()) { // maps are shared
        auto search_sum = ov::symbol::WeakSymbolVector({lhs, rhs});
        auto it = lhs->m_add->find(search_sum);
        if (it != lhs->m_add->end()) // elements were summed before
            return it->second.lock();
        shared_map = lhs->m_add;
    } else {
        if (lhs->m_add)
            shared_map->insert(lhs->m_add->begin(), lhs->m_add->end());
        if (rhs->m_add)
            shared_map->insert(rhs->m_add->begin(), rhs->m_add->end());
        lhs->m_add = shared_map;
        rhs->m_add = shared_map;
        for (auto& item : *shared_map) {
            for (auto& i : item.first)
                i.lock()->m_add = shared_map;
            item.second.lock()->m_add = shared_map;
        }
    }
    auto result = std::make_shared<ov::Symbol>();
    result->m_add = shared_map;
    // add this + other = R to the shared map
    // search for X + Y = this and W + Z = other records to make X + Y + W + Z = R records
    std::vector<ov::symbol::WeakSymbolVector> this_components{{lhs}}, other_components{{rhs}};
    auto collect_records_with_result = [](const std::shared_ptr<ov::Symbol> &result, std::vector<ov::symbol::WeakSymbolVector>& records) {
        if (auto map = result->m_add)
            for (const auto& item : *map)
                if (item.second.lock().get() == result.get())
                    records.push_back(item.first);
    };
    collect_records_with_result(lhs, this_components);
    collect_records_with_result(rhs, other_components);
    for (const auto& this_element : this_components) {
        for (const auto& other_element: other_components) {
            std::vector<ov::symbol::WeakSymbol> new_key;
            new_key.insert(new_key.begin(), this_element.begin(), this_element.end());
            new_key.insert(new_key.begin(), other_element.begin(), other_element.end());
            shared_map->insert({{new_key}, result});
        }
    }
    return result;
}

std::shared_ptr<ov::Symbol> ov::symbol::operator-(const std::shared_ptr<ov::Symbol> &lhs, const std::shared_ptr<ov::Symbol> &rhs) {
    // FIXME: A - B = C; B + C = A;

    return nullptr;
}

