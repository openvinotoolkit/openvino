// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/symbol.hpp"

#include <queue>

ov::SharedSymbol ov::symbol::ancestor_of(const SharedSymbol& symbol) {
    auto x = symbol;
    while (x->m_parent) {
        if (x->m_parent->m_parent)
            x->m_parent = x->m_parent->m_parent;
        x = x->m_parent;
    }
    return x;
}

bool ov::symbol::are_equal(const SharedSymbol& lhs, const SharedSymbol& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return false;
    return ov::symbol::ancestor_of(lhs).get() == ov::symbol::ancestor_of(rhs).get();
}

namespace {  // helpers to work with shared math maps in symbols
void replace_no_check(const ov::SharedSymbol& old_symbol, const ov::SharedSymbol& new_symbol, ov::symbol::MathMap& m) {
    ov::symbol::MathMap map_with_old;
    for (const auto& item : m)
        if (item.second == old_symbol || item.first.has(old_symbol))
            map_with_old.insert(item);
    for (const auto& item : map_with_old)
        m.erase(item.first);
    for (const auto& item : map_with_old) {
        auto new_key = item.first;
        new_key.replace(old_symbol, new_symbol);
        m.insert({new_key, item.second == old_symbol ? new_symbol : item.second});
    }
}

void insert_with_check(const ov::symbol::WeakSymbolVector& key,
                       const ov::symbol::WeakSymbol& value,
                       ov::symbol::MathMap& m,
                       std::queue<std::pair<ov::symbol::WeakSymbol, ov::symbol::WeakSymbol>>& to_equalize) {
    bool insert = true;
    auto it = m.find(key);
    if (it != m.end()) {  // find key, compare value
        insert = false;
        if (it->second != value)
            to_equalize.emplace(it->second, value);
    } else {  // find value, compare key
        for (const auto& item : m) {
            if (item.second == value) {
                if (item.first == key) {
                    insert = false;
                    break;
                } else if (item.first.size() == key.size()) {
                    // try find one element difference to write to equalize
                    ov::symbol::WeakSymbolVector result;
                    std::set_symmetric_difference(item.first.begin(),
                                                  item.first.end(),
                                                  key.begin(),
                                                  key.end(),
                                                  std::back_inserter(result));
                    if (result.size() == 2) {
                        to_equalize.emplace(result[0], result[1]);
                        insert = false;
                    }
                }
            }
        }
    }
    if (insert)
        m.insert({key, value});
}

void replace_inplace_with_check(const ov::SharedSymbol& old_symbol,
                                const ov::SharedSymbol& new_symbol,
                                ov::symbol::MathMap& m,
                                std::queue<std::pair<ov::symbol::WeakSymbol, ov::symbol::WeakSymbol>>& to_equalize) {
    ov::symbol::MathMap with_old;
    for (const auto& item : m)
        if (item.second == old_symbol || item.first.has(old_symbol))
            with_old.insert({item.first, (item.second == old_symbol ? new_symbol : item.second)});
    for (const auto& item : with_old)
        m.erase(item.first);
    for (const auto& item : with_old) {
        auto new_key = item.first;
        new_key.replace(old_symbol, new_symbol);
        insert_with_check(new_key, item.second, m, to_equalize);
    }
}

void replace_with_check(const ov::SharedSymbol& old_symbol,
                        const ov::SharedSymbol& new_symbol,
                        ov::symbol::MathMap& old_map,
                        ov::symbol::MathMap& new_map,
                        std::queue<std::pair<ov::symbol::WeakSymbol, ov::symbol::WeakSymbol>>& to_equalize) {
    ov::symbol::MathMap with_new;
    for (const auto& item : new_map)
        if (item.second == new_symbol || item.first.has(new_symbol))
            with_new.insert({item.first, item.second});
    for (const auto& item : with_new)
        new_map.erase(item.first);
    // new_map contains only independent records, with_new contains records with new_symbol
    ov::symbol::MathMap with_old;
    for (const auto& item : old_map) {
        if (item.second == old_symbol || item.first.has(old_symbol)) {
            auto new_key = item.first;
            new_key.replace(old_symbol, new_symbol);
            with_old.insert({new_key, (item.second == old_symbol ? new_symbol : item.second)});
        } else {
            new_map.insert({item.first, item.second});
        }
    }
    // new_map contains independent records from new and from old map; old_map won't be used further
    old_map.clear();
    // with_old contains records from old_map that had old_symbol, old_symbol was replaced with new_symbol

    // merging with_new and with_old, performing necessary checks to figure out if we have more equality rules
    for (const auto& item : with_old)
        insert_with_check(item.first, item.second, with_new, to_equalize);
    // with_old is merged into the with_new; necessary checks were made to ensure all new equality rules are collected

    for (const auto& item : with_new)
        new_map.insert({item.first, item.second});
}

void collect_records_with_result(const ov::SharedSymbol& result,
                                 const std::shared_ptr<ov::symbol::MathMap>& map,
                                 std::vector<ov::symbol::WeakSymbolVector>& records) {
    if (map)
        for (const auto& item : *map)
            if (item.second.lock() && item.second.lock().get() == result.get())
                records.push_back(item.first);
}
}  // namespace

void ov::symbol::set_equal(const SharedSymbol& l, const SharedSymbol& r) {
    std::queue<std::pair<ov::symbol::WeakSymbol, ov::symbol::WeakSymbol>> to_equalize;
    to_equalize.emplace(l, r);

    do {
        auto item = to_equalize.front();
        to_equalize.pop();
        auto lhs = item.first.lock(), rhs = item.second.lock();
        if (!lhs || !rhs || ov::symbol::are_equal(lhs, rhs))
            continue;  // invalid or already are equal
        auto A = ancestor_of(lhs), B = ancestor_of(rhs);
        A->m_parent = B;

        // m_add unification
        if (A->m_add && !B->m_add) {
            replace_no_check(A, B, *A->m_add);
            B->m_add = A->m_add;  // rhs is the root of lhs now
            A->m_add = nullptr;
        } else if (A->m_add && B->m_add && A->m_add.get() == B->m_add.get()) {
            replace_inplace_with_check(A, B, *B->m_add, to_equalize);
            A->m_add = nullptr;  // rhs is the root of lhs now
        } else if (A->m_add && B->m_add) {
            replace_with_check(A, B, *A->m_add, *B->m_add, to_equalize);
            for (auto& i : *B->m_add) {
                for (auto& j : i.first) {
                    if (!j.expired())
                        j.lock()->m_add = B->m_add;
                }
                if (!i.second.expired())
                    i.second.lock()->m_add = B->m_add;
            }
            A->m_add = nullptr;
        }
    } while (!to_equalize.empty());
}

ov::Symbol::~Symbol() {
    if (m_add) {
        for (auto item = m_add->begin(), last = m_add->end(); item != last;) {
            if (item->first.has(nullptr) || item->first.has(this) || item->second.expired() ||
                item->second.lock().get() == this)
                item = m_add->erase(item);
            else
                ++item;
        }
        m_add = nullptr;
    }
}

ov::SharedSymbol ov::operator+(const SharedSymbol& lhs, const SharedSymbol& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return nullptr;  // should we create a new shared_ptr to a symbol here?
    // A + B = C
    auto A = symbol::ancestor_of(lhs);
    auto B = symbol::ancestor_of(rhs);
    auto shared_map = std::make_shared<symbol::MathMap>();
    if (A->m_add && B->m_add && A->m_add.get() == B->m_add.get()) {  // maps are shared
        auto it = A->m_add->find({A, B});
        if (it != A->m_add->end() && it->second.lock())  // elements were summed before
            return it->second.lock();
        shared_map = A->m_add;
    } else {
        if (A->m_add)
            shared_map->insert(A->m_add->begin(), A->m_add->end());
        if (B->m_add)
            shared_map->insert(B->m_add->begin(), B->m_add->end());
        A->m_add = shared_map;
        B->m_add = shared_map;
        for (auto& item : *shared_map) {
            for (auto& key_weak : item.first)
                if (auto key = key_weak.lock())
                    key->m_add = shared_map;
            if (auto value = item.second.lock())
                value->m_add = shared_map;
        }
    }
    auto C = std::make_shared<ov::Symbol>();
    C->m_add = shared_map;
    // add L + R = result  to the shared map
    // search for X + Y = L and W + Z = R records to make X + Y + W + Z = result records
    std::vector<ov::symbol::WeakSymbolVector> this_components{{A}}, other_components{{B}};
    collect_records_with_result(A, shared_map, this_components);
    collect_records_with_result(B, shared_map, other_components);
    for (const auto& this_element : this_components) {
        for (const auto& other_element : other_components) {
            std::vector<ov::symbol::WeakSymbol> new_key;
            new_key.insert(new_key.begin(), this_element.begin(), this_element.end());
            new_key.insert(new_key.begin(), other_element.begin(), other_element.end());
            shared_map->insert({{new_key}, C});
        }
    }
    return C;
}

ov::SharedSymbol ov::operator-(const SharedSymbol& lhs, const SharedSymbol& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return nullptr;  // should we create a new shared_ptr<Symbol> here?
    // A - B = C  =>  B + C = A
    auto A = symbol::ancestor_of(lhs);
    auto B = symbol::ancestor_of(rhs);

    auto shared_map = std::make_shared<symbol::MathMap>();
    if (A->m_add && B->m_add && A->m_add.get() == B->m_add.get()) {  // maps are shared
        std::vector<ov::symbol::WeakSymbolVector> A_components{}, B_components{};
        collect_records_with_result(A, A->m_add, A_components);
        for (const auto& item : A_components) {
            if (item.size() == 2 && item[0] == B && !item[1].expired())
                return item[1].lock();
            if (item.size() == 2 && item[1] == B && !item[0].expired())
                return item[0].lock();
        }
        // TODO: search for components of A and B and try to find one element difference of it to return
        shared_map = A->m_add;
    } else {
        if (A->m_add)
            shared_map->insert(A->m_add->begin(), A->m_add->end());
        if (B->m_add)
            shared_map->insert(B->m_add->begin(), B->m_add->end());
        A->m_add = shared_map;
        B->m_add = shared_map;
        for (auto& item : *shared_map) {
            for (auto& key_weak : item.first)
                if (auto key = key_weak.lock())
                    key->m_add = shared_map;
            if (auto value = item.second.lock())
                value->m_add = shared_map;
        }
    }
    auto C = std::make_shared<ov::Symbol>();
    C->m_add = shared_map;

    std::vector<ov::symbol::WeakSymbolVector> B_components{{B}};
    collect_records_with_result(B, B->m_add, B_components);
    for (const auto& B_elements : B_components) {
        std::vector<ov::symbol::WeakSymbol> new_key;
        new_key.insert(new_key.begin(), B_elements.begin(), B_elements.end());
        new_key.insert(new_key.begin(), C);
        shared_map->insert({{new_key}, A});
    }
    return C;
}
