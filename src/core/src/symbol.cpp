// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/symbol.hpp"

#include <algorithm>
#include <queue>

ov::SymbolPtr ov::symbol::ancestor_of(const SymbolPtr& symbol) {
    auto x = symbol;
    while (x->m_parent) {
        if (x->m_parent->m_parent)
            x->m_parent = x->m_parent->m_parent;
        x = x->m_parent;
    }
    return x;
}

bool ov::symbol::are_equal(const SymbolPtr& lhs, const SymbolPtr& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return false;
    return ancestor_of(lhs).get() == ancestor_of(rhs).get();
}

std::size_t ov::symbol::WeakSymbolVectorHash::operator()(const WeakSymbolVector& v) const {
    size_t seed = 0;
    for (const auto& element : v) {
        const auto& el_hash = element.expired() ? 0 : std::hash<SymbolPtr>()(element.lock());
        seed ^= el_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

namespace ov {
inline bool operator==(const symbol::WeakSymbol& lhs, const symbol::WeakSymbol& rhs) {
    if (lhs.expired() && rhs.expired())
        return true;
    if (lhs.expired() || rhs.expired())
        return false;
    return lhs.lock().get() == rhs.lock().get();
}

inline bool operator!=(const symbol::WeakSymbol& lhs, const symbol::WeakSymbol& rhs) {
    return !(lhs == rhs);
}

inline bool operator<(const symbol::WeakSymbol& lhs, const symbol::WeakSymbol& rhs) {
    return std::owner_less<ov::SymbolPtr>()(lhs.lock(), rhs.lock());
}
}  // namespace ov

namespace {  // helpers to work with shared math maps in symbols
bool contains(const ov::symbol::WeakSymbolVector& vec, const ov::symbol::WeakSymbol& s) {
    return std::find(vec.cbegin(), vec.cend(), s) != vec.cend();
}

bool contains(const ov::symbol::WeakSymbolVector& vec, const ov::Symbol* s) {
    return std::any_of(vec.begin(), vec.end(), [&s](const ov::symbol::WeakSymbol& i) {
        if (i.lock() == nullptr)
            return s == nullptr;
        return i.lock().get() == s;
    });
}

void sort(ov::symbol::WeakSymbolVector& vec) {
    std::sort(vec.begin(), vec.end());
}

void replace(ov::symbol::WeakSymbolVector& vec,
             const ov::symbol::WeakSymbol& old_s,
             const ov::symbol::WeakSymbol& new_s) {
    std::replace_if(
        vec.begin(),
        vec.end(),
        [&old_s](const ov::symbol::WeakSymbol& s) {
            return s == old_s;
        },
        new_s);
    sort(vec);
}

void replace_no_check(const ov::SymbolPtr& old_symbol, const ov::SymbolPtr& new_symbol, ov::symbol::MathMap& m) {
    ov::symbol::MathMap map_with_old;
    for (const auto& item : m)
        if (item.second == ov::symbol::WeakSymbol(old_symbol) || contains(item.first, old_symbol.get()))
            map_with_old.insert(item);
    for (const auto& item : map_with_old)
        m.erase(item.first);
    for (const auto& item : map_with_old) {
        auto new_key = item.first;
        replace(new_key, old_symbol, new_symbol);
        m.insert({new_key, item.second == old_symbol ? new_symbol : item.second});
    }
}

void remove_same_elements(ov::symbol::WeakSymbolVector& lhs, ov::symbol::WeakSymbolVector& rhs) {
    for (auto lhs_it = lhs.begin(); lhs_it != lhs.end();) {
        if (lhs_it->expired())
            continue;
        auto rhs_it = std::find(rhs.cbegin(), rhs.cend(), *lhs_it);
        if (rhs_it != rhs.end()) {
            rhs.erase(rhs_it);
            lhs_it = lhs.erase(lhs_it);
        } else {
            ++lhs_it;
        }
    }
}

void insert_with_check(const ov::symbol::WeakSymbolVector& key,
                       const ov::symbol::WeakSymbol& value,
                       ov::symbol::MathMap& m,
                       std::queue<std::pair<ov::symbol::WeakSymbol, ov::symbol::WeakSymbol>>& to_equalize) {
    bool insert = true;
    auto it = m.find(key);
    if (it != m.end()) {
        // map m already contains the record with same key, no need to insert new record into the map m
        insert = false;
        if (it->second != value)
            // if the record had different value, then we need to equalize original and new key
            to_equalize.emplace(it->second, value);
    } else {
        // map m doesn't have records with the key, however we are in a search for new equality rules
        for (const auto& item : m) {
            if (item.second == value) {  // map_ker
                if (item.first.size() == key.size()) {
                    auto item_diff = item.first, key_diff = key;
                    remove_same_elements(item_diff, key_diff);
                    if (item_diff.size() == 1 && key_diff.size() == 1) {
                        to_equalize.emplace(item_diff[0], key_diff[0]);
                        insert = false;
                    }
                }
            }
        }
    }
    if (insert)
        m.insert({key, value});
}

void replace_inplace_with_check(const ov::SymbolPtr& old_symbol,
                                const ov::SymbolPtr& new_symbol,
                                ov::symbol::MathMap& m,
                                std::queue<std::pair<ov::symbol::WeakSymbol, ov::symbol::WeakSymbol>>& to_equalize) {
    ov::symbol::MathMap with_old;
    for (const auto& item : m)
        if (item.second == old_symbol || contains(item.first, old_symbol))
            with_old.insert({item.first, item.second});
    for (const auto& item : with_old)
        m.erase(item.first);
    for (const auto& item : with_old) {
        auto new_key = item.first;
        replace(new_key, old_symbol, new_symbol);
        insert_with_check(new_key, (item.second == old_symbol ? new_symbol : item.second), m, to_equalize);
    }
}

void replace_with_check(const ov::SymbolPtr& old_symbol,
                        const ov::SymbolPtr& new_symbol,
                        ov::symbol::MathMap& old_map,
                        ov::symbol::MathMap& new_map,
                        std::queue<std::pair<ov::symbol::WeakSymbol, ov::symbol::WeakSymbol>>& to_equalize) {
    ov::symbol::MathMap with_new;
    for (const auto& item : new_map)
        if (item.second == new_symbol || contains(item.first, new_symbol))
            with_new.insert({item.first, item.second});
    for (const auto& item : with_new)
        new_map.erase(item.first);
    // new_map contains only independent records, with_new contains records with new_symbol
    ov::symbol::MathMap with_old;
    for (const auto& item : old_map) {
        if (item.second == old_symbol || contains(item.first, old_symbol)) {
            auto new_key = item.first;
            replace(new_key, old_symbol, new_symbol);
            with_old.insert({new_key, item.second});
        } else {
            new_map.insert({item.first, item.second});
        }
    }
    // new_map contains independent records from new and from old map; old_map won't be used further
    old_map.clear();
    // with_old contains records from old_map that had old_symbol, old_symbol was replaced with new_symbol

    // merging with_new and with_old, performing necessary checks to figure out if we have more equality rules
    for (const auto& item : with_old)
        insert_with_check(item.first, (item.second == old_symbol ? new_symbol : item.second), with_new, to_equalize);
    // with_old is merged into the with_new; necessary checks were made to ensure all new equality rules are collected

    for (const auto& item : with_new)
        new_map.insert({item.first, item.second});
}

void collect_records_with_result(const ov::SymbolPtr& result,
                                 const std::shared_ptr<ov::symbol::MathMap>& map,
                                 std::vector<ov::symbol::WeakSymbolVector>& records) {
    if (map)
        for (const auto& item : *map)
            if (item.second.lock() && item.second.lock().get() == result.get())
                records.push_back(item.first);
}

void share_map(const std::shared_ptr<ov::symbol::MathMap>& shared_map,
               const std::shared_ptr<ov::symbol::MathMap>& donor_map,
               void (*set_map)(const ov::symbol::WeakSymbol&, const std::shared_ptr<ov::symbol::MathMap>&)) {
    if (!donor_map || donor_map->empty())
        return;
    for (const auto& item : *donor_map) {
        auto key = item.first;
        for (const auto& key_element : key)
            set_map(key_element, shared_map);
        auto value = item.second;
        set_map(value, shared_map);
        shared_map->insert({key, value});
    }
}
}  // namespace

void ov::symbol::set_equal(const SymbolPtr& l, const SymbolPtr& r) {
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
            if (contains(item->first, nullptr) || contains(item->first, this) || item->second.expired() ||
                item->second.lock().get() == this) {
                item = m_add->erase(item);
            } else {
                ++item;
            }
        }
        m_add = nullptr;
    }
}

ov::SymbolPtr ov::operator+(const SymbolPtr& lhs, const SymbolPtr& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return nullptr;  // should we create a new shared_ptr to a symbol here?
    // A + B = C
    auto A = symbol::ancestor_of(lhs);
    auto B = symbol::ancestor_of(rhs);
    auto new_key = ov::symbol::WeakSymbolVector({A, B});
    sort(new_key);
    std::shared_ptr<ov::symbol::MathMap> shared_map = nullptr;
    if (A->m_add && B->m_add && A->m_add.get() == B->m_add.get()) {  // maps are shared
        auto it = A->m_add->find(new_key);
        if (it != A->m_add->end() && it->second.lock()) {  // elements were summed before
            return it->second.lock();
        }
        shared_map = A->m_add;
    } else {
        // share maps (if any) between A and B
        shared_map = std::make_shared<ov::symbol::MathMap>();
        auto set_m_add = [](const ov::symbol::WeakSymbol& s, const std::shared_ptr<ov::symbol::MathMap>& m) -> void {
            if (!s.expired())
                s.lock()->m_add = m;
        };
        share_map(shared_map, A->m_add, set_m_add);
        share_map(shared_map, B->m_add, set_m_add);
        A->m_add = shared_map;
        B->m_add = shared_map;
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
            ov::symbol::WeakSymbolVector new_key = this_element;
            new_key.insert(new_key.begin(), other_element.begin(), other_element.end());
            sort(new_key);
            shared_map->insert({new_key, C});
        }
    }
    return C;
}

ov::SymbolPtr ov::operator-(const SymbolPtr& lhs, const SymbolPtr& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return nullptr;  // should we create a new shared_ptr<Symbol> here?
    // A - B = C  =>  B + C = A
    auto A = symbol::ancestor_of(lhs);
    auto B = symbol::ancestor_of(rhs);
    const auto bw = symbol::WeakSymbol(B);

    std::shared_ptr<symbol::MathMap> shared_map = nullptr;
    if (A->m_add && B->m_add && A->m_add.get() == B->m_add.get() && !A->m_add->empty()) {  // maps are shared
        std::vector<ov::symbol::WeakSymbolVector> A_components{}, B_components{};
        collect_records_with_result(A, A->m_add, A_components);
        for (const auto& item : A_components) {
            if (item.size() == 2 && item[0] == bw && !item[1].expired()) {
                return item[1].lock();
            }
            if (item.size() == 2 && item[1] == bw && !item[0].expired()) {
                return item[0].lock();
            }
        }
        collect_records_with_result(B, B->m_add, B_components);
        for (auto A_equation : A_components) {
            for (auto B_equation : B_components) {
                remove_same_elements(A_equation, B_equation);
                if (A_equation.size() == 1 && B_equation.empty() && !A_equation[0].expired()) {
                    return A_equation[0].lock();
                }
            }
        }
        shared_map = A->m_add;
    } else {
        // share maps (if any) between A and B
        shared_map = std::make_shared<ov::symbol::MathMap>();
        auto set_m_add = [](const ov::symbol::WeakSymbol& s, const std::shared_ptr<ov::symbol::MathMap>& m) -> void {
            if (!s.expired())
                s.lock()->m_add = m;
        };
        share_map(shared_map, A->m_add, set_m_add);
        share_map(shared_map, B->m_add, set_m_add);
        A->m_add = shared_map;
        B->m_add = shared_map;
    }
    auto C = std::make_shared<ov::Symbol>();
    C->m_add = shared_map;

    std::vector<ov::symbol::WeakSymbolVector> B_components{{B}};
    collect_records_with_result(B, B->m_add, B_components);
    for (auto& new_key : B_components) {
        new_key.insert(new_key.begin(), C);
        sort(new_key);
        shared_map->insert({new_key, A});
    }
    return C;
}
