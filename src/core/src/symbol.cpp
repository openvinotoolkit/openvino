// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/symbol.hpp"

#include <algorithm>
#include <map>
#include <queue>

namespace ov {
using WeakSymbol = std::weak_ptr<ov::Symbol>;
using WeakSymbolVector = std::vector<WeakSymbol>;

inline bool operator==(const WeakSymbol& lhs, const WeakSymbol& rhs) {
    if (lhs.expired() && rhs.expired())
        return true;
    if (lhs.expired() || rhs.expired())
        return false;
    return lhs.lock().get() == rhs.lock().get();
}

inline bool operator!=(const WeakSymbol& lhs, const WeakSymbol& rhs) {
    return !(lhs == rhs);
}

inline bool operator<(const WeakSymbol& lhs, const WeakSymbol& rhs) {
    return std::owner_less<ov::SymbolPtr>()(lhs.lock(), rhs.lock());
}

struct WeakSymbolVectorHash {
    std::size_t operator()(const WeakSymbolVector& v) const {
        size_t seed = 0;
        for (const auto& element : v) {
            const auto& el_hash = element.expired() ? 0 : std::hash<ov::SymbolPtr>()(element.lock());
            seed ^= el_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

using MathMap = std::map<WeakSymbolVector, WeakSymbol>;

class ov::Symbol::Impl {
public:
    Impl() = default;
    ~Impl() = default;

private:
    /*
     * Equality relations between symbols are managed by the Disjoint-set data structure.
     * m_parent field gives access to a parent tree. Parent tree grows when we use set_equal routine.
     *
     * The root of parent tree is called ancestor -- a representative of a group of equal symbols.
     * To get an ancestor of any symbol, call ancestor_of routine. If current symbol is an ancestor, then its m_parent
     * field is nullptr and ancestor_of would return current symbol.
     * */
    SymbolPtr m_parent = nullptr;

    /*
     * Shared MathMap represents mathematical relations between symbols. Rules:
     * - map stores only ancestor symbols
     * - all the Symbols in the map must share the same map as their respective field
     *
     * To ensure rules, set_equal routine and operators are managing maps of both symbols.
     *
     * MathMap m_add reflects addition and subtraction. Given symbols A, B, C and D
     * - relation A + B = C would be reflected as key={A, B} value=C in the map
     * - relation A - B = C would be reflected as key={B, C} value=A in the map
     * - relation A + B + C = D would be reflected as key={A, B, C} value=D in the map
     *
     * During destruction of a symbol, m_add map is being cleared from the records with expired symbols
     */
    std::shared_ptr<MathMap> m_add = nullptr;

public:
    const SymbolPtr& get_parent() const {
        return m_parent;
    }

    void set_parent(const SymbolPtr& new_parent) {
        m_parent = new_parent;
    }

    const std::shared_ptr<MathMap>& get_add_map() const {
        return m_add;
    }

    void set_add_map(const std::shared_ptr<MathMap>& new_map) {
        m_add = new_map;
    }
};
}  // namespace ov

ov::SymbolPtr ov::symbol::ancestor_of(const SymbolPtr& symbol) {
    auto x = symbol;
    while (const auto& parent = x->pimpl->get_parent()) {
        const auto& grand_parent = parent->pimpl->get_parent();
        if (grand_parent)
            x->pimpl->set_parent(grand_parent);
        x = x->pimpl->get_parent();
    }
    return x;
}

bool ov::symbol::are_equal(const SymbolPtr& lhs, const SymbolPtr& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return false;
    return ancestor_of(lhs).get() == ancestor_of(rhs).get();
}

namespace {  // helpers to work with shared math maps in symbols
bool contains(const ov::WeakSymbolVector& vec, const ov::WeakSymbol& s) {
    return std::find(vec.cbegin(), vec.cend(), s) != vec.cend();
}

bool contains(const ov::WeakSymbolVector& vec, const ov::Symbol* s) {
    return std::any_of(vec.begin(), vec.end(), [&s](const ov::WeakSymbol& i) {
        if (i.lock() == nullptr)
            return s == nullptr;
        return i.lock().get() == s;
    });
}

void sort(ov::WeakSymbolVector& vec) {
    std::sort(vec.begin(), vec.end());
}

void replace(ov::WeakSymbolVector& vec, const ov::WeakSymbol& old_s, const ov::WeakSymbol& new_s) {
    std::replace_if(
        vec.begin(),
        vec.end(),
        [&old_s](const ov::WeakSymbol& s) {
            return s == old_s;
        },
        new_s);
    sort(vec);
}

void replace_no_check(ov::SymbolPtr old_symbol, ov::SymbolPtr new_symbol, ov::MathMap& m) {
    ov::MathMap map_with_old;
    for (const auto& item : m)
        if (item.second == ov::WeakSymbol(old_symbol) || contains(item.first, old_symbol.get()))
            map_with_old.insert(item);
    for (const auto& item : map_with_old)
        m.erase(item.first);
    for (const auto& item : map_with_old) {
        auto new_key = item.first;
        replace(new_key, old_symbol, new_symbol);
        m.insert({new_key, item.second == old_symbol ? new_symbol : item.second});
    }
}

void remove_same_elements(ov::WeakSymbolVector& lhs, ov::WeakSymbolVector& rhs) {
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

void insert_with_check(const ov::WeakSymbolVector& key,
                       const ov::WeakSymbol& value,
                       ov::MathMap& m,
                       std::queue<std::pair<ov::WeakSymbol, ov::WeakSymbol>>& to_equalize) {
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

void replace_inplace_with_check(ov::SymbolPtr old_symbol,
                                ov::SymbolPtr new_symbol,
                                ov::MathMap& m,
                                std::queue<std::pair<ov::WeakSymbol, ov::WeakSymbol>>& to_equalize) {
    ov::MathMap with_old;
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

void replace_with_check(ov::SymbolPtr old_symbol,
                        ov::SymbolPtr new_symbol,
                        ov::MathMap& old_map,
                        ov::MathMap& new_map,
                        std::queue<std::pair<ov::WeakSymbol, ov::WeakSymbol>>& to_equalize) {
    ov::MathMap with_new;
    for (const auto& item : new_map)
        if (item.second == new_symbol || contains(item.first, new_symbol))
            with_new.insert({item.first, item.second});
    for (const auto& item : with_new)
        new_map.erase(item.first);
    // new_map contains only independent records, with_new contains records with new_symbol
    ov::MathMap with_old;
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

void collect_records_with_result(ov::SymbolPtr result,
                                 const std::shared_ptr<ov::MathMap>& map,
                                 std::vector<ov::WeakSymbolVector>& records) {
    if (map)
        for (const auto& item : *map)
            if (item.second.lock() && item.second.lock().get() == result.get())
                records.push_back(item.first);
}

void share_map(std::shared_ptr<ov::MathMap> shared_map,
               std::shared_ptr<ov::MathMap> donor_map,
               void (*set_map)(const ov::WeakSymbol&, std::shared_ptr<ov::MathMap>)) {
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
    std::queue<std::pair<ov::WeakSymbol, ov::WeakSymbol>> to_equalize;
    to_equalize.emplace(l, r);

    do {
        auto item = to_equalize.front();
        to_equalize.pop();
        auto lhs = item.first.lock(), rhs = item.second.lock();
        if (!lhs || !rhs || ov::symbol::are_equal(lhs, rhs))
            continue;  // invalid or already are equal
        auto A = ancestor_of(lhs), B = ancestor_of(rhs);
        A->pimpl->set_parent(B);

        // m_add unification
        auto A_map = A->pimpl->get_add_map();
        auto B_map = B->pimpl->get_add_map();

        if (A_map && !B_map) {
            replace_no_check(A, B, *A_map);
            B->pimpl->set_add_map(A_map);  // rhs is the root of lhs now
            A->pimpl->set_add_map(nullptr);
        } else if (A_map && B_map && A_map.get() == B_map.get()) {
            replace_inplace_with_check(A, B, *B_map, to_equalize);
            A->pimpl->set_add_map(nullptr);  // rhs is the root of lhs now
        } else if (A_map && B_map) {
            replace_with_check(A, B, *A_map, *B_map, to_equalize);
            for (auto& i : *B_map) {
                for (auto& j : i.first) {
                    if (!j.expired())
                        j.lock()->pimpl->set_add_map(B_map);
                }
                if (!i.second.expired())
                    i.second.lock()->pimpl->set_add_map(B_map);
            }
            A->pimpl->set_add_map(nullptr);
        }
    } while (!to_equalize.empty());
}

ov::Symbol::Symbol() : pimpl(new Impl()) {}

ov::Symbol::~Symbol() {
    if (pimpl->get_add_map()) {
        for (auto item = pimpl->get_add_map()->begin(), last = pimpl->get_add_map()->end(); item != last;) {
            if (contains(item->first, nullptr) || contains(item->first, this) || item->second.expired() ||
                item->second.lock().get() == this) {
                item = pimpl->get_add_map()->erase(item);
            } else {
                ++item;
            }
        }
        pimpl->set_add_map(nullptr);
    }
}

ov::SymbolPtr ov::operator+(const SymbolPtr& lhs, const SymbolPtr& rhs) {
    if (lhs == nullptr || rhs == nullptr)
        return nullptr;  // should we create a new shared_ptr to a symbol here?
    // A + B = C
    auto A = symbol::ancestor_of(lhs);
    auto B = symbol::ancestor_of(rhs);
    auto new_key = ov::WeakSymbolVector({A, B});
    sort(new_key);
    std::shared_ptr<ov::MathMap> shared_map = nullptr;
    if (A->pimpl->get_add_map() && B->pimpl->get_add_map() &&
        A->pimpl->get_add_map().get() == B->pimpl->get_add_map().get() &&
        !A->pimpl->get_add_map()->empty()) {  // maps are shared
        auto it = A->pimpl->get_add_map()->find(new_key);
        if (it != A->pimpl->get_add_map()->end() && it->second.lock()) {  // elements were summed before
            return it->second.lock();
        }
        shared_map = A->pimpl->get_add_map();
    } else {
        // share maps (if any) between A and B
        shared_map = std::make_shared<ov::MathMap>();
        auto set_m_add = [](const ov::WeakSymbol& s, std::shared_ptr<ov::MathMap> m) -> void {
            if (!s.expired())
                s.lock()->pimpl->set_add_map(m);
        };
        share_map(shared_map, A->pimpl->get_add_map(), set_m_add);
        share_map(shared_map, B->pimpl->get_add_map(), set_m_add);
        A->pimpl->set_add_map(shared_map);
        B->pimpl->set_add_map(shared_map);
    }
    auto C = std::make_shared<ov::Symbol>();
    C->pimpl->set_add_map(shared_map);

    // add L + R = result  to the shared map
    // search for X + Y = L and W + Z = R records to make X + Y + W + Z = result records
    std::vector<ov::WeakSymbolVector> this_components{{A}}, other_components{{B}};
    collect_records_with_result(A, shared_map, this_components);
    collect_records_with_result(B, shared_map, other_components);
    for (const auto& this_element : this_components) {
        for (const auto& other_element : other_components) {
            ov::WeakSymbolVector new_key = this_element;
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
    const auto bw = ov::WeakSymbol(B);

    std::shared_ptr<ov::MathMap> shared_map = nullptr;

    if (A->pimpl->get_add_map() && B->pimpl->get_add_map() &&
        A->pimpl->get_add_map().get() == B->pimpl->get_add_map().get() &&
        !A->pimpl->get_add_map()->empty()) {  // maps are shared
        std::vector<ov::WeakSymbolVector> A_components{}, B_components{};
        collect_records_with_result(A, A->pimpl->get_add_map(), A_components);
        for (const auto& item : A_components) {
            if (item.size() == 2 && item[0] == bw && !item[1].expired()) {
                return item[1].lock();
            }
            if (item.size() == 2 && item[1] == bw && !item[0].expired()) {
                return item[0].lock();
            }
        }
        collect_records_with_result(B, B->pimpl->get_add_map(), B_components);
        for (auto A_equation : A_components) {
            for (auto B_equation : B_components) {
                remove_same_elements(A_equation, B_equation);
                if (A_equation.size() == 1 && B_equation.empty() && !A_equation[0].expired()) {
                    return A_equation[0].lock();
                }
            }
        }
        shared_map = A->pimpl->get_add_map();
    } else {
        // share maps (if any) between A and B
        shared_map = std::make_shared<ov::MathMap>();
        auto set_m_add = [](const ov::WeakSymbol& s, std::shared_ptr<ov::MathMap> m) -> void {
            if (!s.expired())
                s.lock()->pimpl->set_add_map(m);
        };
        share_map(shared_map, A->pimpl->get_add_map(), set_m_add);
        share_map(shared_map, B->pimpl->get_add_map(), set_m_add);
        A->pimpl->set_add_map(shared_map);
        B->pimpl->set_add_map(shared_map);
    }
    auto C = std::make_shared<ov::Symbol>();
    C->pimpl->set_add_map(shared_map);

    std::vector<ov::WeakSymbolVector> B_components{{B}};
    collect_records_with_result(B, B->pimpl->get_add_map(), B_components);
    for (auto& new_key : B_components) {
        new_key.insert(new_key.begin(), C);
        sort(new_key);
        shared_map->insert({new_key, A});
    }
    return C;
}
