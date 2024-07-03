// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ov {
class Symbol;
using SharedSymbol = std::shared_ptr<ov::Symbol>;
namespace symbol {

/// @brief Helper class to represent Symbol in a MathMap. Using weak_ptr is necessary to avoid cross-links
class WeakSymbol : public std::weak_ptr<ov::Symbol> {
public:
    WeakSymbol(const SharedSymbol& a) : std::weak_ptr<ov::Symbol>(a) {}
    bool operator==(const std::weak_ptr<ov::Symbol>& other) const {
        if (expired() && other.expired())
            return true;
        if (expired() || other.expired())
            return false;
        return lock().get() == other.lock().get();
    }
    bool operator!=(const std::weak_ptr<ov::Symbol>& other) const {
        return !(*this == other);
    }
    bool operator<(const std::weak_ptr<ov::Symbol>& rhs) const {
        return std::owner_less<>()(lock(), rhs);
    }
};

/**
 * @brief Helper class to represent key of a MathMap -- vector of weak symbols. Vector must be sorted, so that search in
 * a MathMap would be smooth
 */
class WeakSymbolVector : public std::vector<WeakSymbol> {
public:
    WeakSymbolVector() = default;
    WeakSymbolVector(std::initializer_list<WeakSymbol> init) : std::vector<WeakSymbol>(init) {
        sort();
    };
    WeakSymbolVector(const std::vector<WeakSymbol>& init) : std::vector<WeakSymbol>(init) {
        sort();
    };

    /// @brief Syntactic sugar method to sort vector of weak symbols
    void sort() {
        std::sort(begin(), end());
    }
    /// @brief Syntactic sugar method to find out if vector contains a symbol
    bool has(const WeakSymbol& s) const {
        return std::any_of(cbegin(), cend(), [&s](const ov::symbol::WeakSymbol& i) {
            return i == s;
        });
    }
    /// @brief Syntactic sugar method to find out if vector contains a symbol
    bool has(const ov::Symbol* s) const {
        return std::any_of(cbegin(), cend(), [&s](const ov::symbol::WeakSymbol& i) {
            if (i.lock() == nullptr)
                return s == nullptr;
            return i.lock().get() == s;
        });
    }
    /// @brief Syntactic sugar method to replace old_s symbols, if any, with new_s symbols, keeping the vector sorted
    void replace(const WeakSymbol& old_s, const WeakSymbol& new_s) {
        std::replace_if(
            begin(),
            end(),
            [&old_s](const WeakSymbol& s) {
                return s == old_s;
            },
            new_s);
        sort();
    }

    bool operator==(const ov::symbol::WeakSymbolVector& other) const {
        if (size() != other.size())
            return false;
        return std::equal(cbegin(), cend(), other.cbegin());
    }
    bool operator<(const ov::symbol::WeakSymbolVector& other) const {
        size_t common_size = std::min(size(), other.size());
        for (size_t i = 0; i < common_size; ++i) {
            if (at(i) < other.at(i))
                return true;
        }
        return size() < other.size();
    }
};

using MathMap = std::unordered_map<ov::symbol::WeakSymbolVector, ov::symbol::WeakSymbol>;

}  // namespace symbol
}  // namespace ov

template <>
struct std::hash<ov::symbol::WeakSymbolVector> {
    std::size_t operator()(const ov::symbol::WeakSymbolVector& v) const {
        size_t seed = 0;
        for (const auto& element : v) {
            const auto& el_hash = element.expired() ? 0 : std::hash<ov::SharedSymbol>()(element.lock());
            seed ^= el_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
