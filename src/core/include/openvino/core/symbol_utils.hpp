// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <vector>

namespace ov {
class Symbol;
using SharedSymbol = std::shared_ptr<ov::Symbol>;
namespace symbol {
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

class WeakSymbolVector : public std::vector<WeakSymbol> {
public:
    WeakSymbolVector() = default;
    WeakSymbolVector(std::initializer_list<WeakSymbol> init) : std::vector<WeakSymbol>(init) {
        sort();
    };

    WeakSymbolVector(const std::vector<WeakSymbol>& init) : std::vector<WeakSymbol>(init) {
        sort();
    };

    void sort() {
        std::sort(begin(), end());
    }

    bool has(const WeakSymbol& s) const {
        return std::any_of(cbegin(), cend(), [&s](const ov::symbol::WeakSymbol& i) {
            return i == s;
        });
    }

    bool has(const ov::Symbol* s) const {
        return std::any_of(cbegin(), cend(), [&s](const ov::symbol::WeakSymbol& i) {
            if (i.lock() == nullptr)
                return s == nullptr;
            return i.lock().get() == s;
        });
    }

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
        return std::equal(begin(), end(), other.begin());
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
