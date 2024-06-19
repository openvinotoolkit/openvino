// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "openvino/core/core_visibility.hpp"

namespace ov {
    class Symbol;
    using WeakSymbol = std::weak_ptr<ov::Symbol>;
    using WeakSymbolVector = std::vector<std::weak_ptr<ov::Symbol>>;
}

template <>
struct std::hash<ov::WeakSymbolVector> {
    std::size_t operator()(const ov::WeakSymbolVector& v) const {
        size_t seed = 0;
        for (const auto& element : v) {
            const auto& el_hash = element.expired() ? 0 : std::hash<std::shared_ptr<ov::Symbol>>()(element.lock());
            seed ^= el_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

namespace ov {
namespace symbol {
/// \brief If both symbols are valid, sets them as equal
void OPENVINO_API set_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
/// \brief Returns true if both symbols are valid and are equal otherwise returns false
bool OPENVINO_API are_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
/// \brief Returns a representative (the most distant parent) of an equality group of this symbol
std::shared_ptr<Symbol> OPENVINO_API ancestor_of(const std::shared_ptr<Symbol>& x);
}  // namespace symbol

/// \brief Class representing unique symbol for the purpose of symbolic shape inference. Equality of symbols is being
/// tracked by Disjoint-set data structure
/// \ingroup ov_model_cpp_api
class OPENVINO_API Symbol {
public:
    /// \brief Default constructs a unique symbol
    Symbol() = default;

    std::shared_ptr<Symbol> operator+(const std::shared_ptr<Symbol>& other);
    std::shared_ptr<Symbol> operator-(const std::shared_ptr<Symbol>& other);

    ~Symbol() = default; // TODO: remove records from the maps
private:
    friend std::shared_ptr<Symbol> ov::symbol::ancestor_of(const std::shared_ptr<Symbol>& x);
    friend void ov::symbol::set_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);

    std::shared_ptr<std::unordered_map<WeakSymbolVector, WeakSymbol>> m_add;

    std::shared_ptr<Symbol> m_parent = nullptr;
};

}  // namespace ov
