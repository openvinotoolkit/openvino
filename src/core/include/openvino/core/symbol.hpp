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
using SymbolPtr = std::shared_ptr<ov::Symbol>;

namespace symbol {
using WeakSymbol = std::weak_ptr<ov::Symbol>;
using WeakSymbolVector = std::vector<WeakSymbol>;

struct WeakSymbolVectorHash {
    std::size_t operator()(const ov::symbol::WeakSymbolVector& v) const {
        size_t seed = 0;
        for (const auto& element : v) {
            const auto& el_hash = element.expired() ? 0 : std::hash<ov::SymbolPtr>()(element.lock());
            seed ^= el_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

using MathMap = std::unordered_map<WeakSymbolVector, WeakSymbol, WeakSymbolVectorHash>;

/// \brief If both symbols are valid, sets them as equal
void OPENVINO_API set_equal(const SymbolPtr& lhs, const SymbolPtr& rhs);
/// \brief Returns true if both symbols are valid and are equal otherwise returns false
bool OPENVINO_API are_equal(const SymbolPtr& lhs, const SymbolPtr& rhs);
/// \brief Returns a representative (the most distant parent) of an equality group of this symbol
std::shared_ptr<Symbol> OPENVINO_API ancestor_of(const SymbolPtr& x);
}  // namespace symbol

SymbolPtr OPENVINO_API operator+(const SymbolPtr& lhs, const SymbolPtr& rhs);
SymbolPtr OPENVINO_API operator-(const SymbolPtr& lhs, const SymbolPtr& rhs);

/// \brief Class representing unique symbol for the purpose of symbolic shape inference
/// \ingroup ov_model_cpp_api
class OPENVINO_API Symbol {
public:
    /// \brief Default constructs a unique symbol
    Symbol() = default;

    /// @brief Destructor ensures shared MathMaps are cleared from the expired symbols
    ~Symbol();

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

    friend SymbolPtr ov::symbol::ancestor_of(const SymbolPtr& x);
    friend void ov::symbol::set_equal(const SymbolPtr& lhs, const SymbolPtr& rhs);

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
    std::shared_ptr<symbol::MathMap> m_add;

    friend SymbolPtr ov::operator+(const SymbolPtr& lhs, const SymbolPtr& rhs);
    friend SymbolPtr ov::operator-(const SymbolPtr& lhs, const SymbolPtr& rhs);
};

}  // namespace ov
