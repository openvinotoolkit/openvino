// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/symbol_utils.hpp"

namespace ov {
namespace symbol {
/// \brief If both symbols are valid, sets them as equal
void OPENVINO_API set_equal(const SharedSymbol& lhs, const SharedSymbol& rhs);
/// \brief Returns true if both symbols are valid and are equal otherwise returns false
bool OPENVINO_API are_equal(const SharedSymbol& lhs, const SharedSymbol& rhs);
/// \brief Returns a representative (the most distant parent) of an equality group of this symbol
std::shared_ptr<Symbol> OPENVINO_API ancestor_of(const SharedSymbol& x);
}  // namespace symbol

SharedSymbol OPENVINO_API operator+(const SharedSymbol& lhs, const SharedSymbol& rhs);
SharedSymbol OPENVINO_API operator-(const SharedSymbol& lhs, const SharedSymbol& rhs);

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

    SharedSymbol m_parent = nullptr;

    friend SharedSymbol ov::symbol::ancestor_of(const SharedSymbol& x);
    friend void ov::symbol::set_equal(const SharedSymbol& lhs, const SharedSymbol& rhs);

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

    friend SharedSymbol ov::operator+(const SharedSymbol& lhs, const SharedSymbol& rhs);
    friend SharedSymbol ov::operator-(const SharedSymbol& lhs, const SharedSymbol& rhs);
};

}  // namespace ov
