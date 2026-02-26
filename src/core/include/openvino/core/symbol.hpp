// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/core_visibility.hpp"

namespace ov {

class Symbol;

/// \brief Kind of symbol: LEAF for identity-tracked symbols, ADD/MUL for compound expressions
enum class SymbolKind { LEAF, ADD, MUL /* @todo future: SUB */ };

namespace symbol {
/// \brief If both symbols are valid leaf symbols, sets them as equal. No-op for compound symbols.
void OPENVINO_API set_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
/// \brief Returns true if both symbols are valid and are equal otherwise returns false.
/// For compound symbols, delegates to structural equality.
bool OPENVINO_API are_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
/// \brief Returns a representative (the most distant parent) of an equality group of this symbol.
/// For compound symbols, returns self.
std::shared_ptr<Symbol> OPENVINO_API ancestor_of(const std::shared_ptr<Symbol>& x);
/// \brief Creates a compound ADD symbol representing lhs + rhs.
/// Returns lhs if rhs is null, rhs if lhs is null, nullptr if both are null.
std::shared_ptr<Symbol> OPENVINO_API add(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
/// \brief Creates a compound MUL symbol representing lhs * rhs.
/// Returns lhs if rhs is null, rhs if lhs is null, nullptr if both are null.
std::shared_ptr<Symbol> OPENVINO_API mul(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
/// \brief Checks structural equality of two symbols, handling commutativity for ADD and MUL.
bool OPENVINO_API structurally_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
}  // namespace symbol

/// \brief Class representing unique symbol for the purpose of symbolic shape inference. Equality of symbols is being
/// tracked by Disjoint-set data structure. Compound symbols (e.g. ADD) represent expression trees.
/// \ingroup ov_model_cpp_api
class OPENVINO_API Symbol {
public:
    /// \brief Default constructs a unique leaf symbol
    Symbol() = default;

    /// \brief Returns the kind of this symbol (LEAF or compound)
    SymbolKind get_kind() const;
    /// \brief Returns true if this is a leaf symbol
    bool is_leaf() const;
    /// \brief Returns true if this is a compound symbol (not a leaf)
    bool is_compound() const;
    /// \brief Returns the left-hand operand of a compound symbol (nullptr for leaf)
    const std::shared_ptr<Symbol>& get_lhs() const;
    /// \brief Returns the right-hand operand of a compound symbol (nullptr for leaf)
    const std::shared_ptr<Symbol>& get_rhs() const;

private:
    friend std::shared_ptr<Symbol> ov::symbol::ancestor_of(const std::shared_ptr<Symbol>& x);
    friend void ov::symbol::set_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
    friend std::shared_ptr<Symbol> ov::symbol::add(const std::shared_ptr<Symbol>& lhs,
                                                   const std::shared_ptr<Symbol>& rhs);
    friend std::shared_ptr<Symbol> ov::symbol::mul(const std::shared_ptr<Symbol>& lhs,
                                                   const std::shared_ptr<Symbol>& rhs);

    SymbolKind m_kind = SymbolKind::LEAF;
    std::shared_ptr<Symbol> m_parent = nullptr;  // union-find parent (LEAF only)
    std::shared_ptr<Symbol> m_lhs = nullptr;     // left operand (compound only)
    std::shared_ptr<Symbol> m_rhs = nullptr;     // right operand (compound only)
};

}  // namespace ov
