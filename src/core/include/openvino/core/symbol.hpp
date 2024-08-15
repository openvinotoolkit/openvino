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
    Symbol();
    /// @brief Destructor ensures shared MathMaps are cleared from the expired symbols
    ~Symbol();

private:
    friend SymbolPtr ov::symbol::ancestor_of(const SymbolPtr& x);
    friend void ov::symbol::set_equal(const SymbolPtr& lhs, const SymbolPtr& rhs);

    friend SymbolPtr ov::operator+(const SymbolPtr& lhs, const SymbolPtr& rhs);
    friend SymbolPtr ov::operator-(const SymbolPtr& lhs, const SymbolPtr& rhs);

    class Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace ov
