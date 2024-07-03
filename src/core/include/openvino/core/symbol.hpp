// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

/// \brief Class representing unique symbol for the purpose of symbolic shape inference. Equality of symbols is being
/// tracked by Disjoint-set data structure
/// \ingroup ov_model_cpp_api
class OPENVINO_API Symbol {
public:
    /// \brief Default constructs a unique symbol
    Symbol() = default;
    ~Symbol();

private:
    friend SharedSymbol ov::symbol::ancestor_of(const SharedSymbol& x);
    friend void ov::symbol::set_equal(const SharedSymbol& lhs, const SharedSymbol& rhs);

    friend SharedSymbol ov::operator+(const SharedSymbol& lhs, const SharedSymbol& rhs);
    friend SharedSymbol ov::operator-(const SharedSymbol& lhs, const SharedSymbol& rhs);

    std::shared_ptr<symbol::MathMap> m_add;
    SharedSymbol m_parent = nullptr;
};

}  // namespace ov
