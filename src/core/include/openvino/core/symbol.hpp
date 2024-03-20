// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/core_visibility.hpp"

namespace ov {
/// \brief Class representing unique symbol for the purpose of symbolic shape inference. Equality of symbols is being
/// tracked by Disjoint-set data structure
/// \ingroup ov_model_cpp_api
class OPENVINO_API Symbol : public std::enable_shared_from_this<Symbol> {
public:
    /// \brief Default constructs a unique symbol
    Symbol() = default;
    // TODO
    Symbol(Symbol& t);
    /// \brief Records equality of this and other symbol
    void set_equal(const std::shared_ptr<Symbol>& other);
    /// \brief Returns true if this and other symbol are equal
    bool is_equal_to(const std::shared_ptr<Symbol>& other);
    /// \brief Returns root parent of current symbol
    std::shared_ptr<Symbol> root();
    /// \brief Returns true if both symbols are valid and are equal otherwise returns false
    static bool are_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
    static bool set_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
    // friend bool operator==(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs) { return
    // are_equal(lhs, rhs); }
private:
    /// \brief Returns rank of current symbol
    size_t rank();
    std::shared_ptr<Symbol> get_parent();

private:
    std::shared_ptr<Symbol> parent = nullptr;
};

}  // namespace ov