// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace pass {
/// \brief Register openvino node pointers into container.
/// Can create and/or add existing node pointers into register
class NodeRegister {
public:
    /// \brief Make new node and add it to register.
    ///
    /// \tparam T     Node type.
    /// \tparam Args  Node ctor args types.
    ///
    /// \param args   New node ctor arguments.
    /// \return Shared pointer to node of type T.
    template <typename T, class... Args>
    std::shared_ptr<T> make(Args&&... args) {
        auto node = std::make_shared<T>(std::forward<Args>(args)...);
        return add(node);
    }

    /// \brief Add node to register
    ///
    /// \tparam T  Node type.
    ///
    /// \param node  Node to add
    ///
    /// \return Shared pointer to new node added of type T.
    template <typename T>
    std::shared_ptr<T> add(const std::shared_ptr<T>& node) {
        m_nodes.push_back(node);
        return node;
    }

    /// \brief Add generic node to register.
    ///
    /// \note This is required to prevent LTO do optimization (remove) template methods from this class
    /// on Ubuntu 20.
    ///
    /// \param node Node to add.
    ///
    /// \return Shared pointer to new generic node added.
    std::shared_ptr<Node> add(const std::shared_ptr<Node>& node) {
        m_nodes.push_back(node);
        return node;
    }

    /// \brief Get nodes container.
    ///
    /// \return Const reference to nodes container.
    const std::vector<std::shared_ptr<Node>>& get() const {
        return m_nodes;
    }

    /// \brief Clear register.
    void clear() {
        m_nodes.clear();
    }

private:
    std::vector<std::shared_ptr<Node>> m_nodes;  //!< Stores added nodes.
};
}  // namespace pass
}  // namespace ov
