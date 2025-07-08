// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <utility>

#include "memory_state.h"
#include "openvino/core/any.hpp"

namespace ov::intel_cpu::node {

class MemoryNode {
public:
    explicit MemoryNode(std::string id) : m_id(std::move(id)) {}
    explicit MemoryNode(const std::shared_ptr<ov::Node>& op);
    virtual ~MemoryNode() = default;
    [[nodiscard]] const std::string& getId() const {
        return m_id;
    }

private:
    std::string m_id;
};

class MemoryStateNode : public MemoryNode {
public:
    using MemoryNode::MemoryNode;
    virtual void assignState(MemStatePtr newState) = 0;
    [[nodiscard]] virtual MemStatePtr makeState() const = 0;
};

using MmemoryStateNodePtr = std::shared_ptr<MemoryStateNode>;
using MemoryStateNodeCPtr = std::shared_ptr<const MemoryStateNode>;

}  // namespace ov::intel_cpu::node
