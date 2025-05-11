// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "memory_state.h"

namespace ov {
namespace intel_cpu {
namespace node {

class MemoryNode {
public:
    explicit MemoryNode(std::string id) : m_id(std::move(id)) {}
    explicit MemoryNode(const std::shared_ptr<ov::Node>& op);
    virtual ~MemoryNode() = default;
    const std::string& getId() const {
        return m_id;
    }

private:
    std::string m_id;
};

class MemoryStateNode : public MemoryNode {
public:
    using MemoryNode::MemoryNode;
    virtual void assignState(MemStatePtr newState) = 0;
    virtual MemStatePtr makeState() const = 0;
};

using MmemoryStateNodePtr = std::shared_ptr<MemoryStateNode>;
using MemoryStateNodeCPtr = std::shared_ptr<const MemoryStateNode>;

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov