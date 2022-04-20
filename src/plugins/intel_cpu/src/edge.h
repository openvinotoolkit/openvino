// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include "cpu_shape.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/node_config.h"
#include "weights_cache.hpp"

#include <map>
#include <memory>
#include <vector>

namespace ov {
namespace intel_cpu {

class Node;
class Edge;

using EdgePtr = std::shared_ptr<Edge>;
using EdgeWeakPtr = std::weak_ptr<Edge>;

class Edge {
public:
    Edge(const std::shared_ptr<Node>& parent,
         const std::shared_ptr<Node>& child,
         int pr_port = 0, int ch_port = 0);

    enum class Status {
        Uninitialized,
        NeedAllocation,
        NotAllocated,
        Allocated,
        Validated
    };

    enum class ReorderStatus {
        Regular = 0,
        Optimized = 1,
        No = 2
    };

    inline Status getStatus() const noexcept {
        return status;
    }

    void changeStatus(Status state);

    void init();
    void allocate(const void* mem_ptr = nullptr);
    void externalAllocate(WeightsSharing::Ptr weightsCache);
    void reuse(MemoryPtr ptr);
    void validate();
    void drop();

    const std::shared_ptr<Node> getParent() const;
    const std::shared_ptr<Node> getChild() const;

    const Memory& getMemory();
    MemoryPtr& getMemoryPtr();

    ReorderStatus needReorder();
    bool isDropped() const;
    bool isUseExternalMemory() const;

    int getInputNum() const;
    int getOutputNum() const;

    void setChildPort(const size_t port) { child_port = port; }

    void sharedMemFrom(const EdgePtr& edge);
    EdgePtr getSharedEdge() const;
    EdgePtr getSharedEdge(std::nothrow_t) const;

    bool hasDefinedMaxSize() const {
        return getDesc().hasDefinedMaxSize();
    }

private:
    std::string name() const;

    std::weak_ptr<Node> parent;
    std::weak_ptr<Node> child;
    int parent_port;
    int child_port;

    bool useExternalMemory = false;
    EdgeWeakPtr memoryFromEdge;
    MemoryPtr memoryPtr;
    Status status = Status::Uninitialized;

    const MemoryDesc& getInputDesc() const;
    const MemoryDesc& getOutputDesc() const;
    PortDescBaseCPtr getInputPortDesc() const;
    PortDescBaseCPtr getOutputPortDesc() const;

    const MemoryDesc& getDesc() const;
    bool enforceReorder();

    enum LOOK { LOOK_UP = 1, LOOK_DOWN = 2, LOOK_BOTH = LOOK_UP | LOOK_DOWN, LOOK_NO_RECURRENT = 4 };

    EdgePtr getBaseEdge(int look = LOOK_BOTH);
    bool inPlace(LOOK look = LOOK_BOTH);
    friend class Graph;
};

}   // namespace intel_cpu
}   // namespace ov

