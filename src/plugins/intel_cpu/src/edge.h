// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "cpu_shape.h"
#include "internal_properties.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/node_config.h"
#include "weights_cache.hpp"

namespace ov {
namespace intel_cpu {

class Node;
class Edge;

using EdgePtr = std::shared_ptr<Edge>;
using EdgeWeakPtr = std::weak_ptr<Edge>;

class Edge {
public:
    Edge(const std::shared_ptr<Node>& parent, const std::shared_ptr<Node>& child, int pr_port = 0, int ch_port = 0);

    enum class Status {
        Uninitialized,   // base edge is unknown yet
        NeedAllocation,  // edge is the base edge
        NotAllocated,    // edge references another edge
        Allocated,       // edge memory is allocated
        Validated        // edge is validated
    };

    enum class ReorderStatus { Regular = 0, Optimized = 1, No = 2 };

    enum LOOK { LOOK_UP = 1, LOOK_DOWN = 2, LOOK_BOTH = LOOK_UP | LOOK_DOWN };

    inline Status getStatus() const noexcept {
        return status;
    }

    static std::string statusToString(Status status) {
#define CASE(_status)     \
    case Status::_status: \
        return #_status;
        switch (status) {
            CASE(Uninitialized);
            CASE(NeedAllocation);
            CASE(NotAllocated);
            CASE(Allocated);
            CASE(Validated);
        }
#undef CASE
        return "Unexpected";
    }

    void changeStatus(Status state);
    bool inPlace(LOOK look = LOOK_BOTH) const;

    void init();
    void allocate(const void* mem_ptr = nullptr);
    void allocate(MemoryBlockPtr memBlock);
    void externalAllocate(const WeightsSharing::Ptr& weightsCache);
    void reuse(MemoryPtr ptr);
    void validate();

    const std::shared_ptr<Node> getParent() const;
    const std::shared_ptr<Node> getChild() const;

    const IMemory& getMemory();
    MemoryPtr getMemoryPtr() const;

    ReorderStatus needReorder();
    std::shared_ptr<Node> modifiedInPlace() const;
    bool isDropped() const;
    bool isUseExternalMemory() const;

    int getInputNum() const;
    int getOutputNum() const;

    void setChildPort(const size_t port) {
        child_port = port;
    }

    void sharedMemFrom(const EdgePtr& edge);
    EdgePtr getSharedEdge() const;
    EdgePtr getSharedEdge(std::nothrow_t) const;

    bool hasDefinedMaxSize() const {
        return getOriginalDesc().hasDefinedMaxSize();
    }

    std::string hash() const;
    const MemoryDesc& getOriginalDesc() const;

private:
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

    bool enforceReorder();

    void collectConsumers(std::vector<std::shared_ptr<Node>>& result) const;

    EdgePtr getBaseEdge(int look = LOOK_BOTH);
    void allocateCommon(const std::function<MemoryPtr(const MemoryDesc&)>& allocate);

    friend class Graph;
};

std::ostream& operator<<(std::ostream& os, const Edge& edge);

}  // namespace intel_cpu
}  // namespace ov
