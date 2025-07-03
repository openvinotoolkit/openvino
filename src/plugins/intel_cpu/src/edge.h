// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <new>
#include <ostream>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/node_config.h"
#include "weights_cache.hpp"

namespace ov::intel_cpu {

class Node;
class Edge;

using EdgePtr = std::shared_ptr<Edge>;
using EdgeWeakPtr = std::weak_ptr<Edge>;

class Edge {
public:
    Edge(const std::shared_ptr<Node>& parent, const std::shared_ptr<Node>& child, int pr_port = 0, int ch_port = 0);

    enum class Status : uint8_t {
        Uninitialized,   // base edge is unknown yet
        NeedAllocation,  // edge is the base edge
        NotAllocated,    // edge references another edge
        Allocated,       // edge memory is allocated
        Validated        // edge is validated
    };

    enum class ReorderStatus : uint8_t { Regular = 0, Optimized = 1, No = 2 };

    enum LOOK : uint8_t { LOOK_UP = 1, LOOK_DOWN = 2, LOOK_BOTH = LOOK_UP | LOOK_DOWN };

    [[nodiscard]] Status getStatus() const noexcept {
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
    [[nodiscard]] bool inPlace(LOOK look = LOOK_BOTH) const;

    void init();
    void allocate(const void* mem_ptr = nullptr);
    void allocate(MemoryBlockPtr memBlock);
    void externalAllocate(const WeightsSharing::Ptr& weightsCache);
    void reuse(MemoryPtr ptr);
    void validate();

    [[nodiscard]] std::shared_ptr<Node> getParent() const;
    [[nodiscard]] std::shared_ptr<Node> getChild() const;

    const IMemory& getMemory();
    [[nodiscard]] MemoryPtr getMemoryPtr() const;

    ReorderStatus needReorder();
    [[nodiscard]] std::shared_ptr<Node> modifiedInPlace() const;
    [[nodiscard]] bool isDropped() const;
    [[nodiscard]] bool isUseExternalMemory() const;

    [[nodiscard]] int getInputNum() const;
    [[nodiscard]] int getOutputNum() const;

    void setChildPort(const size_t port) {
        child_port = port;
    }

    void sharedMemFrom(const EdgePtr& edge);
    [[nodiscard]] EdgePtr getSharedEdge() const;
    [[nodiscard]] EdgePtr getSharedEdge(std::nothrow_t nothrow_tag) const;

    [[nodiscard]] bool hasDefinedMaxSize() const {
        return getOriginalDesc().hasDefinedMaxSize();
    }

    [[nodiscard]] std::string hash() const;
    [[nodiscard]] const MemoryDesc& getOriginalDesc() const;

private:
    std::weak_ptr<Node> parent;
    std::weak_ptr<Node> child;
    int parent_port;
    int child_port;

    bool useExternalMemory = false;
    EdgeWeakPtr memoryFromEdge;
    MemoryPtr memoryPtr;
    Status status = Status::Uninitialized;

    [[nodiscard]] const MemoryDesc& getInputDesc() const;
    [[nodiscard]] const MemoryDesc& getOutputDesc() const;
    [[nodiscard]] PortDescBaseCPtr getInputPortDesc() const;
    [[nodiscard]] PortDescBaseCPtr getOutputPortDesc() const;

    bool enforceReorder();

    void collectConsumers(std::vector<std::shared_ptr<Node>>& result) const;

    EdgePtr getBaseEdge(int look = LOOK_BOTH);
    void allocateCommon(const std::function<MemoryPtr(const MemoryDesc&)>& allocate);

    friend class Graph;
};

std::ostream& operator<<(std::ostream& os, const Edge& edge);

}  // namespace ov::intel_cpu
