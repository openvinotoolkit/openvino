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

class MKLDNNNode;
class MKLDNNEdge;

using MKLDNNEdgePtr = std::shared_ptr<MKLDNNEdge>;
using MKLDNNEdgeWeakPtr = std::weak_ptr<MKLDNNEdge>;

class MKLDNNEdge {
public:
    MKLDNNEdge(const std::shared_ptr<MKLDNNNode>& parent,
               const std::shared_ptr<MKLDNNNode>& child,
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
    void allocate(DnnlMemoryMngrPtr memMngr);
    void externalAllocate(MKLDNNWeightsSharing::Ptr weightsCache);
    void reuse(MKLDNNMemoryPtr ptr);
    void validate();
    void drop();

    const std::shared_ptr<MKLDNNNode> getParent() const;
    const std::shared_ptr<MKLDNNNode> getChild() const;

    const MKLDNNMemory& getMemory();
    MKLDNNMemoryPtr& getMemoryPtr();

    ReorderStatus needReorder();
    bool isDropped() const;
    bool isUseExternalMemory() const;

    int getInputNum() const;
    int getOutputNum() const;

    void setChildPort(const size_t port) { child_port = port; }

    void sharedMemFrom(const MKLDNNEdgePtr& edge);
    MKLDNNEdgePtr getSharedEdge() const;
    MKLDNNEdgePtr getSharedEdge(std::nothrow_t) const;

    bool hasDefinedMaxSize() const {
        return getDesc().hasDefinedMaxSize();
    }

private:
    std::string name() const;

    std::weak_ptr<MKLDNNNode> parent;
    std::weak_ptr<MKLDNNNode> child;
    int parent_port;
    int child_port;

    bool useExternalMemory = false;
    MKLDNNEdgeWeakPtr memoryFromEdge;
    MKLDNNMemoryPtr memoryPtr;
    Status status = Status::Uninitialized;

    const MemoryDesc& getInputDesc() const;
    const MemoryDesc& getOutputDesc() const;
    PortDescBaseCPtr getInputPortDesc() const;
    PortDescBaseCPtr getOutputPortDesc() const;

    const MemoryDesc& getDesc() const;
    bool enforceReorder();

    enum LOOK { LOOK_UP = 1, LOOK_DOWN = 2, LOOK_BOTH = LOOK_UP | LOOK_DOWN, LOOK_NO_RECURRENT = 4 };

    MKLDNNEdgePtr getBaseEdge(int look = LOOK_BOTH);
    bool inPlace(LOOK look = LOOK_BOTH);
    friend class MKLDNNGraph;
    void allocateCommon(const std::function<void(const MKLDNNMemoryPtr&, const MemoryDesc&)>& allocate);
};

}   // namespace intel_cpu
}   // namespace ov

