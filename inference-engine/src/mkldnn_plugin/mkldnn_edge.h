// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
//#include <memory>
//#include "mkldnn_memory.h"
#include "cpu_shape.h"
#include "cpu_memory_desc.h"
#include "mkldnn_weights_cache.hpp"

//#include "mkldnn/ie_mkldnn.h"

#include <map>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

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

    inline Status getStatus() const noexcept {
        return status;
    }

    void changeStatus(Status state);

    void init();
    void allocate(const void* mem_ptr = nullptr);
    void externalAllocate(MKLDNNWeightsSharing::Ptr weightsCache);
    void reuse(MKLDNNMemoryPtr ptr);
    void validate();
    void drop();

    const std::shared_ptr<MKLDNNNode> getParent() const;
    const std::shared_ptr<MKLDNNNode> getChild() const;

    // TODO [DS]: conversion to IE::TensorDesc shouldn't be part of the Edge class
    InferenceEngine::Blob::Ptr getBlob();
    InferenceEngine::TensorDesc getTensorDesc();

    const Shape &getShape();
    const MemoryDesc& getDesc();
    const MKLDNNMemory& getMemory();
    MKLDNNMemoryPtr& getMemoryPtr();

    bool needReorder();
    bool isDropped() const;
    bool isUseExternalMemory() const;

    int getInputNum() const;
    int getOutputNum() const;

    void setChildPort(const size_t port) { child_port = port; }

    void sharedMemFrom(const MKLDNNEdgePtr& edge);
    MKLDNNEdgePtr getSharedEdge() const;
    MKLDNNEdgePtr getSharedEdge(std::nothrow_t) const;

    const MemoryDesc& getInputDescRO() const;
    const MemoryDesc& getOutputDescRO() const;

private:
    std::string name();

    std::weak_ptr<MKLDNNNode> parent;
    std::weak_ptr<MKLDNNNode> child;
    int parent_port;
    int child_port;

    bool externalMemoryPtr = false;
    MKLDNNEdgeWeakPtr memoryFromEdge;
    Shape shape;
    MKLDNNMemoryPtr memoryPtr;
    Status status = Status::Uninitialized;

    const MemoryDesc& getInputDesc();
    const MemoryDesc& getOutputDesc();

    std::unique_ptr<MemoryDesc> getSpecifiedInputDesc(std::map<mkldnn::memory::format_tag, size_t> formats,
                                                      size_t enterCountUp = 1, size_t enterCountDown = 0);
    std::unique_ptr<MemoryDesc> getSpecifiedOutputDesc(std::map<mkldnn::memory::format_tag, size_t> formats,
                                                       size_t enterCountUp = 0, size_t enterCountDown = 1);

    std::unique_ptr<MemoryDesc> inputDesc;
    std::unique_ptr<MemoryDesc> outputDesc;

    bool nodeCanChangeDesc(const std::shared_ptr<MKLDNNPlugin::MKLDNNNode>& node) const;

    enum LOOK { LOOK_UP = 1, LOOK_DOWN = 2, LOOK_BOTH = LOOK_UP | LOOK_DOWN, LOOK_NO_RECURRENT = 4 };

    MKLDNNEdgePtr getBaseEdge(int look = LOOK_BOTH);
    bool inPlace(LOOK look = LOOK_BOTH);
    friend class MKLDNNGraph;
};

}  // namespace MKLDNNPlugin
