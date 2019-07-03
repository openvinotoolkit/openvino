// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>
#include <memory>
#include "mkldnn_memory.h"
#include "mkldnn_dims.h"
#include <map>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNNode;
class MKLDNNEdge;

using MKLDNNEdgePtr = std::shared_ptr<MKLDNNEdge>;
using MKLDNNEdgeWeakPtr = std::weak_ptr<MKLDNNEdge>;

class MKLDNNEdge : public InferenceEngine::details::no_copy {
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

    virtual void init();
    virtual void allocate(const void* mem_ptr = nullptr);
    virtual void validate();
    void drop();

    const std::shared_ptr<MKLDNNNode> getParent() const;
    const std::shared_ptr<MKLDNNNode> getChild() const;

    InferenceEngine::Blob::Ptr getBlob();
    InferenceEngine::TensorDesc getDesc();

    const MKLDNNDims &getDims();
    const MKLDNNMemory& getMemory();
    MKLDNNMemoryPtr& getMemoryPtr();

    bool needReorder();
    bool isDropped();

    int getInputNum();
    int getOutputNum();

    void sharedMemFrom(const MKLDNNEdgePtr& edge);
    MKLDNNEdgePtr getSharedEdge() const;

private:
    std::weak_ptr<MKLDNNNode> parent;
    std::weak_ptr<MKLDNNNode> child;
    int parent_port;
    int child_port;

    MKLDNNEdgeWeakPtr memoryFromEdge;
    MKLDNNDims dims;
    MKLDNNMemoryPtr memoryPtr;
    Status status = Status::Uninitialized;

    InferenceEngine::TensorDesc getInputDesc();
    InferenceEngine::TensorDesc getOutputDesc();
    InferenceEngine::TensorDesc getSpecifiedInputDesc(std::map<mkldnn::memory::format, size_t> formats);
    InferenceEngine::TensorDesc getSpecifiedOutputDesc(std::map<mkldnn::memory::format, size_t> formats);

    InferenceEngine::TensorDesc inputDesc;
    InferenceEngine::TensorDesc outputDesc;

    bool nodeCanChangeDesc(const std::shared_ptr<MKLDNNPlugin::MKLDNNNode>& node) const;

    enum LOOK { LOOK_UP = 1, LOOK_DOWN = 2, LOOK_BOTH = LOOK_UP | LOOK_DOWN, LOOK_NO_RECURRENT = 4 };

    MKLDNNEdgePtr getBaseEdge(int look = LOOK_BOTH);
    bool inPlace(LOOK look = LOOK_BOTH);
    friend class MKLDNNGraph;
};

}  // namespace MKLDNNPlugin
