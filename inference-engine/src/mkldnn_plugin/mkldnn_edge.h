// Copyright (C) 2018 Intel Corporation
//
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
    enum class Status {
        Uninitialized,
        NeedAllocation,
        NotAllocated,
        Allocated,
        Validated
    };
    MKLDNNEdge(const std::shared_ptr<MKLDNNNode>& parent, const std::shared_ptr<MKLDNNNode>& child);

    inline Status getStatus() noexcept {
        return status;
    }

    void changeStatus(Status state);

    virtual void init();
    virtual void allocate(const void* mem_ptr = nullptr);
    virtual void validate();

    const std::shared_ptr<MKLDNNNode> getParent() const;
    const std::shared_ptr<MKLDNNNode> getChild() const;

    bool needReorder();

    InferenceEngine::Blob::Ptr getBlob();
    const MKLDNNMemory& getMemory();
    MKLDNNMemoryPtr& getMemoryPtr();

    bool isDropped();

    InferenceEngine::TensorDesc getDesc();
    int getInputNum();
    int getOutputNum();
    std::vector<int> getAllOutputNums();
    std::vector<int> getAllInputNums();

    MKLDNNDims &getDims();
    void setDims(MKLDNNDims &dims);

    void sharedMemFrom(const MKLDNNEdgePtr& edge);
    MKLDNNEdgePtr getSharedEdge() const;

private:
    std::weak_ptr<MKLDNNNode> parent;
    std::weak_ptr<MKLDNNNode> child;
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

    enum LOOK { LOOK_UP = 1, LOOK_DOWN = 2, LOOK_BOTH = LOOK_UP | LOOK_DOWN };

    MKLDNNEdgePtr getBaseEdge(LOOK look = LOOK_BOTH);
    bool inPlace(LOOK look = LOOK_BOTH);
    friend class MKLDNNGraph;
};

}  // namespace MKLDNNPlugin
