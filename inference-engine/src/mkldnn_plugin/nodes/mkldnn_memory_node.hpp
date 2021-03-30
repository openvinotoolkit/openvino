// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include "ie_algorithm.hpp"
#include "mkldnn_input_node.h"
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <map>

namespace MKLDNNPlugin {

class MKLDNNMemoryOutputNode;
class MKLDNNMemoryInputNode;
class MKLDNNMemoryNode {
    std::string _id;
 public:
    explicit MKLDNNMemoryNode(std::string id) : _id(id) {}
    explicit MKLDNNMemoryNode(InferenceEngine::CNNLayerPtr lp) {
        if (lp->params.find("id") != lp->params.end()) {
            _id = lp->GetParamAsString("id");
        }
    }
    virtual ~MKLDNNMemoryNode() = default;
    std::string getId() {
        return _id;
    }
    virtual void setInputNode(MKLDNNNode *) = 0;
};

class MKLDNNMemoryOutputNode : public MKLDNNNode, public MKLDNNMemoryNode {
 public:
    MKLDNNMemoryOutputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNMemoryOutputNode() override;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {}
    void execute(mkldnn::stream strm) override;
    bool created() const override {
        return getType() == MemoryOutput;
    }

    void setInputNode(MKLDNNNode* node) override {
        inputNode = node;
    }
    void init() override;

 private:
    /**
     * @brief keeps reference to input sibling node
     */
    MKLDNNNode* inputNode = nullptr;
};

class MKLDNNMemoryInputNode : public MKLDNNInputNode, public MKLDNNMemoryNode {
public:
    MKLDNNMemoryInputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNMemoryInputNode() override;

    bool created() const override {
        return getType() == MemoryInput;
    }
    void execute(mkldnn::stream strm) override;

    void createPrimitive() override;

    void setInputNode(MKLDNNNode* node) override {}
    void storeState(const MKLDNNMemory& mem);
    MKLDNNMemoryPtr getStore();
    void init() override;
 private:
    MKLDNNMemoryPtr dataStore;
};

}  // namespace MKLDNNPlugin

