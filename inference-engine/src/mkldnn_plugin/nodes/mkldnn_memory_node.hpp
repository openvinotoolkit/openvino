// Copyright (C) 2018-2019 Intel Corporation
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
class MKLDNNMemoryOutputNode;
class MKLDNNMemoryInputNode;

/**
 * @brief
 * TODO: ATTENTION: this is a temporary solution, this connection should be keep in graph
 */
class MKLDNNMemoryNodeVirtualEdge {
    using Holder = std::map<std::string, MKLDNNMemoryNode*>;
    static Holder & getExisted() {
        static Holder existed;
        return existed;
    }

    static MKLDNNMemoryNode * getByName(std::string name) {
        auto result = getExisted().find(name);
        if (result != getExisted().end()) {
            return result->second;
        }
        return nullptr;
    }

 public:
    static void registerOutput(MKLDNNMemoryOutputNode * node);
    static void registerInput(MKLDNNMemoryInputNode * node);
    static void remove(MKLDNNMemoryNode * node) {
        InferenceEngine::details::erase_if(getExisted(), [&](const Holder::value_type & it){
            return it.second == node;
        });
    }
};

class MKLDNNMemoryOutputNode : public MKLDNNNode, public MKLDNNMemoryNode {
 public:
    MKLDNNMemoryOutputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket);
    ~MKLDNNMemoryOutputNode() override;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    const MKLDNNEdgePtr getChildEdgeAt(size_t idx) const override;
    void createPrimitive() override {}
    void execute(mkldnn::stream strm) override;
    bool created() const override {
        return getType() == MemoryOutput;
    }

    void setInputNode(MKLDNNNode* node) override {
        inputNode = node;
    }
 private:
    /**
     * @brief keeps reference to input sibling node
     */
    MKLDNNNode* inputNode = nullptr;
    static Register<MKLDNNMemoryOutputNode> reg;
};

class MKLDNNMemoryInputNode : public MKLDNNInputNode, public MKLDNNMemoryNode {
public:
    MKLDNNMemoryInputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket);
    ~MKLDNNMemoryInputNode() override;

    bool created() const override {
        return getType() == MemoryInput;
    }

    void setInputNode(MKLDNNNode* node) override {}
 private:
    static Register<MKLDNNMemoryInputNode> reg;
};

}  // namespace MKLDNNPlugin

