// Copyright (C) 2018-2020 Intel Corporation
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
#if defined (COMPILED_CPU_MKLDNN_INPUT_NODE)
class MKLDNNMemoryInputNode;
#endif

/**
 * @brief
 * TODO: ATTENTION: this is a temporary solution, this connection should be keep in graph
 * WARNING: thread_local and holderMutex are not needed if moved into graph
 */
class MKLDNNMemoryNodeVirtualEdge {
 public:
    using Holder = std::map<std::string, MKLDNNMemoryNode*>;
    static Holder & getExisted() {
        thread_local static Holder existed;
        return existed;
    }

    static MKLDNNMemoryNode * getByName(Holder& holder, std::string name) {
        auto result = holder.find(name);
        if (result != holder.end()) {
            return result->second;
        }
        return nullptr;
    }

    static Holder* registerOutput(MKLDNNMemoryOutputNode * node);
#if defined (COMPILED_CPU_MKLDNN_INPUT_NODE)
    static Holder* registerInput(MKLDNNMemoryInputNode * node);
#endif
    static void remove(MKLDNNMemoryNode * node, Holder* holder);
    static std::mutex holderMutex;
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

 private:
    /**
     * @brief keeps reference to input sibling node
     */
    MKLDNNNode* inputNode = nullptr;
    static Registrar<MKLDNNMemoryOutputNode> reg;
    MKLDNNMemoryNodeVirtualEdge::Holder* holder = nullptr;
};

#if defined (COMPILED_CPU_MKLDNN_INPUT_NODE)
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
 private:
    MKLDNNMemoryPtr dataStore;
    static Registrar<MKLDNNMemoryInputNode> reg;
    MKLDNNMemoryNodeVirtualEdge::Holder* holder = nullptr;
};
#endif

}  // namespace MKLDNNPlugin

