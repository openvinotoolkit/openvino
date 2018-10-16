// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

#include "mkldnn_memory.h"
#include "config.h"
#include "perf_count.h"
#include "mkldnn_dims.h"
#include "mean_image.h"
#include "mkldnn_node.h"
#include "mkldnn_edge.h"
#include "mkldnn_extension_utils.h"

namespace MKLDNNPlugin {

class MKLDNNGraph {
public:
    typedef std::shared_ptr<MKLDNNGraph> Ptr;

    enum Status {
        NotReady = 0,
        Ready = 1,
    };

    MKLDNNGraph(): status(NotReady), eng(mkldnn::engine(mkldnn::engine::kind::cpu, 0)) {}

    Status GetStatus() {
        return status;
    }

    bool IsReady() {
        return (GetStatus() == Ready);
    }

    void setConfig(const Config &cfg);
    void setProperty(const std::map<std::string, std::string> &properties);
    Config getProperty();

    void getInputBlobs(InferenceEngine::BlobMap &in_map);
    void getOutputBlobs(InferenceEngine::BlobMap &out_map);

    void CreateGraph(InferenceEngine::ICNNNetwork &network, const MKLDNNExtensionManager::Ptr& extMgr);

    bool hasMeanImageFor(const std::string& name) {
        return _meanImages.find(name) != _meanImages.end();
    }

    void PushInputData(const std::string& name, const InferenceEngine::Blob::Ptr &in);
    void PullOutputData(InferenceEngine::BlobMap &out);

    void Infer(int batch = -1);

    std::vector<MKLDNNNodePtr>& GetNodes() {
        return graphNodes;
    }

    std::vector<MKLDNNEdgePtr>& GetEdges() {
        return graphEdges;
    }

    std::vector<MKLDNNNodePtr>& GetOutputNodes() {
        return outputNodes;
    }

    mkldnn::engine getEngine() const {
        return eng;
    }

    void GetPerfData(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const;

protected:
    MKLDNNNodePtr FindNodeWithName(const std::string& name) const;
    void VisitNode(MKLDNNNodePtr node, std::vector<MKLDNNNodePtr>& sortedNodes);
    void SortTopologically();

    void ForgetGraphData() {
        status = NotReady;
        eng = mkldnn::engine(mkldnn::engine::kind::cpu, 0);

        inputNodes.clear();
        outputNodes.clear();
        graphNodes.clear();
        graphEdges.clear();
        _meanImages.clear();
    }
    Status status;
    Config config;

    MKLDNNMemoryPtr memWorkspace;

    std::map<std::string, MKLDNNNodePtr> inputNodes;
    std::vector<MKLDNNNodePtr> outputNodes;
    std::vector<MKLDNNNodePtr> graphNodes;
    std::vector<MKLDNNEdgePtr> graphEdges;

    std::map<std::string, MeanImage> _meanImages;

    mkldnn::engine eng;

    void InitNodes();
    void InitEdges();
    void Allocate();
    void AllocateWithReuse();
    void CreatePrimitives();

    friend class MKLDNNInferRequest;

private:
    struct ParsedLayer {
        MKLDNNNodePtr parent;
        InferenceEngine::CNNLayerPtr cnnLayer;
        size_t outIdx;
    };
    void ParseNode(const InferenceEngine::CNNLayerPtr& cnnLayer, MKLDNNNodePtr& parent,
                   const MKLDNNExtensionManager::Ptr& extMgr, size_t outIdx, std::vector<ParsedLayer>& layers);
};


class MKLDNNExecNetwork: public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<MKLDNNExecNetwork> Ptr;

    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override;

    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) override;

    MKLDNNExecNetwork(InferenceEngine::ICNNNetwork &network, const Config &cfg,
                      const MKLDNNExtensionManager::Ptr& extMgr);

    ~MKLDNNExecNetwork() override;

    void setProperty(const std::map<std::string, std::string> &properties);

protected:
    MKLDNNGraph::Ptr graph;
    MKLDNNExtensionManager::Ptr extensionManager;

    bool CanProcessDynBatch(InferenceEngine::ICNNNetwork &network) const;
};

}  // namespace MKLDNNPlugin
