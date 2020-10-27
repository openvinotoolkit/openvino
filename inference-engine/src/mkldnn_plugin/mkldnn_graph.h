// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_parallel.hpp"
#include "cpp/ie_cnn_network.h"
#include "config.h"
#include "mkldnn_memory.h"
#include "mean_image.h"
#include "mkldnn_node.h"
#include "mkldnn_edge.h"
#include "threading/ie_thread_local.hpp"
#include <map>
#include <string>
#include <vector>
#include <memory>

namespace MKLDNNPlugin {

class MKLDNNGraph {
public:
    typedef std::shared_ptr<MKLDNNGraph> Ptr;
    MKLDNNWeightsSharing::Ptr weightsCache;

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

    template<typename NET>
    void CreateGraph(const NET &network,
                     const MKLDNNExtensionManager::Ptr& extMgr,
                     MKLDNNWeightsSharing::Ptr &w_cache);

    bool hasMeanImageFor(const std::string& name) {
        return _meanImages.find(name) != _meanImages.end();
    }

    void PushInputData(const std::string& name, const InferenceEngine::Blob::Ptr &in);
    void PullOutputData(InferenceEngine::BlobMap &out);

    void Infer(int batch = -1);

    std::vector<MKLDNNNodePtr>& GetNodes() {
        return graphNodes;
    }

    std::string GetName() {
        return _name;
    }

    std::vector<MKLDNNEdgePtr>& GetEdges() {
        return graphEdges;
    }

    std::vector<MKLDNNNodePtr>& GetOutputNodes() {
        return outputNodes;
    }

    std::map<std::string, MKLDNNNodePtr>& GetInputNodes() {
        return inputNodes;
    }


    mkldnn::engine getEngine() const {
        return eng;
    }

    void GetPerfData(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const;

    void RemoveDroppedNodes();
    void RemoveDroppedEdges();
    void DropNode(const MKLDNNNodePtr& node);
    void DropDWConvNode(const MKLDNNNodePtr& node);

    InferenceEngine::CNNNetwork dump() const;

    template<typename NET>
    static void ApplyUnrollPasses(NET &net);

    void ResetInferCount() { infer_count = 0; }

    void SortTopologically();

protected:
    void VisitNode(MKLDNNNodePtr node, std::vector<MKLDNNNodePtr>& sortedNodes);

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

    // For dumping purposes. -1 - no counting, all other positive
    // values mean increment it within each Infer() call
    int infer_count = -1;

    bool reuse_io_tensors = true;

    MKLDNNMemoryPtr memWorkspace;

    std::map<std::string, MKLDNNNodePtr> inputNodes;
    std::vector<MKLDNNNodePtr> outputNodes;
    std::vector<MKLDNNNodePtr> graphNodes;
    std::vector<MKLDNNEdgePtr> graphEdges;

    std::map<std::string, MeanImage> _meanImages;
    std::string _name;

    mkldnn::engine eng;

    void Replicate(const InferenceEngine::ICNNNetwork &network, const MKLDNNExtensionManager::Ptr& extMgr);
    void Replicate(const InferenceEngine::TensorIterator::Body &subgraph, const MKLDNNExtensionManager::Ptr& extMgr);
    void InitGraph();
    void InitNodes();
    void InitDescriptors();
    void InitEdges();
    void Allocate();
    void AllocateWithReuse();
    void CreatePrimitives();

    void do_before(const std::string &dir, const MKLDNNNodePtr &node);
    void do_after(const std::string &dir, const MKLDNNNodePtr &node);

    friend class MKLDNNInferRequest;
    friend class MKLDNNGraphlessInferRequest;
    friend InferenceEngine::CNNNetwork dump_graph_as_ie_net(const MKLDNNGraph &graph);
    friend InferenceEngine::CNNNetwork dump_graph_as_ie_ngraph_net(const MKLDNNGraph &graph);

private:
    void dumpToDotFile(std::string file) const;
    struct ParsedLayer {
        MKLDNNNodePtr parent;
        InferenceEngine::CNNLayerPtr cnnLayer;
        size_t outIdx;
    };
};

}  // namespace MKLDNNPlugin
