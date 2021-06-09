// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp/ie_cnn_network.h"
#include "config.h"
#include "mkldnn_memory.h"
#include "normalize_preprocess.h"
#include "mkldnn_node.h"
#include "mkldnn_edge.h"
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <atomic>

namespace MKLDNNPlugin {
class MKLDNNInferRequest;
class MKLDNNGraph {
public:
    typedef std::shared_ptr<MKLDNNGraph> Ptr;
    MKLDNNWeightsSharing::Ptr weightsCache;

    enum Status {
        NotReady = 0,
        Ready = 1,
    };

    MKLDNNGraph() = default;

    Status GetStatus() {
        return status;
    }

    bool IsReady() {
        return (GetStatus() == Ready);
    }

    void setConfig(const Config &cfg);
    const Config& getConfig() const;

    void setProperty(const std::map<std::string, std::string> &properties);
    Config getProperty() const;

    void getInputBlobs(InferenceEngine::BlobMap &in_map);
    void getOutputBlobs(InferenceEngine::BlobMap &out_map);

    template<typename NET>
    void CreateGraph(NET &network,
                     const MKLDNNExtensionManager::Ptr& extMgr,
                     MKLDNNWeightsSharing::Ptr &w_cache);

    bool hasMeanImageFor(const std::string& name) {
        return _normalizePreprocMap.find(name) != _normalizePreprocMap.end();
    }

    void PushInputData(const std::string& name, const InferenceEngine::Blob::Ptr &in);
    void PullOutputData(const InferenceEngine::BlobMap &out);

    void Infer(MKLDNNInferRequest* request = nullptr, int batch = -1);

    const std::vector<MKLDNNNodePtr>& GetNodes() const {
        return graphNodes;
    }

    std::vector<MKLDNNNodePtr>& GetNodes() {
        return graphNodes;
    }

    std::string GetName() {
        return _name;
    }

    std::vector<MKLDNNEdgePtr>& GetEdges() {
        return graphEdges;
    }

    std::map<std::string, MKLDNNNodePtr>& GetInputNodesMap() {
        return inputNodesMap;
    }

    std::map<std::string, MKLDNNNodePtr>& GetOutputNodesMap() {
        return outputNodesMap;
    }

    bool hasInputWithName(const std::string& name) const {
        return inputNodesMap.count(name);
    }

    bool hasOutputWithName(const std::string& name) const {
        return outputNodesMap.count(name);
    }

    mkldnn::engine getEngine() const {
        return eng;
    }

    void GetPerfData(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const;

    void RemoveDroppedNodes();
    void RemoveDroppedEdges();
    void DropNode(const MKLDNNNodePtr& node);
    void DropDWConvNode(const MKLDNNNodePtr& node);

    /**
     * @brief Insert Reorder node at the edge-specified location.
     * The Reorder node must be inserted in case when there are inplace conflicts or the input and output tensor descriptors do not match.
     * The Reorder node rearranges the elements in memory according to inDesc and outDesc, or reinterprets memory descriptor without
     * rearrangement of elements if isOptimized is true.
     * @param edge
     * pointer to the edge in the graph where Reorder node will be inserted
     * @param layerName
     * Reorder layer name
     * @param inDesc
     * input tensor descriptor
     * @param outDesc
     * output tensor descriptor
     * @param isOptimized
     * optimization flag; if isOptimized is true then Reorder node does nothing
     * @param scales
     * pointer to the blob containing scales
     * @return pointer to the new Reorder node.
     */
    MKLDNNNodePtr InsertReorder(MKLDNNEdgePtr edge, std::string layerName, const InferenceEngine::TensorDesc& inDesc,
            const InferenceEngine::TensorDesc& outDesc, bool isOptimized = false, InferenceEngine::Blob::Ptr scales = nullptr);

    /**
     * @brief Insert MKLDNNNode at the edge-specified location.
     * This method supports two regimes. First, the node is inserted without initialization (i.e. supported descriptors initialization,
     * supported primitive descriptors selection, etc.), which can be useful after the InitEdges() completes. The second is just inserting the
     * node without initialization.
     * @param edge
     * pointer to the edge in the graph where the node will be inserted
     * @param node
     * pointer to the inserted node
     * @param initNode
     * parameter that determines whether the node needs to be initialized
     * @return true in case of success, false otherwise.
     */
    bool InsertNode(MKLDNNEdgePtr edge, MKLDNNNodePtr node, bool initNode = false);

    /**
     * @brief Insert MKLDNNNode between two specified nodes.
     * This procedure creates two edges that link the parent and child nodes to the inserted one and adds all created objects to the graph.
     * This method supports two regimes. First, the node is inserted without initialization (i.e. supported descriptors initialization,
     * supported primitive descriptors selection, etc.), which can be useful after the InitEdges() completes. The second is just inserting the
     * node without initialization.
     * @param parent
     * pointer to the parent node
     * @param child
     * pointer to the child node
     * @param parentPort
     * port number of the parent node to which the inserted node should be connected
     * @param childPort
     * port number of the child node to which the inserted node should be connected
     * @param initNode
     * parameter that determines whether the node needs to be initialized
     * @return true in case of success, false otherwise.
     */
    bool InsertNode(MKLDNNNodePtr parent, MKLDNNNodePtr child, MKLDNNNodePtr node, int parentPort, int childPort, bool initNode = false);

    InferenceEngine::CNNNetwork dump() const;

    void ResetInferCount() { infer_count = 0; }

    void SortTopologically();

    bool isQuantized() const {
        return isQuantizedFlag;
    }

protected:
    void VisitNode(MKLDNNNodePtr node, std::vector<MKLDNNNodePtr>& sortedNodes);

    void ForgetGraphData() {
        status = NotReady;
        eng = mkldnn::engine(mkldnn::engine::kind::cpu, 0);

        inputNodesMap.clear();
        outputNodesMap.clear();
        graphNodes.clear();
        graphEdges.clear();
        _normalizePreprocMap.clear();
    }
    Status status { NotReady };
    Config config;

    // For dumping purposes. -1 - no counting, all other positive
    // values mean increment it within each Infer() call
    int infer_count = -1;

    bool reuse_io_tensors = true;

    MKLDNNMemoryPtr memWorkspace;

    std::map<std::string, MKLDNNNodePtr> inputNodesMap;
    std::map<std::string, MKLDNNNodePtr> outputNodesMap;
    std::vector<MKLDNNNodePtr> graphNodes;
    std::vector<MKLDNNEdgePtr> graphEdges;

    std::map<std::string, NormalizePreprocess> _normalizePreprocMap;
    std::string _name;

    bool isQuantizedFlag = false;

    static mkldnn::engine eng;

    void Replicate(const InferenceEngine::CNNNetwork &network, const MKLDNNExtensionManager::Ptr& extMgr);
    void Replicate(const std::shared_ptr<const ngraph::Function> &subgraph, const MKLDNNExtensionManager::Ptr& extMgr);
    void InitGraph();
    void InitNodes();
    void InitDescriptors();
    void InitOptimalPrimitiveDescriptors();
    void InitEdges();
    void Allocate();
    void AllocateWithReuse();
    void CreatePrimitives();
    void ExecuteConstantNodesOnly();

    friend class MKLDNNInferRequest;
    friend class MKLDNNGraphlessInferRequest;
    friend InferenceEngine::CNNNetwork dump_graph_as_ie_ngraph_net(const MKLDNNGraph &graph);

private:
    void EnforceBF16();
};

}  // namespace MKLDNNPlugin
