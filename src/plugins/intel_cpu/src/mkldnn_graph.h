// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp/ie_cnn_network.h"
#include "config.h"
#include "mkldnn_memory.h"
#include "normalize_preprocess.h"
#include "mkldnn_node.h"
#include "mkldnn_edge.h"
#include "cache/multi_cache.h"
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <atomic>

namespace MKLDNNPlugin {
class MKLDNNInferRequestBase;
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

    template<typename NET>
    void CreateGraph(NET &network,
                     const MKLDNNExtensionManager::Ptr& extMgr,
                     MKLDNNWeightsSharing::Ptr &w_cache);

    bool hasMeanImageFor(const std::string& name) {
        return _normalizePreprocMap.find(name) != _normalizePreprocMap.end();
    }

    void PushInputData(const std::string& name, const InferenceEngine::Blob::Ptr &in);
    void PullOutputData(InferenceEngine::BlobMap &out);

    void Infer(MKLDNNInferRequestBase* request = nullptr, int batch = -1);

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

    MKLDNNNodePtr getInputNodeByName(const std::string &name) {
        auto input = inputNodesMap.find(name);
        if (input == inputNodesMap.end())
            IE_THROW() << "CPU execution graph doesn't contain input node with name: " << name;
        return input->second;
    }

    MKLDNNNodePtr getOutputNodeByName(const std::string &name) {
        auto output = outputNodesMap.find(name);
        if (output == outputNodesMap.end())
            IE_THROW() << "CPU execution graph doesn't contain output node with name: " << name;
        return output->second;
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
    void RemoveEdge(MKLDNNEdgePtr& edge);
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
     * input memory descriptor
     * @param outDesc
     * output memory descriptor
     * @param isOptimized
     * optimization flag; if isOptimized is true then Reorder node does nothing
     * @param scales
     * pointer to the blob containing scales
     * @return pointer to the new Reorder node.
     */
    MKLDNNNodePtr InsertReorder(MKLDNNEdgePtr edge, std::string layerName, const MemoryDesc& inDesc,
            const MemoryDesc& outDesc, bool isOptimized = false);

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

    std::shared_ptr<ngraph::Function> dump() const;

    void ResetInferCount() { infer_count = 0; }

    void SortTopologically();

    bool isQuantized() const {
        return isQuantizedFlag;
    }

    bool hasDynamicInput() const {
        return graphHasDynamicInput;
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

    std::vector<MKLDNNNodePtr> graphNodes;
    std::vector<MKLDNNEdgePtr> graphEdges;

    std::map<std::string, NormalizePreprocess> _normalizePreprocMap;
    std::string _name;

    bool isQuantizedFlag = false;
    bool graphHasDynamicInput = false;

    static mkldnn::engine eng;

    void Replicate(const InferenceEngine::CNNNetwork &network, const MKLDNNExtensionManager::Ptr& extMgr);
    void Replicate(const std::shared_ptr<const ov::Model> &subgraph, const MKLDNNExtensionManager::Ptr& extMgr);
    void InitGraph();
    void InitNodes();
    void InitDescriptors();
    void InitOptimalPrimitiveDescriptors();
    void InitEdges();
    void Allocate();
    void AllocateWithReuse();
    void CreatePrimitives();
    void ExtractConstantAndExecutableNodes();
    void ExecuteNode(const MKLDNNNodePtr& node, const mkldnn::stream& stream) const;
    void ExecuteConstantNodesOnly() const;

    friend class MKLDNNInferRequestBase;
    friend class MKLDNNLegacyInferRequest;
    friend class MKLDNNInferRequest;
    friend std::shared_ptr<ngraph::Function> dump_graph_as_ie_ngraph_net(const MKLDNNGraph &graph);

private:
    // TODO: change std::map to std::unordered_map
    std::map<std::string, MKLDNNNodePtr> inputNodesMap;
    std::map<std::string, MKLDNNNodePtr> outputNodesMap;

    // these node pointers (from graphNodes) are to avoid regular checking for
    // constantness of nodes in ExecuteConstantNodesOnly, Infer methods and calls of
    // non-executable (optimized out) nodes, such as Input, Reshape, etc.
    std::vector<MKLDNNNodePtr> constantGraphNodes;
    std::vector<MKLDNNNodePtr> executableGraphNodes;

    MultiCachePtr rtParamsCache;

    void EnforceBF16();
};

}  // namespace MKLDNNPlugin
