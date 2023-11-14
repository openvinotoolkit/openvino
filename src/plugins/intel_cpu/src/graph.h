// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/multi_cache.h"
#include "config.h"
#include "cpu_memory.h"
#include "dnnl_scratch_pad.h"
#include "edge.h"
#include "graph_context.h"
#include "node.h"
#include "normalize_preprocess.h"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/profiling_info.hpp"

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "proxy_mem_mgr.h"

namespace ov {
namespace intel_cpu {

class SyncInferRequest;

class Graph {
public:
    typedef std::shared_ptr<Graph> Ptr;

    enum class Status {
        NotReady = 0,
        ReadyStatic = 1,
        ReadyDynamic = 2
    };

    Graph() = default;
    ~Graph();

    bool IsReady() {
        return (status != Status::NotReady);
    }

    void IsSubgraphOf(const Node* node) {
        parent_node = node;

        if (parent_node->getType() == Type::If)
            reuse_io_tensors = false;
    }

    const Config & getConfig() const {
        return context->getConfig();
    }

    template<typename NET>
    void CreateGraph(NET &network, const GraphContext::CPtr ctx);

    void CreateGraph(const std::vector<NodePtr> &graphNodes,
                     const std::vector<EdgePtr> &graphEdges,
                     const GraphContext::CPtr ctx,
                     std::string name);

    bool hasMeanImageFor(const std::string& name) {
        return _normalizePreprocMap.find(name) != _normalizePreprocMap.end();
    }

    void PushInputData(const std::string& name, const ov::SoPtr<ITensor>& input);
    void PullOutputData(std::unordered_map<std::string, ov::SoPtr<ITensor>>& output);

    void Infer(SyncInferRequest* request = nullptr);

    const std::vector<NodePtr>& GetNodes() const {
        return graphNodes;
    }

    std::vector<NodePtr>& GetNodes() {
        return graphNodes;
    }

    std::string GetName() const {
        return _name;
    }

    std::vector<EdgePtr>& GetEdges() {
        return graphEdges;
    }

    std::map<std::string, NodePtr>& GetInputNodesMap() {
        return inputNodesMap;
    }

    std::map<std::string, NodePtr>& GetOutputNodesMap() {
        return outputNodesMap;
    }

    NodePtr getInputNodeByName(const std::string &name) {
        auto input = inputNodesMap.find(name);
        if (input == inputNodesMap.end())
            OPENVINO_THROW("CPU execution graph doesn't contain input node with name: ", name);
        return input->second;
    }

    NodePtr getOutputNodeByName(const std::string &name) {
        auto output = outputNodesMap.find(name);
        if (output == outputNodesMap.end())
            OPENVINO_THROW("CPU execution graph doesn't contain output node with name: ", name);
        return output->second;
    }

    bool hasOutputWithName(const std::string& name) const {
        return outputNodesMap.count(name);
    }

    dnnl::engine getEngine() const {
        return context->getEngine();
    }

    GraphContext::CPtr getGraphContext() const {
        return context;
    }

    void GetPerfData(std::vector<ov::ProfilingInfo> &perfMap) const;

    void RemoveDroppedNodes();
    void RemoveDroppedEdges();
    void RemoveEdge(EdgePtr& edge);
    void DropNode(const NodePtr& node);
    void DropDWConvNode(const NodePtr& node);

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
     * @param src_perm
     * optimization flag; permutation applied to input desc before passing to reorder primitive
     * @param scales
     * pointer to the blob containing scales
     * @return pointer to the new Reorder node.
     */
    NodePtr InsertReorder(EdgePtr edge, std::string layerName, const MemoryDesc& inDesc,
            const MemoryDesc& outDesc, bool isOptimized = false, const std::vector<int> & src_perm = {});

    /**
     * @brief Insert Node at the edge-specified location.
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
    bool InsertNode(EdgePtr edge, NodePtr node, bool initNode = false);

    /**
     * @brief Insert Node between two specified nodes.
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
    bool InsertNode(NodePtr parent, NodePtr child, NodePtr node, int parentPort, int childPort, bool initNode = false);

    std::shared_ptr<ov::Model> dump() const;

    void ResetInferCount() { infer_count = 0; }

    void SortTopologically();

    bool hasDynamicInput() const {
        return graphHasDynamicInput;
    }

    Status getStatus() const {return status;}

    std::unordered_map<std::string, ProxyMemoryMngrPtr> inputNodesMemMngrMap;

protected:
    void VisitNode(NodePtr node, std::vector<NodePtr>& sortedNodes);

    void ForgetGraphData() {
        status = Status::NotReady;

        inputNodesMap.clear();
        outputNodesMap.clear();
        graphNodes.clear();
        graphEdges.clear();
        _normalizePreprocMap.clear();
        syncNodesInds.clear();
    }
    Status status { Status::NotReady };

    // For dumping purposes. -1 - no counting, all other positive
    // values mean increment it within each Infer() call
    int infer_count = -1;

    bool reuse_io_tensors = true;

    MemoryPtr memWorkspace;

    std::vector<NodePtr> graphNodes;
    std::vector<EdgePtr> graphEdges;

    std::map<std::string, NormalizePreprocess> _normalizePreprocMap;
    std::string _name;

    bool graphHasDynamicInput = false;

    void Replicate(const std::shared_ptr<const ov::Model> &subgraph);
    void InitGraph();
    void InitNodes();
    void InitDescriptors();
    void ResolveInplaceDirections();
    void InitOptimalPrimitiveDescriptors();
    void InitEdges();
    bool ProcessDynNodes();
    void Allocate();
    void AllocateWithReuse();
    void ExtractExecutableNodes();
    void ExecuteNode(const NodePtr& node, const dnnl::stream& stream) const;
    void CreatePrimitivesAndExecConstants() const;
    void InferStatic(SyncInferRequest* request);
    void InferDynamic(SyncInferRequest* request);

    friend class intel_cpu::SyncInferRequest;
    friend std::shared_ptr<ov::Model> dump_graph_as_ie_ngraph_net(const Graph &graph);

private:
    // TODO: change std::map to std::unordered_map
    std::map<std::string, NodePtr> inputNodesMap;
    std::map<std::string, NodePtr> outputNodesMap;

    std::unordered_map<std::string, ProxyMemoryMngrPtr> outputNodesMemMngrMap;
    // std::unordered_map<std::string, ProxyMemoryMngrPtr> inputNodesMemMngrMap;

    // these node pointers (from graphNodes) are to avoid regular checking for
    // constantness of nodes in Infer methods and calls of
    // non-executable (optimized out) nodes, such as Input, Reshape, etc.
    std::vector<NodePtr> executableGraphNodes;

    std::unordered_map<Node*, size_t> syncNodesInds;

    GraphContext::CPtr context;

    const Node* parent_node = nullptr;

    void EnforceInferencePrecision();
    void EnforceBF16();
    void resolveInPlaceDirection(const NodePtr& node) const;
};

}  // namespace intel_cpu
}  // namespace ov
