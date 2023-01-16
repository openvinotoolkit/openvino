// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "config.h"
#include "cpu_memory.h"
#include "openvino/runtime/profiling_info.hpp"
#include "node.h"
#include "edge.h"
#include "graph_context.h"
#include "openvino/runtime/profiling_info.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/runtime/so_ptr.hpp"
#include "proxy_mem_mgr.h"

namespace ov {
namespace intel_cpu {

class SyncInferRequest;
namespace node {
class MemoryStateNode;
} // namespace node

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

    const Config & getConfig() const {
        return context->getConfig();
    }

    template<typename NET>
    void CreateGraph(NET &network, const GraphContext::CPtr ctx);

    void CreateGraph(const std::vector<NodePtr> &graphNodes,
                     const std::vector<EdgePtr> &graphEdges,
                     const GraphContext::CPtr ctx,
                     std::string name);

    void PushInputData(const std::size_t& index, const ov::SoPtr<ITensor>& input);
    void PullOutputData(std::unordered_map<std::size_t, ov::SoPtr<ITensor>>& output);

    void Infer(SyncInferRequest* request = nullptr);

    const std::vector<NodePtr>& GetNodes() const {
        return graphNodes;
    }

    std::string GetName() const {
        return _name;
    }

    const std::map<std::size_t, NodePtr>& GetInputNodesMap() const {
        return inputNodesMap;
    }

    std::map<std::size_t, NodePtr>& GetOutputNodesMap() {
        return outputNodesMap;
    }

    NodePtr getInputNodeByIndex(const std::size_t &index) {
        auto input = inputNodesMap.find(index);
        if (input == inputNodesMap.end())
            OPENVINO_THROW("CPU execution graph doesn't contain input node with index: ", index);
        return input->second;
    }

    NodePtr getOutputNodeByIndex(const std::size_t &index) {
        auto output = outputNodesMap.find(index);
        if (output == outputNodesMap.end())
            OPENVINO_THROW("CPU execution graph doesn't contain output node with index: ", index);
        return output->second;
    }

    dnnl::engine getEngine() const {
        return context->getEngine();
    }

    GraphContext::CPtr getGraphContext() const {
        return context;
    }

    void GetPerfData(std::vector<ov::ProfilingInfo> &perfMap) const;

    void CreateEdge(const NodePtr& parent,
                 const NodePtr& child,
                 int parentPort = 0,
                 int childPort = 0);
    void RemoveEdge(const EdgePtr& edge);
    void RemoveDroppedNodes();
    void RemoveDroppedEdges();
    void AddNode(NodePtr node);
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
     * supported primitive descriptors selection, etc.), which can be useful after the ResolveEdgeConflicts() completes. The second is just inserting the
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
     * supported primitive descriptors selection, etc.), which can be useful after the ResolveEdgeConflicts() completes. The second is just inserting the
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

    void SortTopologically();

    bool hasDynamicInput() const {
        return graphHasDynamicInput;
    }

    Status getStatus() const {return status;}
    const std::unordered_map<std::string, node::MemoryStateNode*>& getInternalStateNodes() const;
    void InitGraph(bool optimize = true);

protected:
    void ForgetGraphData() {
        status = Status::NotReady;

        inputNodesMap.clear();
        outputNodesMap.clear();
        graphNodes.clear();
        graphEdges.clear();
        m_executableSyncNodesInds.clear();
    }
    Status status { Status::NotReady };

    bool reuse_io_tensors = true;

    MemoryPtr memWorkspace;

    std::vector<NodePtr> graphNodes;
    std::vector<EdgePtr> graphEdges;

    std::string _name;

    bool graphHasDynamicInput = false;

    void Replicate(const std::shared_ptr<const ov::Model> &subgraph);
    void InitNodes();
    void InitDescriptors();
    void ResolveInplaceDirections();
    void InitOptimalPrimitiveDescriptors();
    void ResolveEdgeConflicts();
    void ResolveComplexInplaceConflicts();
    bool ProcessDynNodes();
    void GroupParallelNodes();
    void Allocate(const std::vector<size_t>& syncNodesInds);
    void AllocateWithReuse(const std::vector<size_t>& syncNodesInds);
    void ExecuteNode(const NodePtr& node, const dnnl::stream& stream) const;
    void CreatePrimitivesAndExecConstants() const;
    void InferStatic(SyncInferRequest* request);
    void InferDynamic(SyncInferRequest* request);
    void ParalleMtNuma(size_t num_nodes,
                       ov::threading::CPUStreamsExecutor::Ptr executor,
                       const std::function<void(size_t, size_t)>& func) const;

    friend class intel_cpu::SyncInferRequest;
    friend std::shared_ptr<ov::Model> dump_graph_as_ie_ngraph_net(const Graph &graph);

private:
    // TODO: change std::map to std::unordered_map
    std::map<std::size_t, NodePtr> inputNodesMap;
    std::map<std::size_t, NodePtr> outputNodesMap;

    std::unordered_map<std::size_t, ProxyMemoryMngrPtr> outputNodesMemMngrMap;

    // these node pointers (from graphNodes) are to avoid regular checking for
    // constantness of nodes in Infer methods and calls of
    // non-executable (optimized out) nodes, such as Input, Reshape, etc.
    std::vector<NodePtr> m_executableGraphNodes;
    std::vector<size_t> m_executableSyncNodesInds;

    GraphContext::CPtr context;

    void EnforceInferencePrecision();
    void EnforceBF16();
    void insertReorder(EdgePtr& edge, bool isOptimized, std::unordered_set<std::string>& uniqueLayerNames);
    void resolveInPlaceDirection(const NodePtr& node) const;

#ifdef CPU_DEBUG_CAPS

public:
    void setNestingLevel(const uint8_t level) { nestingLevel = level; }
    void ResetInferCount() { infer_count = 0; }

private:
    // Main CPU plugin execution graph has level 1,
    // other ones are nested graphs used for particular nodes.
    uint8_t nestingLevel = 2;
    int infer_count = 0;

    std::map<std::vector<VectorDims>, PerfKey> perfKeysMap;
    friend PerfKey perfGetKey(Graph& graph);
    friend void perfDump(const CompiledModel& execNet);
#endif // CPU_DEBUG_CAPS
};

using GraphPtr = std::shared_ptr<Graph>;

}  // namespace intel_cpu
}  // namespace ov
