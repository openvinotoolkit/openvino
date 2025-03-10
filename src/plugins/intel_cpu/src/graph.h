// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "allocation_context.hpp"
#include "config.h"
#include "cpu_memory.h"
#include "edge.h"
#include "graph_context.h"
#include "memory_control.hpp"
#include "memory_state.h"
#include "node.h"
#include "nodes/input.h"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "proxy_mem_blk.h"

namespace ov {
namespace intel_cpu {

class SyncInferRequest;
namespace node {
class MemoryStateNode;
}  // namespace node

class Graph {
public:
    using Ptr = std::shared_ptr<Graph>;
    using OutputMemoryBlocks = std::unordered_map<std::size_t, ProxyMemoryBlockPtr>;

    enum class Status {
        NotReady = 0,
        Initialized = 1,
        ReadyStatic = 2,
        ReadyDynamic = 3,
        ReadyDynamicSeq = 4,
    };

    Graph() = default;
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;

    ~Graph();

    bool IsStatic() const {
        return Status::ReadyStatic == status;
    }

    bool IsDynamic() const {
        return one_of(status, Status::ReadyDynamic, Status::ReadyDynamicSeq);
    }

    bool IsReady() const {
        return IsStatic() || IsDynamic();
    }

    const Config& getConfig() const {
        return m_context->getConfig();
    }

    /**
     * Obsolete way of creating graph
     * To enable layout propagation and global memory reuse
     * two-stage creation should be used instead:
     * - Init()
     * - Activate()
     */
    template <typename NET>
    void CreateGraph(NET& model, const GraphContext::CPtr& context);

    /**
     * Obsolete way of creating graph
     * To enable layout propagation and global memory reuse
     * two-stage creation should be used instead:
     * - Init()
     * - Activate()
     */
    void CreateGraph(const std::vector<NodePtr>& graphNodes,
                     const std::vector<EdgePtr>& graphEdges,
                     const GraphContext::CPtr& context,
                     std::string name);

    void PushInputData(const std::size_t& index, const ov::SoPtr<ITensor>& input);
    void PullOutputData(std::unordered_map<std::size_t, ov::SoPtr<ITensor>>& output);

    // Returns Output nodes memory descriptors
    VecMemoryDescs getOutputMemoryDescriptors() const;

    void Infer(SyncInferRequest* request = nullptr);

    const std::vector<NodePtr>& GetNodes() const {
        return graphNodes;
    }

    std::string GetName() const {
        return _name;
    }

    NodePtr getInputNodeByIndex(std::size_t index) {
        auto input = inputNodesMap.find(index);
        if (input == inputNodesMap.end())
            return nullptr;
        return input->second;
    }

    NodePtr getOutputNodeByIndex(std::size_t index) {
        auto output = outputNodesMap.find(index);
        if (output == outputNodesMap.end())
            return nullptr;
        return output->second;
    }

    NodeConstPtr getInputNodeByIndex(std::size_t index) const {
        auto input = inputNodesMap.find(index);
        if (input == inputNodesMap.end())
            return nullptr;
        return input->second;
    }

    NodeConstPtr getOutputNodeByIndex(std::size_t index) const {
        auto output = outputNodesMap.find(index);
        if (output == outputNodesMap.end())
            return nullptr;
        return output->second;
    }

    size_t inputsNumber() const {
        return inputNodesMap.size();
    }

    size_t outputsNumber() const {
        return outputNodesMap.size();
    }

    dnnl::engine getEngine() const {
        return m_context->getEngine();
    }

    GraphContext::CPtr getGraphContext() const {
        return m_context;
    }

    std::vector<MemStatePtr> memoryStates() const;
    void assignStates(const std::vector<MemStatePtr>& state);

    void GetPerfData(std::vector<ov::ProfilingInfo>& perfMap) const;

    void CreateEdge(const NodePtr& parent, const NodePtr& child, int parentPort = 0, int childPort = 0);
    void RemoveEdge(const EdgePtr& edge);
    void RemoveDroppedNodes();
    void RemoveDroppedEdges();
    void AddNode(const NodePtr& node);
    void DropNode(const NodePtr& node);
    void DropDWConvNode(const NodePtr& node);

    /**
     * @brief Insert Reorder node at the edge-specified location.
     * The Reorder node must be inserted in case when there are inplace conflicts or the input and output tensor
     * descriptors do not match. The Reorder node rearranges the elements in memory according to inDesc and outDesc, or
     * reinterprets memory descriptor without rearrangement of elements if isOptimized is true.
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
    NodePtr InsertReorder(const EdgePtr& edge,
                          const std::string& layerName,
                          const MemoryDesc& inDesc,
                          const MemoryDesc& outDesc,
                          bool isOptimized = false,
                          const std::vector<int>& src_perm = {});

    /**
     * @brief Insert Node at the edge-specified location.
     * This method supports two regimes. First, the node is inserted without initialization (i.e. supported descriptors
     * initialization, supported primitive descriptors selection, etc.), which can be useful after the
     * ResolveEdgeConflicts() completes. The second is just inserting the node without initialization.
     * @param edge
     * pointer to the edge in the graph where the node will be inserted
     * @param node
     * pointer to the inserted node
     * @param initNode
     * parameter that determines whether the node needs to be initialized
     * @return true in case of success, false otherwise.
     */
    bool InsertNode(const EdgePtr& edge, const NodePtr& node, bool initNode = false);

    /**
     * @brief Insert Node between two specified nodes.
     * This procedure creates two edges that link the parent and child nodes to the inserted one and adds all created
     * objects to the graph. This method supports two regimes. First, the node is inserted without initialization (i.e.
     * supported descriptors initialization, supported primitive descriptors selection, etc.), which can be useful after
     * the ResolveEdgeConflicts() completes. The second is just inserting the node without initialization.
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
    bool InsertNode(const NodePtr& parent,
                    const NodePtr& child,
                    const NodePtr& node,
                    int parentPort,
                    int childPort,
                    bool initNode = false);

    std::shared_ptr<ov::Model> dump() const;

    void ResetInferCount() {
        infer_count = 0;
    }

    void SortTopologically();

    bool hasDynamicInput() const {
        return graphHasDynamicInput;
    }

    void Init(const std::vector<NodePtr>& graphNodes,
              const std::vector<EdgePtr>& graphEdges,
              const GraphContext::CPtr& context,
              std::string name);

    /**
     * Init graph using \p model, \p context, \p inputConfigs and \p outputConfigs
     */
    void Init(const std::shared_ptr<const ov::Model>& model,
              const GraphContext::CPtr& context,
              const std::vector<node::Input::InputConfig>& inputConfigs = {},
              const std::vector<node::Input::OutputConfig>& outputConfigs = {});

    /**
     * Activate execution graph
     */
    void Activate();

    /**
     * Register the graph in the global allocation context by transforming
     * local execution data into the global one:
     * 1) Local execution indices are transformed into global ones, represented by input and output execution index
     *    where output execution index is an index of the last node of the inner graph
     * 2) Local sync node indices are transformed into global ones using global input execution index
     * 3) Local edges are added to the global list of edges
     *
     * Example graph with subgraphs:
     * 0 -> 1 -> 2 -> 3 [0 -> 1 -> 2] -> 4 [0 -> 1] -> 5
     *
     * Virtually flatten:
     * 0(0) -> 1(1) -> 2(2) -> 3(5) [3 -> 4 -> 5] -> 6(7) [6 -> 7] -> 8
     *
     * This is basically an equivalent to the actually flatten graph:
     * 0 -> 1 -> 2 -> [3 -> 4 -> 5] -> [6 -> 7] -> 8
     */
    int RegisterToAllocationContext(int offset, AllocationContext& context);

    const std::unordered_map<std::size_t, ProxyMemoryBlockPtr>& getOutputNodesMemBlocksMap() const {
        return m_outputNodesMemBlocks;
    }

protected:
    void ForgetGraphData() {
        status = Status::NotReady;

        inputNodesMap.clear();
        outputNodesMap.clear();
        graphNodes.clear();
        graphEdges.clear();
        m_executableSyncNodesInds.clear();
    }
    Status status{Status::NotReady};

    // For dumping purposes. -1 - no counting, all other positive
    // values mean increment it within each Infer() call
    int infer_count = -1;

    std::vector<NodePtr> graphNodes;
    std::vector<EdgePtr> graphEdges;

    std::string _name;

    bool graphHasDynamicInput = false;

    void Replicate(const std::shared_ptr<const ov::Model>& subgraph,
                   const std::vector<node::Input::InputConfig>& inputConfigs = {},
                   const std::vector<node::Input::OutputConfig>& outputConfigs = {});

    void Configure(bool optimize = true);
    void Allocate();

    void InitNodes();
    void InitDescriptors();
    void ResolveInplaceDirections();
    void InitOptimalPrimitiveDescriptors();
    void ResolveEdgeConflicts();
    void ResolveComplexInplaceConflicts();
    bool ProcessDynNodes() const;
    void AllocateWithReuse(const std::vector<size_t>& syncNodesInds, GlobalExecutionIndex globalExecIndex);
    void CreatePrimitivesAndExecConstants() const;
    std::vector<size_t> CreateExecutionGraph();

    /**
     * Execute a given \p node within \p request using \p numaId
     * and catch possible exceptions to include extra information
     *
     * @params node     Node to execute
     * @params request  Current inference request, which is checked for cancelation
     * @params numaId   Numa Id to be used for an execution
     */
    void ExecuteNodeWithCatch(const NodePtr& node, SyncInferRequest* request = nullptr, int numaId = -1) const;

    /**
     * Execute a given \p node within \p request using \p numaId
     *
     * @params node     Node to execute
     * @params request  Current inference request, which is checked for cancelation
     * @params numaId   Numa Id to be used for an execution
     */
    void ExecuteNode(const NodePtr& node, SyncInferRequest* request = nullptr, int numaId = -1) const;

    void InferStatic(SyncInferRequest* request, int numaId);
    template <typename UpdateStrategy>
    void InferDynamic(SyncInferRequest* request, int numaId, UpdateStrategy&& update);

    friend std::shared_ptr<ov::Model> dump_graph_as_ie_ngraph_net(const Graph& graph);

private:
    using event_t = void (Graph::*)(void);

private:
    void EnforceInferencePrecision();
    void EnforceBF16();
    void insertReorder(EdgePtr& edge, bool isOptimized, std::unordered_set<std::string>& uniqueLayerNames);
    void insertConvert(EdgePtr& edge);

private:
    // TODO: change std::map to std::unordered_map
    std::map<std::size_t, NodePtr> inputNodesMap;
    std::map<std::size_t, NodePtr> outputNodesMap;

    OutputMemoryBlocks m_outputNodesMemBlocks;

    // these node pointers (from graphNodes) are to avoid regular checking for
    // constantness of nodes in Infer methods and calls of
    // non-executable (optimized out) nodes, such as Input, Reshape, etc.
    std::vector<NodePtr> m_executableGraphNodes;
    std::vector<size_t> m_executableSyncNodesInds;

    GraphContext::CPtr m_context;
    dnnl::stream m_stream;
};

using GraphPtr = std::shared_ptr<Graph>;

}  // namespace intel_cpu
}  // namespace ov
