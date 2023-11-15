// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/primitive_desc.hpp"
#include "common/primitive_desc_iface.hpp"
#include "dnnl_extension_utils.h"
#include "extension_mngr.h"
#include "graph_dumper.h"
#include "graph_optimizer.h"
#include "ie_algorithm.hpp"
#include "infer_request.h"
#include "itt.h"
#include "low_precision/low_precision.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "memory_solver.hpp"
#include "nodes/common/cpu_convert.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/convert.h"
#include "nodes/fullyconnected.h"
#include "nodes/input.h"
#include "nodes/reorder.h"
#include "nodes/subgraph.h"
#include "nodes/memory.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/ops.hpp"
#include "precision_utils.h"
#include "transformations/utils/utils.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"
#include "utils/node_dumper.h"
#include "utils/verbose.h"
#include "memory_desc/cpu_memory_desc_utils.h"

#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
#    include <tbb/task.h>
#endif

using namespace dnnl;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_cpu {

typedef std::unordered_set<EdgePtr> edge_cluster_t;
typedef std::vector<edge_cluster_t> edge_clusters_t;

Graph::~Graph() {
    CPU_DEBUG_CAP_ENABLE(summary_perf(*this));
}

template<typename NET>
void Graph::CreateGraph(NET &net, const GraphContext::CPtr ctx) {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "CreateGraph");

    if (IsReady())
        ForgetGraphData();

    context = ctx;

    Replicate(net);

    InitGraph();

    CPU_DEBUG_CAP_ENABLE(serialize(*this));
}

void Graph::CreateGraph(const std::vector<NodePtr>& graphNodes,
                        const std::vector<EdgePtr>& graphEdges,
                        const GraphContext::CPtr ctx,
                        std::string name) {
    if (IsReady())
        ForgetGraphData();

    context = ctx;

    this->_name = std::move(name);
    this->reuse_io_tensors = false;

    this->graphNodes = graphNodes;
    this->graphEdges = graphEdges;

    for (auto node : graphNodes) {
        if ("Parameter" == node->getTypeStr()) {
            inputNodesMap[node->getName()] = node;
        } else if ("Result" == node->getTypeStr()) {
            outputNodesMap[node->getName()] = node;
        }
    }

    InitGraph();

    CPU_DEBUG_CAP_ENABLE(serialize(*this));
}

template void Graph::CreateGraph(const std::shared_ptr<const ov::Model>&, const GraphContext::CPtr);
void Graph::Replicate(const std::shared_ptr<const ov::Model> &model) {
    OV_ITT_SCOPE_CHAIN(FIRST_INFERENCE, taskChain, itt::domains::intel_cpu_LT, "Graph::Replicate", "ov::Model");
    this->_name = model->get_friendly_name();
    this->reuse_io_tensors = false;

    // Map data object onto producer node
    std::map<std::shared_ptr<ov::Node>, NodePtr> op2node;

    // nodes which has no consumers (output or just unused). But doesn't marked as graph output.
    // Will be stored as fake output separately.
    std::deque<ov::Output<ov::Node>> unusedOutputs;

    auto getParentOutputPort = [](const std::shared_ptr<ov::Node> childOp,
                                  const std::shared_ptr<ov::Node> parentOp,
                                  const size_t childInputPort) -> int {
        for (size_t parentPort = 0; parentPort < parentOp->get_output_size(); parentPort++) {
            if (childOp->input(childInputPort).get_tensor_ptr() == parentOp->output(parentPort).get_tensor_ptr()) {
                return static_cast<int>(parentPort);
            }
        }

        return -1;
    };

    const bool is_legacy_api = getConfig().isLegacyApi;
    for (const auto& op : model->get_ordered_ops()) {
        const NodePtr node {Node::factory().create(op, context)};

        graphNodes.push_back(node);
        if (op->get_type_info() == op::v0::Parameter::get_type_info_static()) {
            const std::string name = get_port_name(ov::Output<ov::Node>(op, 0), is_legacy_api);
            inputNodesMap[name] = node;
            if (node->isDynamicNode()) {
                graphHasDynamicInput = true;
            }
        }

        if (op->get_type_info() == op::v0::Result::get_type_info_static()) {
            const std::string inputID = get_port_name(op->output(0), is_legacy_api);
            outputNodesMap[inputID] = node;
        }

        op2node[op] = node;

        for (size_t port = 0; port < op->get_input_size(); port++) {
            auto parentOp = op->get_input_node_shared_ptr(port);
            auto parentNode = op2node[parentOp];

            EdgePtr edge(new Edge(parentNode, node, getParentOutputPort(op, parentOp, port), static_cast<int>(port)));
            node->addEdge(edge);
            graphEdges.push_back(edge);
        }

        if (!one_of(op->get_type_info(),
                op::v0::Result::get_type_info_static(),
                op::v3::Assign::get_type_info_static(),
                op::v6::Assign::get_type_info_static())) {
            for (size_t oi = 0; oi < op->get_output_size(); oi++) {
                if (op->get_output_target_inputs(oi).empty()) {
                    unusedOutputs.push_back(op->output(oi));
                }
            }
        }
    }

    // Add stub output node for unused data
    for (auto unusedOutput : unusedOutputs) {
        auto parentNode = op2node[unusedOutput.get_node_shared_ptr()];
        const auto port = unusedOutput.get_index();
        const auto nodeName = std::string("stub_") + std::to_string(unusedOutput.get_index()) + "_" + parentNode->getName();
        const NodePtr outNode = std::make_shared<node::Input>(parentNode->outputShapes[port],
                                                                        parentNode->getOriginalOutputPrecisionAtPort(port),
                                                                        nodeName, "Result", context);
        EdgePtr edge(new Edge(parentNode, outNode, port, 0));
        outNode->addEdge(edge);
        graphEdges.push_back(edge);
        graphNodes.push_back(outNode);
    }

    auto hasSubgraphConsumers = [](const NodePtr& node) -> bool {
        const auto& childEdges = node->getChildEdges();
        return std::any_of(childEdges.begin(), childEdges.end(), [](const EdgeWeakPtr& edge) -> bool {
            auto edgePtr = edge.lock();
            if (!edgePtr)
                return false;
            return edgePtr->getChild()->getType() == Type::Subgraph;
        });
    };

    // enforce must be performed after inputs and outputs info are taken into account
    EnforceInferencePrecision();
    // also we need to change input/output precisions for consumers/producers to avoid inserting reorder
    for (auto &input : inputNodesMap) {
        const auto& inputNode = input.second;
        const auto precToSet = inputNode->getOriginalOutputPrecisionAtPort(0);
        const auto childEdges = inputNode->getChildEdgesAtPort(0);
        for (size_t i = 0; i < childEdges.size(); i++) {
            const auto child = childEdges[i]->getChild();
            const auto child_prec = child->getOriginalInputPrecisionAtPort(childEdges[i]->getOutputNum());
            if (!one_of(child_prec, Precision::BF16, Precision::FP16) &&
                // remove this WA when #78939 is resolved
                !hasSubgraphConsumers(child))
                child->setOriginalInputPrecisionAtPort(childEdges[i]->getOutputNum(), precToSet);
        }
    }

    for (auto &output : outputNodesMap) {
        const auto& outputNode = output.second;
        const auto precToSet = outputNode->getOriginalInputPrecisionAtPort(0);
        const auto parentEdges = outputNode->getParentEdgesAtPort(0);
        for (size_t i = 0; i < parentEdges.size(); i++) {
            const auto parent = parentEdges[i]->getParent();
            parent->setOriginalOutputPrecisionAtPort(parentEdges[i]->getInputNum(), precToSet);
        }
    }
}

void Graph::InitGraph() {
    GraphOptimizer optimizer;

    SortTopologically();
    InitNodes();

    optimizer.ApplyCommonGraphOptimizations(*this);
    SortTopologically();

    InitDescriptors();

    ResolveInplaceDirections();

    InitOptimalPrimitiveDescriptors();

    ResolveEdgeConflicts();

    optimizer.ApplyImplSpecificGraphOptimizations(*this);
    SortTopologically();

    const bool hasDynNodes = ProcessDynNodes();

    Allocate();

    CreatePrimitivesAndExecConstants();

#ifndef CPU_DEBUG_CAPS
    for (auto &graphNode : graphNodes) {
        graphNode->cleanup();
    }
#endif

    ExtractExecutableNodes();
    SearchInternalStateNodes();

    status = hasDynNodes ? Status::ReadyDynamic : Status::ReadyStatic;
}

void Graph::InitNodes() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::InitNodes");
    for (auto &node : graphNodes) {
        node->init();
    }
}

void Graph::InitDescriptors() {
    OV_ITT_SCOPE_CHAIN(FIRST_INFERENCE, taskChain, itt::domains::intel_cpu_LT, "InitDescriptors", "Prepare");

    for (auto &node : graphNodes) {
        OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, node->profiling.getSupportedDescriptors);
        DEBUG_LOG("Get supported primitive descriptors for node: ", node->getName());
        node->getSupportedDescriptors();

        OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, node->profiling.initSupportedPrimitiveDescriptors);
        DEBUG_LOG("Init supported primitive descriptors for node: ", node->getName());
        node->initSupportedPrimitiveDescriptors();

        OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, node->profiling.filterSupportedPrimitiveDescriptors);
        DEBUG_LOG("Filter supported primitive descriptors for node: ", node->getName());
        node->filterSupportedPrimitiveDescriptors();

#ifdef CPU_DEBUG_CAPS
        const auto& SPDs = node->getSupportedPrimitiveDescriptors();
        for (size_t i = 0; i < SPDs.size(); i++) {
            DEBUG_LOG("#",
                      node->getExecIndex(),
                      " ",
                      node->getName(),
                      "  SupportedPrimitiveDescriptors [",
                      i,
                      "/",
                      SPDs.size(),
                      "]: \n",
                      SPDs[i]);
        }
#endif
    }

    for (auto &node : graphNodes) {
        OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, node->profiling.selectOptimalPrimitiveDescriptor);
        DEBUG_LOG("Select optimal primitive descriptors for node: ", node->getName());
        node->selectOptimalPrimitiveDescriptor();
    }
}

void Graph::ResolveInplaceDirections() {
     OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "Graph::ResolveInplaceDirections");

    for (auto& node : graphNodes) {
        resolveInPlaceDirection(node);
    }
}


void Graph::InitOptimalPrimitiveDescriptors() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "Graph::InitOptimalPrimitiveDescriptors");
    for (auto &node : graphNodes) {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, node->profiling.initOptimalPrimitiveDescriptor);
        DEBUG_LOG("Init optimal primitive descriptors for node: ", node->getName());
        node->initOptimalPrimitiveDescriptor();
        DEBUG_LOG("#", node->getExecIndex(), " ", node->getName(), "\n",
                  *node->getSelectedPrimitiveDescriptor(), "selectedPrimitiveDescriptorIdx = ", node->selectedPrimitiveDescriptorIndex);
    }
}

void Graph::ExtractExecutableNodes() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::ExtractExecutableNodes");
    for (const auto& graphNode : graphNodes) {
        if ((!graphNode->isConstant() && CPU_DEBUG_CAPS_ALWAYS_TRUE(graphNode->isExecutable())) || graphNode->isDynamicNode()) {
            /* @todo
             * Revise implementation.
             * With current way it is possible that with debug_caps enabled
             * we execute a node, which is not ready to be executed
             */
            auto itr = syncNodesInds.find(graphNode.get());
            if (itr != syncNodesInds.end()) {
                itr->second = executableGraphNodes.size();
            }
            executableGraphNodes.emplace_back(graphNode);
        }
    }
}

void Graph::CreatePrimitivesAndExecConstants() const {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::CreatePrimitivesAndExecConstants");
    dnnl::stream stream(getEngine());

    using shared_memory_ptr = WeightsSharing::SharedMemory::Ptr;

    auto acquireSharedOutputs = [this](const NodePtr & node) {
        std::vector<shared_memory_ptr> outputs;
        bool hasLocalAllocatedEdges = false;
        bool hasExternalInvalidEdges = false;

        for (size_t i = 0; i < node->getChildEdges().size(); ++i) {
            auto edgePtr = node->getChildEdgeAt(i);
            if (edgePtr) {
                if (edgePtr->isUseExternalMemory()) {
                    auto ptr = context->getWeightsCache()->get(edgePtr->name());
                    outputs.emplace_back(ptr);
                    if (!ptr->isValid())
                        hasExternalInvalidEdges = true;
                } else {
                    hasLocalAllocatedEdges = true;
                }
            }
        }

        return std::make_tuple(hasExternalInvalidEdges, hasLocalAllocatedEdges, outputs);
    };

    for (const auto &node : graphNodes) {
        {
            OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, node->profiling.createPrimitive);
            DEBUG_LOG(*node);
            node->createPrimitive();
        }

        if (!node->isConstant()) {
            continue;
        }

        if (context->getWeightsCache()) {
            auto sharedOutputs = acquireSharedOutputs(node);

            if (std::get<0>(sharedOutputs) || std::get<1>(sharedOutputs)) {
                ExecuteNode(node, stream);

                for (auto & output : std::get<2>(sharedOutputs))
                    output->valid(true);
            }
        } else {
            ExecuteNode(node, stream);
        }
    }
}

static bool isReorderAvailable(const MemoryDescPtr& parentDesc, const MemoryDescPtr& childDesc, const dnnl::engine& eng) {
    auto definedParentDesc = parentDesc->isDefined() ? parentDesc : MemoryDescUtils::makeDummyDesc(*parentDesc);
    memory::desc srcMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(definedParentDesc)->getDnnlDesc();

    auto definedChildDesc = childDesc->isDefined() ? childDesc : MemoryDescUtils::makeDummyDesc(*childDesc);
    memory::desc dstMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(definedChildDesc)->getDnnlDesc();

    dnnl::primitive_attr attr;

    dnnl_primitive_desc_t result = nullptr;
    auto status = dnnl_reorder_primitive_desc_create(&result, srcMemDesc.get(), eng.get(), dstMemDesc.get(), eng.get(),
                                                     attr.get());
#if defined(OV_CPU_ARM_ENABLE_FP16)
    // temporary WA for slow FP32->FP16 conversion reorder in oneDNN on ARM
    // pretend the reorder is not available to use Convert node instead
    if (result && parse_impl_name(result->impl()->name()) == ref_any) {
        dnnl_primitive_desc_destroy(result);
        return false;
    }
#endif
    if (result) {
        dnnl_primitive_desc_destroy(result);
    }

    return dnnl_success == status;
}

void Graph::ResolveEdgeConflicts() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::ResolveEdgeConflicts");

    ptrdiff_t numberOfEdges = static_cast<ptrdiff_t>(graphEdges.size());

    std::unordered_set<std::string> uniqueLayerNames;
    for (auto node : graphNodes) {
        uniqueLayerNames.insert(node->getName());
    }

    auto insertReorder = [&](EdgePtr& edge, bool isOptimized) {
        std::string basicLayerName = edge->getParent()->getName() + "_" +
                                     node::Reorder::getReorderArgs(edge->getInputDesc(), edge->getOutputDesc()) + "_" +
                                     edge->getChild()->getName();
        std::string layerName = basicLayerName;
        int idx = 0;
        while (uniqueLayerNames.find(layerName) != uniqueLayerNames.end()) {
            idx++;
            layerName = basicLayerName + "_" + std::to_string(idx);
        }
        uniqueLayerNames.insert(layerName);

        // optimized flag indicate that just desc update w/o actual physical memory movement.
        InsertReorder(edge, layerName, edge->getInputDesc(), edge->getOutputDesc(), isOptimized);
    };

    auto updateEdge = [&](ptrdiff_t& i) {
        graphEdges.erase(graphEdges.begin() + i);
        i--;
        numberOfEdges--;
    };

    for (ptrdiff_t i = 0; i < numberOfEdges; i++) {
        auto edge = graphEdges[i];
        auto reorderStatus = graphEdges[i]->needReorder();
        DEBUG_LOG(graphEdges[i]->name(), " reorderStatus = ", reorderStatus);
        if (reorderStatus == Edge::ReorderStatus::Regular) {
            Edge::ReorderStatus reorderStatusInternal = Edge::ReorderStatus::Regular;
            // Check if there is a reorder that needs the precision conversion
            if (edge->getInputDesc().getPrecision() != edge->getOutputDesc().getPrecision() &&
                    !isReorderAvailable(edge->getInputPortDesc()->getMemDesc(),
                                        edge->getOutputPortDesc()->getMemDesc(),
                                        this->getEngine())) {
                // If we are here, then we need to insert Convert, because there are no reorders that support such type conversion
                const auto& inDesc = edge->getInputDesc();
                const auto& outDesc = edge->getOutputDesc();

                std::string convertName = edge->getParent()->getName() + "_" +
                                          inDesc.getPrecision().name() + "_" + outDesc.getPrecision().name();

                auto convertNode = std::make_shared<node::Convert>(inDesc.getShape(), inDesc.getPrecision(), outDesc.getPrecision(),
                                                                       convertName, context);
                convertNode->setDescs(inDesc, outDesc);
                InsertNode(edge, convertNode, true);

                //Check if reorder is still needed
                reorderStatusInternal = convertNode->getChildEdgeAt(0)->needReorder();
                if (reorderStatusInternal != Edge::ReorderStatus::No)
                    edge = convertNode->getChildEdgeAt(0);
            }
            if (reorderStatusInternal != Edge::ReorderStatus::No) {
                insertReorder(edge, reorderStatusInternal == Edge::ReorderStatus::Optimized);
            }
            updateEdge(i);
        } else if (reorderStatus == Edge::ReorderStatus::Optimized) {
            insertReorder(edge, true);
            updateEdge(i);
        }
    }

    // secondary pass to eliminate complex inplace conflicts
    auto needReorder = [](const EdgePtr& edge) -> bool {
        int inNumber = edge->getInputNum();
        const auto portChildEdges = edge->getParent()->getChildEdgesAtPort(inNumber);
        if (portChildEdges.size() > 1) {
            if (auto modifyingNode = edge->modifiedInPlace()) {
                auto execIndex = modifyingNode->getExecIndex();
                for (auto pEdgePeer : portChildEdges) {
                    if (pEdgePeer == edge)
                        continue;
                    std::vector<NodePtr> vecConsumers;
                    pEdgePeer->collectConsumers(vecConsumers);

                    for (auto node : vecConsumers) {
                        if (node->getExecIndex() >= execIndex ||
                            one_of(node->getType(), Type::MemoryOutput, Type::Output)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    };

    numberOfEdges = graphEdges.size(); //update the total number

    for (ptrdiff_t i = 0; i < numberOfEdges; i++) {
        auto edge = graphEdges[i];
        if (needReorder(edge)) {
            constexpr bool optimizedReorder = false;
            insertReorder(edge, optimizedReorder);
            updateEdge(i);
        }
    }
}

static inline bool isConstOutput(EdgePtr edge) {
    return edge->getParent()->isConstant() && !edge->getChild()->isConstant();
}

static edge_clusters_t findEdgeClusters(const std::vector<EdgePtr> & graphEdges) {
    typedef std::unordered_map<EdgePtr, size_t> edge_cluster_idx_map_t;

    edge_clusters_t edge_clusters;
    edge_cluster_idx_map_t edge_cluster_indices;

    for (auto &edge : graphEdges) {
        auto edge_it = edge_cluster_indices.find(edge);
        if (edge_it != edge_cluster_indices.end())
            continue;   // edge is visited

        size_t cluster_idx = edge_clusters.size();
        EdgePtr last_shared_edge = nullptr;

        // find cluster index
        for (auto shared_edge = edge->getSharedEdge(std::nothrow);
            shared_edge;
            shared_edge = shared_edge->getSharedEdge(std::nothrow)) {
            auto shared_edge_it = edge_cluster_indices.find(shared_edge);
            if (shared_edge_it != edge_cluster_indices.end()) {
                cluster_idx = shared_edge_it->second;
                last_shared_edge = shared_edge;
                break;
            }
        }

        // add shared edges to cluster
        edge_cluster_indices.emplace(edge, cluster_idx);

        if (cluster_idx == edge_clusters.size())
            edge_clusters.emplace_back(edge_cluster_t { edge });
        else
            edge_clusters[cluster_idx].emplace(edge);

        for (auto shared_edge = edge->getSharedEdge(std::nothrow);
            shared_edge != last_shared_edge;
            shared_edge = shared_edge->getSharedEdge(std::nothrow)) {
            edge_cluster_indices.emplace(shared_edge, cluster_idx);
            edge_clusters[cluster_idx].emplace(shared_edge);
        }
    }

    return edge_clusters;
}

void Graph::AllocateWithReuse() {
    edge_clusters_t edge_clusters = findEdgeClusters(graphEdges);

    size_t remaining_edge_clusters_count = edge_clusters.size();

    // Resolve special cases:
    for (size_t i = 0; i < remaining_edge_clusters_count;) {
        auto &cluster = edge_clusters[i];
        bool erase = false;
        for (auto &edge : cluster) {
            // Remove already allocated edges from the mem reuse algo
            if (edge->getStatus() == Edge::Status::Allocated) {
                erase = true;
                break;
            }
            // Special allocation for constants
            if (edge->getStatus() != Edge::Status::NeedAllocation || !edge->getParent()->isConstant()) {
                continue;
            }
            if (edge->getParent()->getType() == Type::Input) {
                auto constNode = std::static_pointer_cast<node::Input>(edge->getParent());
                edge->reuse(std::const_pointer_cast<IMemory>(constNode->getMemoryPtr()));
            } else {
                edge->externalAllocate(context->getWeightsCache());
            }
            erase = true;
        }

        if (erase) {
            std::swap(edge_clusters[i], edge_clusters[remaining_edge_clusters_count - 1]);
            --remaining_edge_clusters_count;
        } else {
            ++i;
        }
    }

    const int64_t alignment = 32;  // 32 bytes

    // Markup the boxes
    std::vector<MemorySolver::Box> definedBoxes;
    std::vector<MemorySolver::Box> undefinedBoxes;
    for (size_t i = 0; i < remaining_edge_clusters_count; i++) {
        MemorySolver::Box box = { std::numeric_limits<int>::max(), 0, 0, static_cast<int64_t>(i) };
        int64_t boxSize = 0;
        for (auto &edge : edge_clusters[i]) {
            int e_start = edge->getParent()->execIndex;
            int e_finish = edge->getChild()->execIndex;

            if (boxSize != -1 && edge->getDesc().isDefined()) {
                int64_t e_size = edge->getDesc().getCurrentMemSize();  // size in bytes (from the beginning of data to the last element)
                boxSize = std::max(e_size, boxSize);
            } else {
                boxSize = -1;
            }

            box.start = std::min(e_start, box.start);
            box.finish = std::max(e_finish, box.finish);
        }

        // Constant data are filled once on load.
        // So we need it untouchable during all execution time
        // -1 is a place holder for a max timestamp.
        bool isConst = false, isOutput = false, isInput = false;
        for (auto &edge : edge_clusters[i]) {
            isConst  |= isConstOutput(edge);
            isOutput |= edge->getChild()->getType() == Type::Output;
            isInput  |= edge->getParent()->getType() == Type::Input;
        }

        if (reuse_io_tensors) {
            if (isInput | isConst) box.start = 0;
            if (isOutput | isConst) box.finish = -1;
        } else {
            if (isInput  | isOutput | isConst) {
                box.start = 0;
                box.finish = -1;
            }
        }

        if (boxSize != -1) {
            box.size = div_up(boxSize, alignment);
            definedBoxes.push_back(box);
        } else {
            box.size = boxSize;
            undefinedBoxes.push_back(box);
        }
    }

    // Process defined boxes (static shapes)
    MemorySolver staticMemSolver(definedBoxes);
    size_t total_size = static_cast<size_t>(staticMemSolver.solve()) * alignment;

    memWorkspace = std::make_shared<Memory>(getEngine(), DnnlBlockedMemoryDesc(InferenceEngine::Precision::I8, Shape(VectorDims{total_size})));

    if (edge_clusters.empty())
        return;

    auto* workspace_ptr = static_cast<int8_t*>(memWorkspace->getData());

    for (const auto& box : definedBoxes) {
        int count = 0;
        for (auto& edge : edge_clusters[box.id]) {
            if (edge->getStatus() == Edge::Status::NeedAllocation) {
                int64_t offset = staticMemSolver.getOffset(box.id);
                // !! Fallback to individual memory allocation !!
                // if you like to check infer without reuse just call this function without arguments.
                edge->allocate(workspace_ptr + offset * alignment);  // alignment in byte

                // TODO: WA for some test (like strided_slice_test) which use tensors with
                //       shapes {0}. And it is implicitly converted into {1} tensor.
                //       Zeroing of input data allow pass tests.
                if (edge->getParent()->type == Type::Input && edge->hasDefinedMaxSize())
                    edge->getMemoryPtr()->nullify();

                count++;
            }
        }
        OPENVINO_ASSERT(count == 1);
    }

    //Process undefined boxes (dynamic shapes)
    if (!undefinedBoxes.empty()) {
        // Use proxy memory manager for output edges
        for (const auto& box : undefinedBoxes) {
            for (auto& edge : edge_clusters[box.id]) {
                const auto child = edge->getChild();
                if (child->getType() == Type::Output &&
                    edge->getStatus() == Edge::Status::NeedAllocation) {
                    auto proxyMemMngr =
                        std::make_shared<ProxyMemoryMngr>();
                    DEBUG_LOG("ProxyMemoryMngr ", proxyMemMngr, " ", this);
                    edge->allocate(proxyMemMngr);

                    // Store the output memory managers.
                    // So that, the infer requests can be able to access them.
                    int count = 0;
                    for (auto &output : outputNodesMap) {
                        if (output.second == child) {
                            outputNodesMemMngrMap[output.first] = proxyMemMngr;
                            count++;
                        }
                    }
                    // sometimes there are unused output ports.
                    OPENVINO_ASSERT(count <= 1, "cannot find output node. count ", count);
                }
            }
        }

        if (!syncNodesInds.empty()) {
            //We have to extend the lifespan of tensors that are crossing a sync point border in order to save
            //the intermediate computation results from possible loss due to the tensor resize
            std::vector<int> vecIntervals = {0};
            for (const auto& item : syncNodesInds) {
                vecIntervals.push_back(item.first->execIndex);
            }
            std::sort(vecIntervals.begin(), vecIntervals.end());
            for (auto& box : undefinedBoxes) {
                if (-1 == box.finish) {
                    continue;
                }
                auto itr_upper = std::upper_bound(vecIntervals.begin(), vecIntervals.end(), box.finish, [](int y, int x) { return y <= x;});
                auto itr_lower = std::lower_bound(vecIntervals.begin(), vecIntervals.end(), box.start);
                if (itr_lower != itr_upper) { // across sections
                    if (itr_upper == vecIntervals.end()) {
                        box.finish = -1;
                    } else {
                        box.finish = *itr_upper;
                    }
                }
            }
        }

        MemorySolver::normalizeBoxes(undefinedBoxes);

        std::vector<std::vector<MemorySolver::Box>> groups; //groups of nonoverlapping boxes
        constexpr bool enableMemReuse = true; // set false to disable mem reuse for debug purposes
        if (enableMemReuse) {
            groups.push_back({undefinedBoxes.front()});
            for (size_t i = 1; i < undefinedBoxes.size(); ++i) {
                const auto& box = undefinedBoxes[i];
                bool groupFound = false;
                for (auto& group : groups) {
                    const auto& lastBox = group.back();
                    if (lastBox.start > box.finish || lastBox.finish < box.start) {
                        group.push_back(box);
                        groupFound = true;
                        break;
                    }
                }

                if (!groupFound) {
                    groups.push_back({box});
                }
            }
        } else {
            for (auto& box : undefinedBoxes) {
                groups.push_back({box});
            }
        }
        for (auto& group : groups) {
            auto grpMemMngr =
                std::make_shared<DnnlMemoryMngr>(make_unique<MemoryMngrWithReuse>());
            for (auto& box : group) {
                for (auto& edge : edge_clusters[box.id]) {
                    if (edge->getStatus() == Edge::Status::NeedAllocation) {
                        edge->allocate(grpMemMngr);
                    }
                }
            }
        }
    }

    // Resolve all other edges with status NotAllocated and in-place
    for (auto& cluster : edge_clusters) {
        for (auto& edge : cluster) {
            if (edge->getStatus() != Edge::Status::NotAllocated) {
                continue;
            }
            std::vector<EdgePtr> edges_to_process;
            edges_to_process.push_back(edge);
            for (auto next_edge = edge->getSharedEdge(std::nothrow);
                next_edge;
                next_edge = next_edge->getSharedEdge(std::nothrow)) {
                edges_to_process.push_back(next_edge);
            }
            std::for_each(edges_to_process.rbegin(), edges_to_process.rend(), [](const EdgePtr& edge) {
                if (edge->getStatus() == Edge::Status::NotAllocated) {
                    if (edge->inPlace(Edge::LOOK_DOWN)) {
                        edge->getChild()->resolveInPlaceEdges(Edge::LOOK_DOWN);
                    } else if (edge->inPlace(Edge::LOOK_UP)) {
                        edge->getParent()->resolveInPlaceEdges(Edge::LOOK_UP);
                    } else {
                        auto sharedEdge = edge->getSharedEdge();
                        auto sharedEdgeParent = sharedEdge->getParent();
                        edge->allocate(sharedEdge->getMemoryPtr()->getMemoryMngr());
                        DEBUG_LOG(*edge, " sharedEdge with ", *sharedEdge);
                    }
                }
            });
        }
    }
}

void Graph::Allocate() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::Allocate");

    //resolve inplace dead end nodes
    for (const auto& edge : graphEdges) {
        if (edge->getStatus() == Edge::Status::Uninitialized) {
            if (one_of(edge->getParent()->getType(), Type::Input, Type::MemoryInput) && edge->inPlace(Edge::LOOK_UP)) {
                edge->getParent()->resolveInPlaceEdges(Edge::LOOK_UP);
            } else if (one_of(edge->getChild()->getType(), Type::Output, Type::MemoryOutput) && edge->inPlace(Edge::LOOK_DOWN)) {
                edge->getChild()->resolveInPlaceEdges(Edge::LOOK_DOWN);
            }
        }
    }

    // resolve edges. Define which will be a view on others
    //   NeedAllocation - real blob
    //   NotAllocated - view on other blob, peer or in-place
    for (auto& edge : graphEdges) edge->init();

    // Allocate memory space for all edges marked with NeedAllocation
    AllocateWithReuse();

    // Check all getters. Should work.
    for (auto& edge : graphEdges) edge->validate();
}

bool Graph::ProcessDynNodes() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::ProcessDynNodes");

    bool result = false;
    for (size_t i = 0; i < graphNodes.size(); ++i) {
        const auto& node = graphNodes[i];
        if (node->isDynamicNode()) {
            result = true;
            if (node->outputShapeDataDependency() ||
                // WA: for convolution plus sum(broadcast). Due to the fact that a convolution with sum use the same memory for second sum term and the output
                // tensors (inPlace) resizing the output tensor, may lead to reallocation of this second term memory and possible data lost. The reallocation
                // may happen when the second term shape is broadcasted to the output tensor shape. To avoid the data loss, we have a special processing for
                // such cases inside the convolution node, but it works properly only when dynamic shapes inference, preparation and execution a called
                // for this node sequentially.
                (node->getType() == Type::Convolution && node->isInPlace())) {
                syncNodesInds.insert({node.get(), i});
            }
        }
    }

    // In case of dynamic shapes, tensors may be resized due to the shapes variations.
    // If the input tensor is included to memory reuse, it means that its memory manager is shared with other tensors in the graph, which in turn may cause data
    // loss when one of the tensors down the graph requests mem resize, while the input data have not been yet read by the consumers. To avoid such situations
    // we disable io mem reuse for the case of dynamic shapes.
    if (result) {
        this->reuse_io_tensors = false;
    }
    return result;
}

void Graph::PushInputData(const std::string& name, const ov::SoPtr<ITensor>& input) {
    if (!IsReady()) OPENVINO_THROW("Wrong state. Topology not ready.");
    auto input_itr = inputNodesMap.find(name);
    if (input_itr != inputNodesMap.end()) {
        auto create_mem_desc = [&](const ov::SoPtr<ITensor>& tensor) -> CpuBlockedMemoryDesc {
            auto element_type = tensor->get_element_type();
            auto shape = tensor->get_shape();
            if (shape.empty())
                shape = {tensor->get_size()};
            std::vector<size_t> blk_order(shape.size());
            std::iota(blk_order.begin(), blk_order.end(), 0);
            std::vector<size_t> dim_offset(shape.size(), 0);
            std::vector<size_t> blk_strides;
            auto byte_strides = element_type.bitwidth() >= 8 ? tensor->get_strides() : Strides{};
            if (byte_strides.empty()) {
                blk_strides = ov::row_major_strides(shape);
            } else {
                // ROI tensor need figure out correct blk_strides
                blk_strides.resize(byte_strides.size());
                std::transform(byte_strides.begin(),
                               byte_strides.end(),
                               blk_strides.begin(),
                               [&element_type](size_t byte_stride) {
                                   OPENVINO_ASSERT(byte_stride % element_type.size() == 0,
                                                   "Limitation: Stride in bytes ",
                                                   byte_stride,
                                                   " should be divisible by size of element ",
                                                   element_type.size());
                                   return byte_stride / element_type.size();
                               });
            }
            InferenceEngine::TensorDesc tensorDesc(
                InferenceEngine::details::convertPrecision(tensor->get_element_type()),
                shape,
                InferenceEngine::BlockingDesc{shape, blk_order, 0, dim_offset, blk_strides});
            return MemoryDescUtils::convertToCpuBlockedMemoryDesc(tensorDesc);
        };

        auto node = input_itr->second;
        auto childEdge = node->getChildEdgeAt(0);
        const auto& outDims = node->getOutputShapeAtPort(0);

        const void* ext_data_ptr = input->data();
        void* inter_data_ptr = childEdge->getMemory().getData();

        if (ext_data_ptr != inter_data_ptr) {
            auto ext_tensor_desc = create_mem_desc(input);
            Memory ext_mem(getEngine(), ext_tensor_desc, ext_data_ptr, false);
            childEdge->getMemory().load(ext_mem, false);
        }

        // todo: make sure 'name' exists in this map...
        if (_normalizePreprocMap.find(name) != _normalizePreprocMap.end()) {
            if (input->get_element_type() == ov::element::f32) {
                _normalizePreprocMap[name].NormalizeImage(outDims,
                                                          reinterpret_cast<float*>(inter_data_ptr),
                                                          TensorDesc::getLayoutByDims(input->get_shape()));
            } else {
                OPENVINO_THROW("Mean image of type ", input->get_element_type().get_type_name(), " is unsupported");
            }
        }
    } else {
        OPENVINO_THROW("Input blob for infer '", name, "' doesn't correspond to input in network");
    }
}

// suppose always being shared infer_request intel_cpu::Tensor to Graph if isDynamic.
void Graph::PullOutputData(std::unordered_map<std::string, ov::SoPtr<ITensor>>& output) {
    if (!IsReady())
        OPENVINO_THROW("Wrong state. Topology not ready.");

    for (auto &outputMap : outputNodesMap) {
        auto name = outputMap.first;
        auto node = outputMap.second;
        auto parentEdge = node->getParentEdgeAt(0);
        const auto& intr_blob = parentEdge->getMemory();

        const auto ext_blob_map = output.find(name);
        const auto ext_blob = ext_blob_map->second;
        if (ext_blob_map == output.end()) {
            OPENVINO_THROW("The CPU plugin graph doesn't contain output node with name: ", name.c_str());
        }

        InferenceEngine::TensorDesc expectedDesc(
            InferenceEngine::details::convertPrecision(ext_blob->get_element_type()),
            ext_blob->get_shape(),
            InferenceEngine::TensorDesc::getLayoutByRank(ext_blob->get_shape().size()));
        DEBUG_LOG(name, ", tensor data addr ", static_cast<void*>(output[name]->data()));

        const auto actualDesc = MemoryDescUtils::convertToTensorDesc(intr_blob.getDesc());

        // TODO [NM]: need to create universal reorder which will be detect cases when we really need to use it
        // WA: for cases when output shape after transformation will be 1x1x1x1 but model output is scalar
        bool isScalarOutput = false;
        if (actualDesc.getLayout() == SCALAR) {
            isScalarOutput = expectedDesc.getLayout() == SCALAR ||
                             (!expectedDesc.getDims().empty() &&
                             std::accumulate(expectedDesc.getDims().begin(), expectedDesc.getDims().end(), (size_t)1, std::multiplies<size_t>()) == 1);
        } else if (expectedDesc.getLayout() == SCALAR) {
            isScalarOutput = actualDesc.getLayout() == SCALAR ||
                             (!actualDesc.getDims().empty() &&
                             std::accumulate(actualDesc.getDims().begin(), actualDesc.getDims().end(), (size_t)1, std::multiplies<size_t>()) == 1);
        }

        auto outDims = intr_blob.getStaticDims();
        if (ext_blob->get_shape() != outDims && !isScalarOutput) {
            // WA: because input/output info initially contains non empty dims, order etc.
            // and setDims (called inside setShape) can't correct modify blocked desc for desc with blocked layout
            if (expectedDesc.getLayout() == InferenceEngine::Layout::BLOCKED) {
                expectedDesc = TensorDesc(expectedDesc.getPrecision(), expectedDesc.getLayout());
            }
            DEBUG_LOG(name, ", tensor data addr ", static_cast<void*>(output[name]->data()),
            " dims ", PartialShape(output[name]->get_shape()), " -> ", PartialShape(outDims),
            ", intr ptr ", intr_blob.getData(), " , parentedge's memory object ", parentEdge->getMemoryPtr().get());
            ext_blob->set_shape(outDims);
            DEBUG_LOG(name, ", tensor data addr ", static_cast<void*>(output[name]->data()),
            " dims ", PartialShape(output[name]->get_shape()), ", intr ptr ", intr_blob.getData());
            expectedDesc =
                InferenceEngine::TensorDesc(InferenceEngine::details::convertPrecision(ext_blob->get_element_type()),
                                            ext_blob->get_shape(),
                                            InferenceEngine::TensorDesc::getLayoutByRank(ext_blob->get_shape().size()));
        }

        // check for empty output blob
        if (std::any_of(outDims.begin(), outDims.end(), [](const Dim dim) {return dim == 0;})) {
            continue;
        }

        auto srcPrec = actualDesc.getPrecision();
        auto dstPrec = expectedDesc.getPrecision();
        if (!getConfig().isLegacyApi && srcPrec == dstPrec && ext_blob->get_byte_size() != intr_blob.getSize())
            OPENVINO_THROW("Output blob byte size is not equal network output byte size (",
                           ext_blob->get_byte_size(),
                           "!=",
                           intr_blob.getSize(),
                           ").");

        void *ext_blob_ptr = ext_blob->data();
        void *intr_blob_ptr = intr_blob.getData();
        DEBUG_LOG(name, " @ ", intr_blob_ptr, " -> ", ext_blob_ptr, " zero-copy: ", intr_blob_ptr == ext_blob_ptr, " graph ", this, "\r\n");

        // That is the same memory. No need to copy
        if (ext_blob_ptr == intr_blob_ptr) continue;

        if (actualDesc.getBlockingDesc() != expectedDesc.getBlockingDesc() && !isScalarOutput) {
            // User can initialize output via SetOutput API using tensorDesc with ANY layout.
            // For these cases we create planar memory descriptor.
            auto outBlobDesc = expectedDesc.getLayout() == InferenceEngine::Layout::ANY
                                   ? DnnlBlockedMemoryDesc(expectedDesc.getPrecision(), Shape(expectedDesc.getDims()))
                                   : MemoryDescUtils::convertToDnnlBlockedMemoryDesc(expectedDesc);
            Memory outBloMem(getEngine(), outBlobDesc, ext_blob_ptr, false);
            outBloMem.load(intr_blob, false);
        } else {
            size_t size_to_copy = intr_blob.getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
            DEBUG_LOG("pull_output: convert ", srcPrec, " to ", dstPrec);
            cpu_convert(intr_blob_ptr, ext_blob_ptr, srcPrec, dstPrec, size_to_copy);
        }
    }
}

void Graph::InferStatic(SyncInferRequest* request) {
    dnnl::stream stream(getEngine());

    for (const auto& node : executableGraphNodes) {
        VERBOSE(node, getConfig().debugCaps.verbose);
        PERF(node, getConfig().collectPerfCounters);

        if (request)
            request->throw_if_canceled();
        ExecuteNode(node, stream);
    }
}

namespace {

class IUpdateNodes {
public:
    virtual void run(size_t stopIndx) = 0;
    virtual ~IUpdateNodes() = default;
};

class UpdateNodesSeq : public IUpdateNodes {
public:
    explicit UpdateNodesSeq(std::vector<NodePtr>& executableGraphNodes) : m_executableGraphNodes(executableGraphNodes) {}
    void run(size_t stopIndx) override {
        for (; prepareCounter < stopIndx; ++prepareCounter) {
            const auto& node = m_executableGraphNodes[prepareCounter];
            if (node->isDynamicNode()) {
                node->updateShapes();
                node->updateDynamicParams();
            }
        }
    }

private:
    size_t prepareCounter = 0;
    std::vector<NodePtr>& m_executableGraphNodes;
};

#if (OV_THREAD == OV_THREAD_SEQ)
    using UpdateNodes = UpdateNodesSeq;
#endif

#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO || OV_THREAD == OV_THREAD_OMP)
class UpdateNodesBase : public IUpdateNodes {
public:
    explicit UpdateNodesBase(std::vector<NodePtr>& executableGraphNodes) : m_executableGraphNodes(executableGraphNodes) {}
    void updateShapes(size_t node_indx, size_t stop_indx) {
        try {
            for (size_t i = node_indx; i < stop_indx; i++) {
                const auto& node = m_executableGraphNodes[i];
                if (node->isDynamicNode()) {
                    node->updateShapes();
                }
                m_prepareCounter.store(i, std::memory_order::memory_order_release);
            }
        }
        catch(...) {
            m_completion.store(true, std::memory_order::memory_order_relaxed);
            throw;
        }
        m_prepareCounter.store(stop_indx, std::memory_order::memory_order_relaxed);
        m_completion.store(true, std::memory_order::memory_order_release);
    }

    void updateDynParams(size_t node_indx, size_t /*unused*/) {
        size_t local_counter = node_indx;
        while (true) {
            const bool completion = m_completion.load(std::memory_order::memory_order_acquire);
            const size_t prepareCounter = m_prepareCounter.load(std::memory_order::memory_order_relaxed);
            if (completion && local_counter == prepareCounter) {
                break;
            }
            while (local_counter < prepareCounter) {
                const auto& node = m_executableGraphNodes[local_counter++];
                if (node->isDynamicNode()) {
                    node->updateDynamicParams();
                }
            }
        }
    }

protected:
    std::atomic<size_t> m_prepareCounter{0};
    std::atomic<bool> m_completion{false};
    std::vector<NodePtr>& m_executableGraphNodes;
};

#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
#if (TBB_VERSION_MAJOR > 2020)
template <typename Body>
class AsyncTask : public tbb::detail::d1::task {
public:
    AsyncTask(Body& body, tbb::detail::d1::wait_context& wait, size_t node_indx, size_t stop_indx) :
        m_body(body), m_wait(wait), m_node_indx(node_indx), m_stop_indx(stop_indx) {}
    task* execute(tbb::detail::d1::execution_data&) override {
        m_body(m_node_indx, m_stop_indx);
        m_wait.release();
        return nullptr;
    }
    task* cancel(tbb::detail::d1::execution_data&) override {
        m_wait.release();
        return nullptr;
    }

private:
    Body& m_body;
    tbb::detail::d1::wait_context& m_wait;
    size_t m_node_indx;
    size_t m_stop_indx;
};

class UpdateNodes : public UpdateNodesBase {
public:
    using UpdateNodesBase::UpdateNodesBase;
    void run(size_t stopIndx) override {
        m_completion.store(false);
        auto startCounter = m_prepareCounter.load();
        tbb::detail::d1::wait_context wait_ctx(2);

        auto task1 = [this](size_t start, size_t stop) {
            this->updateShapes(start, stop);
        };
        AsyncTask<decltype(task1)> t1(task1, wait_ctx, startCounter, stopIndx);

        auto task2 = [this](size_t start, size_t stop) {
            this->updateDynParams(start, stop);
        };
        AsyncTask<decltype(task2)> t2(task2, wait_ctx, startCounter, stopIndx);

        tbb::detail::d1::spawn(t2, ctx, /* always submit the task to a thread that occupies the first slot */ 1);
        tbb::detail::d1::execute_and_wait(t1, ctx, wait_ctx, ctx);
    }

private:
    tbb::task_group_context ctx;
};
#else
template <typename Body>
class AsyncTask : public tbb::task {
public:
    AsyncTask(Body& body, size_t node_indx, size_t stop_indx) : m_body(body), m_node_indx(node_indx), m_stop_indx(stop_indx) {}
    task* execute() override {
        m_body(m_node_indx, m_stop_indx);
        return nullptr;
    }

private:
    Body& m_body;
    size_t m_node_indx;
    size_t m_stop_indx;
};

class UpdateNodes : public UpdateNodesBase {
public:
    using UpdateNodesBase::UpdateNodesBase;
    void run(size_t stopIndx) override {
        m_completion.store(false);
        auto startCounter = m_prepareCounter.load();
        tbb::task& root = *new(tbb::task::allocate_root()) tbb::empty_task;
        root.set_ref_count(3); // two for children and one preserved

        auto task1 = [this](size_t start, size_t stop) {
            this->updateShapes(start, stop);
        };
        AsyncTask<decltype(task1)>& a = *new (root.allocate_child()) AsyncTask<decltype(task1)>(task1, startCounter, stopIndx);

        auto task2 = [this](size_t start, size_t stop) {
            this->updateDynParams(start, stop);
        };
        AsyncTask<decltype(task2)>& b = *new (root.allocate_child()) AsyncTask<decltype(task2)>(task2, startCounter, stopIndx);

        b.set_affinity(2); // slot 1 plus 1
        tbb::task::spawn(b);
        root.spawn_and_wait_for_all(a);
    }
};
#endif
#endif

#if (OV_THREAD == OV_THREAD_OMP)
class UpdateNodes : public UpdateNodesBase {
public:
    using UpdateNodesBase::UpdateNodesBase;
    void run(size_t stopIndx) override {
        m_completion.store(false);
        auto startCounter = m_prepareCounter.load();

        #pragma omp parallel
        #pragma omp sections
        {
            #pragma omp section
            {
                updateDynParams(startCounter, stopIndx);
            }
            #pragma omp section
            {
                updateShapes(startCounter, stopIndx);
            }
        }
    }
};
#endif

#endif
} // namespace


void Graph::InferDynamic(SyncInferRequest* request) {
    dnnl::stream stream(getEngine());

    std::set<size_t> syncIndsWorkSet;
    for (const auto& nodeIndx : syncNodesInds) {
        syncIndsWorkSet.insert(nodeIndx.second);
        //since sometimes we need to run the synchronization node  alone (for example in the case of internal dynamism)
        //let's add another sync index after the sync point node
        syncIndsWorkSet.insert(nodeIndx.second + 1);
    }
    syncIndsWorkSet.insert(executableGraphNodes.size());

    std::unique_ptr<IUpdateNodes> updateNodes{};
    if (parallel_get_max_threads() > 1) {
        updateNodes.reset(new UpdateNodes(executableGraphNodes));
    } else {
        updateNodes.reset(new UpdateNodesSeq(executableGraphNodes));
    }
    size_t inferCounter = 0;

    for (auto stopIndx : syncIndsWorkSet) {
        updateNodes->run(stopIndx);
        for (; inferCounter < stopIndx; ++inferCounter) {
            auto& node = executableGraphNodes[inferCounter];
            VERBOSE(node, getConfig().debugCaps.verbose);
            PERF(node, getConfig().collectPerfCounters);

            if (request)
                request->throw_if_canceled();
            ExecuteNode(node, stream);
        }
    }
}

inline void Graph::ExecuteNode(const NodePtr& node, const dnnl::stream& stream) const {
    DUMP(node, getConfig().debugCaps, infer_count);

    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, node->profiling.execute);
    DEBUG_LOG(*node);
    if (node->isDynamicNode()) {
        node->executeDynamic(stream);
    } else {
        node->execute(stream);
    }
}

void Graph::Infer(SyncInferRequest* request) {
    if (!IsReady()) {
        OPENVINO_THROW("Wrong state of the ov::intel_cpu::Graph. Topology is not ready.");
    }

    if (Status::ReadyDynamic == status) {
        InferDynamic(request);
    } else if (Status::ReadyStatic == status) {
        InferStatic(request);
    } else {
        OPENVINO_THROW("Unknown ov::intel_cpu::Graph state: " , static_cast<size_t>(status));
    }

    if (infer_count != -1) infer_count++;
}

void Graph::VisitNode(NodePtr node, std::vector<NodePtr>& sortedNodes) {
    if (node->temporary) {
        return;
    }

    if (node->permanent) {
        return;
    }

    node->temporary = true;

    for (size_t i = 0; i < node->getChildEdges().size(); i++) {
        VisitNode(node->getChildEdgeAt(i)->getChild(), sortedNodes);
    }

    node->permanent = true;
    node->temporary = false;

    sortedNodes.insert(sortedNodes.begin(), node);
}

void Graph::SortTopologically() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::SortTopologically");

    std::vector<NodePtr> unsorted;
    std::vector<NodePtr> sorted;

    for (size_t i = 0; i < graphNodes.size(); i++) {
        NodePtr node = graphNodes[i];

        node->permanent = false;
        node->temporary = false;

        unsorted.push_back(node);
    }

    while (!unsorted.empty()) {
        NodePtr node = unsorted.at(0);
        unsorted.erase(unsorted.begin());

        VisitNode(node, sorted);
    }

    for (size_t i = 0; i < sorted.size(); i++)
        sorted[i]->execIndex = static_cast<int>(i);

    graphNodes.erase(graphNodes.begin(), graphNodes.end());
    graphNodes.assign(sorted.begin(), sorted.end());

    // TODO: Sort in/out edges by port index because of backward compatibility
    //       A lot of plugin logic are build on top of assumption that index in
    //       vector childEdges/parentEdges is port number. But that is not
    //       truth anymore. But to keep old logic correct need to simulate ordering.
    //
    // Make first N (N == port_num) edge indexes are matched with port index
    for (auto &node : graphNodes) {
        {
            int port_num = node->inputShapes.size();
            std::vector<EdgePtr> res(port_num);

            for (size_t i = 0; i < node->parentEdges.size(); i++) {
                auto edge = node->getParentEdgeAt(i);
                int port = edge->getOutputNum();
                if (port < port_num && !res[port])
                    res[port] = edge;
                else
                    res.push_back(edge);
            }
            node->parentEdges = {res.begin(), res.end()};
        }
        {
            int port_num = node->outputShapes.size();
            std::vector<EdgePtr> res(port_num);

            for (size_t i = 0; i < node->childEdges.size(); i++) {
                auto edge = node->getChildEdgeAt(i);
                int port = edge->getInputNum();
                if (port < port_num && !res[port])
                    res[port] = edge;
                else
                    res.push_back(edge);
            }
            node->childEdges = {res.begin(), res.end()};
        }
    }
}

void Graph::GetPerfData(std::vector<ov::ProfilingInfo>& perfMap) const {
    std::function<void(std::vector<ov::ProfilingInfo>&, const NodePtr&)> getPerfMapFor =
        [&](std::vector<ov::ProfilingInfo>& perfMap, const NodePtr& node) {
            ov::ProfilingInfo pc;
            pc.node_name = node->getName();
            // pc.execution_index = i++;
            uint64_t avg_time = node->PerfCounter().avg();
            pc.cpu_time = pc.real_time = std::chrono::microseconds(avg_time);
            pc.status = avg_time > 0 ? ov::ProfilingInfo::Status::EXECUTED : ov::ProfilingInfo::Status::NOT_RUN;
            pc.exec_type = node->getPrimitiveDescriptorType();
            pc.node_type = node->typeStr;
            perfMap.emplace_back(pc);

            for (auto& fusedNode : node->fusedWith) {
                getPerfMapFor(perfMap, fusedNode);
            }

            for (auto& mergedWith : node->mergedWith) {
                getPerfMapFor(perfMap, mergedWith);
            }
        };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        if (graphNodes[i]->isConstant())
            continue;
        getPerfMapFor(perfMap, graphNodes[i]);
    }
}

void Graph::RemoveEdge(EdgePtr& edge) {
    for (auto it = graphEdges.begin(); it != graphEdges.end(); it++) {
        if ((*it) == edge) {
            edge->drop();
            graphEdges.erase(it);
            return;
        }
    }
}

void Graph::DropNode(const NodePtr &node) {
    auto children = node->childEdges;
    auto parents = node->parentEdges;

    for (size_t i = 0; i < parents.size(); i++) {
        auto p_edge = parents[i].lock();
        if (!p_edge) continue;
        auto parent = p_edge->getParent();
        if (!parent) continue;

        const int inNum = p_edge->getInputNum();
        p_edge->drop();
        RemoveEdge(p_edge);

        for (size_t j = 0; j < children.size(); j++) {
            auto c_edge = children[j].lock();
            if (!c_edge) continue;
            auto child = c_edge->getChild();
            if (!child) continue;

            const int outNum = c_edge->getOutputNum();
            c_edge->drop();
            RemoveEdge(c_edge);

            EdgePtr newEdge(new Edge(parent, child, inNum, outNum));
            graphEdges.push_back(newEdge);
            parent->addEdge(newEdge);
        }
    }
}

void Graph::DropDWConvNode(const NodePtr &node) {
    auto children = node->childEdges;
    auto parents = node->parentEdges;

    auto parentConvEdge = parents[0].lock();
    if (!parentConvEdge) return;
    auto parentConv = parentConvEdge->getParent();
    if (!parentConv) return;

    parentConv->outputShapes[0] = node->outputShapes[0];

    for (size_t i = 0; i < 1; i++) {
        auto p_edge = parents[i].lock();
        if (!p_edge) continue;
        auto parent = p_edge->getParent();
        if (!parent) continue;

        const int inNum = p_edge->getInputNum();
        p_edge->drop();
        RemoveEdge(p_edge);

        for (size_t j = 0; j < children.size(); j++) {
            auto c_edge = children[j].lock();
            if (!c_edge) continue;
            auto child = c_edge->getChild();
            if (!child) continue;

            const int outNum = c_edge->getOutputNum();
            c_edge->drop();
            RemoveEdge(c_edge);

            EdgePtr newEdge(new Edge(parent, child, inNum, outNum));
            graphEdges.push_back(newEdge);
            parent->addEdge(newEdge);
        }
    }

    for (size_t i = 1; i < parents.size(); i++) {
        auto p_edge = parents[i].lock();
        if (!p_edge) continue;
        auto parent = p_edge->getParent();
        if (!parent) continue;

        const int inNum = p_edge->getInputNum();
        const int portCandidate = p_edge->getOutputNum();
        p_edge->drop();
        RemoveEdge(p_edge);
        const int outNum = parentConv->parentEdges.size();

        EdgePtr newEdge(new Edge(parent, parentConv, inNum, outNum));
        graphEdges.push_back(newEdge);
        parent->addEdge(newEdge);
        parentConv->inputShapes.push_back(node->getInputShapeAtPort(portCandidate));
    }
    parentConv->outputShapes[0] = node->getOutputShapeAtPort(0);
}

void Graph::RemoveDroppedNodes() {
    auto& nodes = this->GetNodes();

    auto it = nodes.begin();

    while (it != nodes.end()) {
        if ((*it)->isDropped()) {
            it = nodes.erase(it);
        } else {
            it++;
        }
    }
}

void Graph::RemoveDroppedEdges() {
    auto& edges = this->GetEdges();

    auto it = edges.begin();

    while (it != edges.end()) {
        if ((*it)->isDropped()) {
            it = edges.erase(it);
        } else {
            it++;
        }
    }
}

NodePtr Graph::InsertReorder(EdgePtr edge, std::string layerName, const MemoryDesc& inDesc, const MemoryDesc& outDesc,
                                         bool isOptimized, const std::vector<int> & src_perm) {
    NodePtr newReorder(new node::Reorder(layerName, context));
    auto *reorderPtr = dynamic_cast<node::Reorder *>(newReorder.get());
    if (reorderPtr == nullptr) {
        OPENVINO_THROW("Graph::InsertReorder: Cannot cast to Reorder");
    }
    reorderPtr->setDescs(inDesc, outDesc);
    reorderPtr->setOptimized(isOptimized);
    reorderPtr->setSrcPermutation(src_perm);

    DEBUG_LOG(reorderPtr->getName(), " edge=", edge->name(), " isOptimized=", isOptimized);
    DEBUG_LOG("    inDesc: ", inDesc.getShape().toString(), inDesc.getPrecision().name(), " ", inDesc.serializeFormat());
    DEBUG_LOG("   outDesc: ", outDesc.getShape().toString(), outDesc.getPrecision().name(), " ", outDesc.serializeFormat());

    InsertNode(edge, newReorder, true);

    // Using the method Edge::getDesc() we can check that input and output tensor descriptors are equal.
    // Due to the specificity of GraphOptimizer::MergeTransposeAndReorder() that isOptimized flag uses, we shouldn't do these checks.
    if (!isOptimized) {
        newReorder->getParentEdgeAt(0)->getDesc();
        newReorder->getChildEdgeAt(0)->getDesc();
    }

    return newReorder;
}

bool Graph::InsertNode(EdgePtr edge, NodePtr node, bool initNode) {
    auto oIndex = edge->getOutputNum();
    auto iIndex = edge->getInputNum();
    if (iIndex < 0 || oIndex < 0)
        OPENVINO_THROW("Cannot insert node '",
                       node->getName(),
                       "' between nodes: ",
                       edge->getParent()->getName(),
                       " and ",
                       edge->getChild()->getName(),
                       ".");
    edge->drop();

    return InsertNode(edge->getParent(), edge->getChild(), node, iIndex, oIndex, initNode);
}

bool Graph::InsertNode(NodePtr parent, NodePtr child, NodePtr node, int parentPort, int childPort, bool initNode) {
    EdgePtr beforeNode(new Edge(parent, node, parentPort, 0));
    EdgePtr afterNode(new Edge(node, child, 0, childPort));

    // Add edge for beforeNode
    beforeNode->getChild()->parentEdges.push_back(beforeNode);
    parent->childEdges.push_back(beforeNode);

    // Add edge for afterNode
    afterNode->getParent()->childEdges.push_back(afterNode);
    child->parentEdges.push_back(afterNode);

    if (initNode) {
        node->getSupportedDescriptors();
        node->initSupportedPrimitiveDescriptors();
        node->filterSupportedPrimitiveDescriptors();
        node->selectOptimalPrimitiveDescriptor();
        resolveInPlaceDirection(node);
        node->initOptimalPrimitiveDescriptor();
    }

    graphEdges.push_back(beforeNode);
    graphEdges.push_back(afterNode);
    graphNodes.push_back(node);
    return true;
}

// Apply inference precision configuration
void Graph::EnforceInferencePrecision() {
    CPU_DEBUG_CAP_ENABLE(static EnforceInferPrcDebug inferPrecDebug);

    const auto inferPrec = convertPrecision(getConfig().inferencePrecision);

    if (inferPrec == Precision::FP32)
        return; // nothing to do, only precision reduction is currently allowed
#if defined(OV_CPU_ARM_ENABLE_FP16)
    if (inferPrec == Precision::FP16)
        return; // precision of configured by ov::pass::ConvertPrecision
#endif
    std::function<void(const NodePtr&, std::unordered_set<NodePtr>& skipNodes)> searchForNodesToSkip;
    searchForNodesToSkip = [&](const NodePtr& node, std::unordered_set<NodePtr>& skipNodes) -> void {
        for (size_t i = 0; i < node->getParentEdges().size(); i++) {
            const auto& parent = node->getParentEdgeAt(i)->getParent();

            if (inferPrec == InferenceEngine::Precision::BF16) {
                /* list of node types that must be forced to be executed in BF16 precision
                * because of performance gains */
                if (one_of(parent->getType(),
                        Type::Convolution,    // conv nets
                        Type::FullyConnected, // conv / bert nets
                        Type::RNNCell,        // recurent nets
                        Type::RNNSeq,         // recurent nets
                        Type::MatMul,         // bert nets
                        Type::ROIPooling,     // object detection nets
                        Type::Interpolate))    // super resolution nets
                    continue;   // stop at significant nodes
            } else if (inferPrec == InferenceEngine::Precision::FP16) {
                /* list of node types that must be forced to be executed in FP16 precision
                * because of performance gains */
                if (one_of(parent->getType(),
                        Type::Convolution,    // conv nets
                        Type::Deconvolution,  // deconv
                        Type::FullyConnected, // conv / bert nets
                        Type::MatMul,         // bert nets
                        Type::Pooling,
                        Type::MVN))
                    continue;   // stop at significant nodes
            }

            const auto res = skipNodes.insert(parent);
            if (res.second) // node not visited yet
                searchForNodesToSkip(parent, skipNodes);
        }
    };

    /* Skip low-precision float point enforcement for tail of the graph by forming set of nodes to skip.
     * Necessary to maintain accuracy.
     * Experiments show zero peformance impact on average */
    std::unordered_set<NodePtr> nodesToSkip;
    // starting from output nodes
    for (const auto& entry : outputNodesMap) {
        const auto& output = entry.second;
        // do not skip outputs which precisions are explicitly set equal to inferPrec
        if (output->getOriginalInputPrecisionAtPort(0) == inferPrec)
            continue;

        searchForNodesToSkip(output, nodesToSkip);
    }

    for (const auto& node : graphNodes) {
        if (nodesToSkip.count(node) && !node->enforceBF16evenForGraphTail)
            continue;

        if (one_of(node->getType(), Type::Input, Type::Output))
            continue;

#ifdef CPU_DEBUG_CAPS
        if (!inferPrecDebug.enabled(NameFromType(node->getType()), node->getName()))
            continue;
#endif
        DEBUG_LOG("#", node->getExecIndex(), " ", node->getName(), " is enforced to use", inferPrec);

        for (size_t i = 0; i < node->getOriginalInputsNumber(); i++) {
            auto keepOriginalInputPrecisionAtPort = [](const NodePtr& node, const size_t inPort) {
                // keep non-float precisions
                if (node->getOriginalInputPrecisionAtPort(inPort) != Precision::FP32)
                    return true;

                const auto &parent = node->getParentEdgesAtPort(inPort)[0]->getParent();
                /* Skip BF16 enforcement for nodes after Constant Inputs for maintaining precision for fusing.
                 * Precision conversion to BF16 is done automatically, if convolution follows up after Constant Inputs
                 * and activation is BF16 */
                if (parent->getType() == Type::Input && parent->isConstant() &&
                    // Concatenation node is exception because it doesn't change an accuracy for BF16 activation
                    node->getType() != Type::Concatenation)
                    return true;
                // Eltwise and Subgraph (snippets) nodes support precision conversion
                if (parent->getType() == Type::Input && one_of(node->getType(), Type::Eltwise, Type::Subgraph))
                    return true;

                return false;
            };

            if (keepOriginalInputPrecisionAtPort(node, i))
                continue;

            node->setOriginalInputPrecisionAtPort(i, inferPrec);
        }

        for (size_t i = 0; i < node->getOriginalOutputsNumber(); i++) {
            // keep non-float precisions
            if (node->getOriginalOutputPrecisionAtPort(i) != Precision::FP32)
                continue;

            // exclude Convert before Range since it may cause precision loss when integter type to LP.
            // TODO: Incorrect subgraph is generated by ONNX FE + ticket 117861.
            const auto &child = node->getChildEdgesAtPort(i)[0]->getChild();
            if (child->getType() == Type::Range && node->getType() == Type::Convert)
                continue;

            node->setOriginalOutputPrecisionAtPort(i, inferPrec);
        }
    }
}

std::shared_ptr<ov::Model> Graph::dump() const {
    return dump_graph_as_ie_ngraph_net(*this);
}

void Graph::resolveInPlaceDirection(const NodePtr& node) const {
    enum InplaceDirectionType {UP, DOWN, CYCLIC, NONE};
    enum PortType {INPUT, OUTPUT};

    auto inPlaceDirection = [](const NodePtr& node, PortType portType, int portNum) -> InplaceDirectionType {
        if (PortType::INPUT == portType) {
            auto inPlaceInpPort = node->inPlaceInputPort(portNum);
            if (inPlaceInpPort >= 0) {
                auto inPlaceOutPort = node->inPlaceOutPort(inPlaceInpPort);
                if (inPlaceOutPort == inPlaceInpPort) {
                    return InplaceDirectionType::CYCLIC;
                } else if (inPlaceOutPort < 0) {
                    return InplaceDirectionType::DOWN;
                } else {
                    OPENVINO_THROW("Non trivial inPlace memory dependency has been detected");
                }
            }
            // the requested port has a negative inPlace tag, let's check whether it is referenced from the output
            auto& config = node->getSelectedPrimitiveDescriptor()->getConfig();
            for (auto& portConf : config.outConfs) {
                if (portConf.inPlace() == portNum) {
                    return InplaceDirectionType::UP;
                }
            }
        } else if (PortType::OUTPUT == portType) {
            auto inPlaceOutPort = node->inPlaceOutPort(portNum);
            if (inPlaceOutPort >= 0) {
                auto inPlaceInpPort = node->inPlaceInputPort(inPlaceOutPort);
                if (inPlaceOutPort == inPlaceInpPort) {
                    return InplaceDirectionType::CYCLIC;
                } else if (inPlaceInpPort < 0) {
                    return InplaceDirectionType::UP;
                } else {
                    OPENVINO_THROW("Non trivial inPlace memory dependency has been detected");
                }
            }
            // the requested port has a negative inPlace tag, let's check whether it is referenced from the input
            auto& config = node->getSelectedPrimitiveDescriptor()->getConfig();
            for (auto& portConf : config.inConfs) {
                if (portConf.inPlace() == portNum) {
                    return InplaceDirectionType::DOWN;
                }
            }
        }
        return InplaceDirectionType::NONE;
    };

    auto& inpEdges = node->getParentEdges();
    for (auto& wEdge : inpEdges) {
        if (auto pEdge = wEdge.lock()) {
            auto inpPort = pEdge->getOutputNum();
            auto inPlaceInpPort = node->inPlaceInputPort(inpPort);
            if (inPlaceInpPort < 0 || inPlaceDirection(node, PortType::INPUT, inpPort) != InplaceDirectionType::CYCLIC) {
                continue;
            }
            // inPlace memory cyclic dependency detected, need to resolve
            // let's check the parent node first
            auto pParent = pEdge->getParent();
            auto parentInPlaceDirection = inPlaceDirection(pParent, PortType::OUTPUT, pEdge->getInputNum());
            if (parentInPlaceDirection == InplaceDirectionType::UP) {
                auto config = node->getSelectedPrimitiveDescriptor()->getConfig();
                config.inConfs[inpPort].inPlace(-1);
                node->initDescriptor(config);
            } else if (parentInPlaceDirection == InplaceDirectionType::DOWN) {
                //search if siblings already have downstream direction
                auto downstreamPeers = [&] {
                    for (auto& peerEdge : pParent->getChildEdgesAtPort(pEdge->getInputNum())) {
                        auto peerNode = peerEdge->getChild();
                        if (peerNode == node) continue;
                        if (inPlaceDirection(peerNode, PortType::INPUT, peerEdge->getOutputNum()) == InplaceDirectionType::DOWN) {
                            return true;
                        }
                    }
                    return false;
                }();
                if (downstreamPeers) {
                    // when there is an downstream peer we have to resolve upstream inplace for the node
                    // to avoid inplace conflict
                    auto config = node->getSelectedPrimitiveDescriptor()->getConfig();
                    config.inConfs[inpPort].inPlace(-1);
                    node->initDescriptor(config);
                } else {
                    auto config = node->getSelectedPrimitiveDescriptor()->getConfig();
                    config.outConfs[inPlaceInpPort].inPlace(-1);
                    node->initDescriptor(config);
                }
            } else {
                // the parent node does not use inPlace memory, let's check children
                std::function<InplaceDirectionType(const NodePtr& node, int portIdx)> searchNonCyclicDirection;
                searchNonCyclicDirection = [&](const NodePtr& node, int portIdx) -> InplaceDirectionType {
                    auto& childEdges = node->getChildEdgesAtPort(portIdx);
                    for (auto& edge : childEdges) {
                        auto pChild = edge->getChild();
                        auto result = inPlaceDirection(pChild, PortType::INPUT, edge->getOutputNum());
                        if (InplaceDirectionType::UP == result || InplaceDirectionType::DOWN == result) {
                            return result;
                        } else if (InplaceDirectionType::CYCLIC == result) {
                            return searchNonCyclicDirection(pChild, pChild->inPlaceInputPort(edge->getOutputNum()));
                        }
                    }
                    return InplaceDirectionType::NONE;
                };
                auto result = searchNonCyclicDirection(node, inPlaceInpPort);
                if (one_of(result, InplaceDirectionType::UP, InplaceDirectionType::NONE)) {
                    auto config = node->getSelectedPrimitiveDescriptor()->getConfig();
                    config.inConfs[inpPort].inPlace(-1);
                    node->initDescriptor(config);
                } else if (InplaceDirectionType::DOWN == result) {
                    auto config = node->getSelectedPrimitiveDescriptor()->getConfig();
                    config.outConfs[inPlaceInpPort].inPlace(-1);
                    node->initDescriptor(config);
                } else {
                    OPENVINO_THROW("A node without an inPlace memory cyclic dependency has not been found");
                }
            }
        }
    }
}

void Graph::SearchInternalStateNodes() {
    for (auto&& node : graphNodes) {
        if (node->getType() == Type::MemoryInput) {
            auto cur_node = std::dynamic_pointer_cast<node::MemoryInput>(node);
            if (!cur_node) {
                OPENVINO_THROW("Cannot cast ", node->getName(), " to MemoryInput");
            }
            internalStateNodes.insert({cur_node->getId(), cur_node});
        }
    }
}

}   // namespace intel_cpu
}   // namespace ov
