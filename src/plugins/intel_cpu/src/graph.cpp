// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "edge.h"
#include "graph_dumper.h"
#include "graph_optimizer.h"
#include "infer_request.h"
#include "itt.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "node.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/convert.h"
#include "nodes/input.h"
#include "nodes/reorder.h"
#include "nodes/memory.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"
#include "utils/node_dumper.h"
#include "utils/verbose.h"

#include <oneapi/dnnl/dnnl.hpp>
#if defined(OV_CPU_ARM_ENABLE_FP16)
#include "common/primitive_desc_iface.hpp"
#endif

#include "openvino/runtime/memory_solver.hpp"

#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
#    include <tbb/task.h>
#endif

using namespace dnnl;
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

    for (const auto& op : model->get_ordered_ops()) {
        const NodePtr node {Node::factory().create(op, context)};

        AddNode(node);
        if (op->get_type_info() == op::v0::Parameter::get_type_info_static()) {
            const std::string name = get_port_name(ov::Output<ov::Node>(op, 0));
            inputNodesMap[name] = node;
            if (node->isDynamicNode()) {
                graphHasDynamicInput = true;
            }
        }

        if (op->get_type_info() == op::v0::Result::get_type_info_static()) {
            const std::string inputID = get_port_name(op->output(0));
            outputNodesMap[inputID] = node;
        }

        op2node[op] = node;

        for (size_t port = 0; port < op->get_input_size(); port++) {
            auto parentOp = op->get_input_node_shared_ptr(port);
            auto parentNode = op2node[parentOp];

            CreateEdge(parentNode, node, getParentOutputPort(op, parentOp, port), static_cast<int>(port));
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
        CreateEdge(parentNode, outNode, port, 0);
        AddNode(outNode);
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
            if (!one_of(child_prec, ov::element::bf16, ov::element::f16) &&
                // remove this WA when #78939 is resolved
                !hasSubgraphConsumers(child))
                child->setOriginalInputPrecisionAtPort(childEdges[i]->getOutputNum(), precToSet);
        }
    }

    for (auto &output : outputNodesMap) {
        const auto& outputNode = output.second;
        const auto precToSet = outputNode->getOriginalInputPrecisionAtPort(0);
        const auto parentEdge = outputNode->getParentEdgeAt(0);
        const auto parent = parentEdge->getParent();
        parent->setOriginalOutputPrecisionAtPort(parentEdge->getInputNum(), precToSet);
    }
}

void Graph::InitGraph(bool optimize) {
    DEBUG_LOG("Initializing graph with name: ",  GetName());

    GraphOptimizer optimizer;

    SortTopologically();
    InitNodes();

    optimizer.ApplyCommonGraphOptimizations(*this);
    SortTopologically();

    InitDescriptors();

    ResolveInplaceDirections();

    InitOptimalPrimitiveDescriptors();

    ResolveEdgeConflicts();

    optimizer.ShareReorders(*this);
    RemoveDroppedNodes();

    ResolveComplexInplaceConflicts();

    optimizer.ApplyImplSpecificGraphOptimizations(*this);
    SortTopologically();

    const auto hasDynNodes = ProcessDynNodes();

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

    CPU_DEBUG_CAP_ENABLE(serialize(*this));
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
        node->resolveInPlaceDirection();
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

void Graph::insertReorder(EdgePtr& edge, bool isOptimized, std::unordered_set<std::string>& uniqueLayerNames) {
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
}

void Graph::ResolveEdgeConflicts() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::ResolveEdgeConflicts");

    ptrdiff_t numberOfEdges = static_cast<ptrdiff_t>(graphEdges.size());

    std::unordered_set<std::string> uniqueLayerNames;
    for (auto node : graphNodes) {
        uniqueLayerNames.insert(node->getName());
    }

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
                                          inDesc.getPrecision().get_type_name() + "_" + outDesc.getPrecision().get_type_name();

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
                insertReorder(edge, reorderStatusInternal == Edge::ReorderStatus::Optimized, uniqueLayerNames);
            }
            updateEdge(i);
        } else if (reorderStatus == Edge::ReorderStatus::Optimized) {
            insertReorder(edge, true, uniqueLayerNames);
            updateEdge(i);
        }
    }
}

void Graph::ResolveComplexInplaceConflicts() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::ResolveComplexInplaceConflicts");

    ptrdiff_t numberOfEdges = static_cast<ptrdiff_t>(graphEdges.size());

    std::unordered_set<std::string> uniqueLayerNames;
    for (auto node : graphNodes) {
        uniqueLayerNames.insert(node->getName());
    }

    auto updateEdge = [&](ptrdiff_t& i) {
        graphEdges.erase(graphEdges.begin() + i);
        i--;
        numberOfEdges--;
    };

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

    for (ptrdiff_t i = 0; i < numberOfEdges; i++) {
        auto edge = graphEdges[i];
        if (needReorder(edge)) {
            insertReorder(edge, false, uniqueLayerNames);
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

            // Special allocation for string tensors
            if (edge->getDesc().getPrecision() == element::string && edge->getStatus() == Edge::Status::NeedAllocation) {
                StringMemory::StringMemoryMngrPtr mngr;
                if (edge->getParent()->isConstant()) {
                    if (edge->getParent()->getType() == Type::Input) {
                        auto constNode = static_cast<node::Input *>(edge->getParent().get());
                        edge->reuse(std::const_pointer_cast<IMemory>(constNode->getMemoryPtr()));
                    } else {
                        edge->externalAllocate(context->getWeightsCache());
                    }
                    auto stringMemory = dynamic_cast<StringMemory *>(edge->getMemoryPtr().get());
                    OPENVINO_ASSERT(stringMemory, "[CPU] Edge between nodes '",
                            edge->getParent()->getName(), "' and '", edge->getChild()->getName(), "' must have StringMemory.");
                    mngr = stringMemory->getStringMemoryMngrPtr();
                } else {
                    auto memory = std::make_shared<StringMemory>(getEngine(), edge->getDesc());
                    edge->reuse(memory);
                    mngr = memory->getStringMemoryMngrPtr();
                }
                for (auto& edge_c : cluster) {
                    if (edge_c == edge) {
                        continue;
                    }
                    OPENVINO_ASSERT(edge_c->getDesc().getPrecision() == element::string, "All edges in the cluster must be string.");
                    if (edge_c->getStatus() == Edge::Status::NotAllocated) {
                        auto memory = std::make_shared<StringMemory>(getEngine(), edge_c->getDesc(), mngr);
                        edge_c->reuse(memory);
                    } else {
                        OPENVINO_THROW("[CPU] String tensors allocation in the cluster. Edge between nodes '", edge_c->getParent()->getName(), "' and '",
                            edge_c->getChild()->getName(), "' has an unexpected status: ", static_cast<int>(edge_c->getStatus()));
                    }
                }
                erase = true;
                continue;
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
    std::vector<ov::MemorySolver::Box> definedBoxes;
    std::vector<ov::MemorySolver::Box> undefinedBoxes;
    for (size_t i = 0; i < remaining_edge_clusters_count; i++) {
        ov::MemorySolver::Box box = { std::numeric_limits<int>::max(), 0, 0, static_cast<int64_t>(i) };
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
    ov::MemorySolver staticMemSolver(definedBoxes);
    size_t total_size = static_cast<size_t>(staticMemSolver.solve()) * alignment;

    memWorkspace = std::make_shared<Memory>(getEngine(), DnnlBlockedMemoryDesc(ov::element::i8, Shape(VectorDims{total_size})));

    if (edge_clusters.empty())
        return;

    auto* workspace_ptr = static_cast<int8_t*>(memWorkspace->getData());

    for (const auto& box : definedBoxes) {
        int count = 0;
        for (auto& edge : edge_clusters[box.id]) {
            if (edge->getStatus() == Edge::Status::NeedAllocation) {
                int64_t offset = staticMemSolver.get_offset(box.id);
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

        ov::MemorySolver::normalize_boxes(undefinedBoxes);

        std::vector<std::vector<ov::MemorySolver::Box>> groups; //groups of nonoverlapping boxes
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
            if (edge->getParent()->getParentEdges().empty() &&
                one_of(edge->getParent()->getType(), Type::Input, Type::MemoryInput) &&
                edge->inPlace(Edge::LOOK_UP)) {
                edge->getParent()->resolveInPlaceEdges(Edge::LOOK_UP);
            } else if (edge->getChild()->getChildEdges().empty() &&
                one_of(edge->getChild()->getType(), Type::Output, Type::MemoryOutput) &&
                edge->inPlace(Edge::LOOK_DOWN)) {
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
        auto node = input_itr->second;
        auto childEdge = node->getChildEdgeAt(0);
        auto edgeMemory = childEdge->getMemoryPtr();

        const void* ext_data_ptr = input->data();
        void* inter_data_ptr = edgeMemory->getData();

        if (ext_data_ptr != inter_data_ptr) {
            auto ext_tensor_desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(input);
            auto actualDesc = edgeMemory->getDescPtr();

            if (actualDesc->getPrecision() == element::string) {
                StringMemory ext_mem(getEngine(), ext_tensor_desc, ext_data_ptr);
                edgeMemory->load(ext_mem);
            } else if (!actualDesc->isCompatible(*ext_tensor_desc)) {
                Memory ext_mem(getEngine(), ext_tensor_desc, ext_data_ptr, false);
                edgeMemory->load(ext_mem, false);
            } else {
                size_t size_to_copy = ext_tensor_desc->getCurrentMemSize();
                cpu_parallel_memcpy(inter_data_ptr, ext_data_ptr, size_to_copy);
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

        auto expected_desc_ptr = MemoryDescUtils::generateCpuBlockedMemoryDesc(ext_blob);
        const auto actualDesc = intr_blob.getDescWithType<BlockedMemoryDesc>();

        DEBUG_LOG(name, ", tensor data addr ", static_cast<void*>(output[name]->data()));

        // TODO [NM]: need to create universal reorder which will be detect cases when we really need to use it
        // WA: for cases when output shape after transformation will be 1x1x1x1 but model output is scalar
        bool isScalarOutput = false;
        if (ext_blob->get_shape().empty() && ext_blob->get_size() == 1) {
            const auto& actualDims = expected_desc_ptr->getShape().getStaticDims();
            isScalarOutput =
                !actualDims.empty() &&
                std::accumulate(actualDims.begin(), actualDims.end(), (size_t)1, std::multiplies<size_t>()) == 1;
        }

        auto outDims = intr_blob.getStaticDims();
        if (ext_blob->get_shape() != outDims && !isScalarOutput) {
            // WA: because input/output info initially contains non empty dims, order etc.
            // and setDims (called inside setShape) can't correct modify blocked desc for desc with blocked layout
            DEBUG_LOG(name, ", tensor data addr ", static_cast<void*>(output[name]->data()),
            " dims ", PartialShape(output[name]->get_shape()), " -> ", PartialShape(outDims),
            ", intr ptr ", intr_blob.getData(), " , parentedge's memory object ", parentEdge->getMemoryPtr().get());
            ext_blob->set_shape(outDims);
            DEBUG_LOG(name, ", tensor data addr ", static_cast<void*>(output[name]->data()),
            " dims ", PartialShape(output[name]->get_shape()), ", intr ptr ", intr_blob.getData());
            expected_desc_ptr = MemoryDescUtils::generateCpuBlockedMemoryDesc(ext_blob);
        }

        // check for empty output blob
        if (std::any_of(outDims.begin(), outDims.end(), [](const Dim dim) {return dim == 0;})) {
            continue;
        }

        auto srcPrec = actualDesc->getPrecision();
        auto dstPrec = expected_desc_ptr->getPrecision();
        if (srcPrec == dstPrec && ext_blob->get_byte_size() != intr_blob.getSize())
            OPENVINO_THROW("Output tensor byte size is not equal model output byte size (",
                           ext_blob->get_byte_size(),
                           "!=",
                           intr_blob.getSize(),
                           ").");

        void *ext_blob_ptr = ext_blob->data();
        void *intr_blob_ptr = intr_blob.getData();
        DEBUG_LOG(name, " @ ", intr_blob_ptr, " -> ", ext_blob_ptr, " zero-copy: ", intr_blob_ptr == ext_blob_ptr, " graph ", this, "\r\n");

        // That is the same memory. No need to copy
        if (ext_blob_ptr == intr_blob_ptr) continue;

        if (actualDesc->getPrecision() == element::string) {
            StringMemory outBloMem(getEngine(), expected_desc_ptr, ext_blob_ptr);
            outBloMem.load(intr_blob);
        } else if (!actualDesc->isCompatible(*expected_desc_ptr) && !isScalarOutput) {
            Memory outBloMem(getEngine(), expected_desc_ptr, ext_blob_ptr, false);
            outBloMem.load(intr_blob, false);
        } else {
            OPENVINO_ASSERT(srcPrec == dstPrec, "The precision of the CPU output tensor ", name, " is different from the external one");
            size_t size_to_copy = intr_blob.getSize();
            cpu_parallel_memcpy(ext_blob_ptr, intr_blob_ptr, size_to_copy);
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

#    if (defined(_MSVC_LANG) && (_MSVC_LANG > 201703L)) || (defined(__cplusplus) && (__cplusplus > 201703L))
#        define ov_memory_order_release std::memory_order_release
#        define ov_memory_order_relaxed std::memory_order_relaxed
#        define ov_memory_order_acquire std::memory_order_acquire
#    else
#        define ov_memory_order_release std::memory_order::memory_order_release
#        define ov_memory_order_relaxed std::memory_order::memory_order_relaxed
#        define ov_memory_order_acquire std::memory_order::memory_order_acquire
#    endif

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
                m_prepareCounter.store(i, ov_memory_order_release);
            }
        }
        catch(...) {
            m_completion.store(true, ov_memory_order_relaxed);
            throw;
        }
        m_prepareCounter.store(stop_indx, ov_memory_order_relaxed);
        m_completion.store(true, ov_memory_order_release);
    }

    void updateDynParams(size_t node_indx, size_t /*unused*/) {
        size_t local_counter = node_indx;
        while (true) {
            const bool completion = m_completion.load(ov_memory_order_acquire);
            const size_t prepareCounter = m_prepareCounter.load(ov_memory_order_relaxed);
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
    DEBUG_LOG("Starting inference of the graph: ", GetName(), ". Status: ", static_cast<int>(status));
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

void Graph::SortTopologically() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::SortTopologically");

    auto sort = [](const std::vector<NodePtr>& nodes) {
        std::unordered_set<NodePtr> visited;
        visited.reserve(nodes.size());
        std::vector<NodePtr> sorted;
        sorted.reserve(nodes.size());

        std::function<void(const NodePtr)> visit;
        visit = [&visited, &sorted, &visit](const NodePtr node) {
            const bool inserted = visited.insert(node).second;
            if (!inserted)
                return; // already visited

            for (size_t i = 0; i < node->getChildEdges().size(); i++) {
                visit(node->getChildEdgeAt(i)->getChild());
            }

            sorted.push_back(node);
        };

        for (const auto& node : nodes) {
            visit(node);
        }

        return sorted;
    };

    // as a first step sort in reversed topological order to avoid an insertion into the front of the vector
    graphNodes = sort(graphNodes);
    // reverse to the actual topological order
    std::reverse(graphNodes.begin(), graphNodes.end());
    // number the nodes based on topological order
    for (size_t i = 0; i < graphNodes.size(); i++)
        graphNodes[i]->execIndex = static_cast<int>(i);

    // Sort in / out child edges by port index
    // Make first N (N == port_num) edge indexes match with port index
    for (auto &node : graphNodes) {
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

void Graph::CreateEdge(const NodePtr& parent,
                       const NodePtr& child,
                       int parentPort,
                       int childPort) {
    assert(parentPort >= 0 && childPort >= 0);

    auto edge = std::make_shared<Edge>(parent, child, parentPort, childPort);

    parent->addChildEdge(edge);
    child->addParentEdge(edge);
    graphEdges.push_back(edge);
}

void Graph::RemoveEdge(const EdgePtr& edge) {
    edge->getParent()->removeChildEdge(edge);
    edge->getChild()->removeParentEdge(edge);

    graphEdges.erase(std::remove(graphEdges.begin(), graphEdges.end(), edge), graphEdges.end());
}

void Graph::AddNode(NodePtr node) {
    assert(node);
    assert(std::find(graphNodes.begin(), graphNodes.end(), node) == graphNodes.end());

    graphNodes.push_back(node);
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
        RemoveEdge(p_edge);

        for (size_t j = 0; j < children.size(); j++) {
            auto c_edge = children[j].lock();
            if (!c_edge) continue;
            auto child = c_edge->getChild();
            if (!child) continue;

            const int outNum = c_edge->getOutputNum();
            RemoveEdge(c_edge);
            CreateEdge(parent, child, inNum, outNum);
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
        RemoveEdge(p_edge);

        for (size_t j = 0; j < children.size(); j++) {
            auto c_edge = children[j].lock();
            if (!c_edge) continue;
            auto child = c_edge->getChild();
            if (!child) continue;

            const int outNum = c_edge->getOutputNum();
            RemoveEdge(c_edge);
            CreateEdge(parent, child, inNum, outNum);
        }
    }

    for (size_t i = 1; i < parents.size(); i++) {
        auto p_edge = parents[i].lock();
        if (!p_edge) continue;
        auto parent = p_edge->getParent();
        if (!parent) continue;

        const int inNum = p_edge->getInputNum();
        const int portCandidate = p_edge->getOutputNum();
        RemoveEdge(p_edge);
        const int outNum = parentConv->parentEdges.size();

        parentConv->inputShapes.push_back(node->getInputShapeAtPort(portCandidate));
        CreateEdge(parent, parentConv, inNum, outNum);
    }
    parentConv->outputShapes[0] = node->getOutputShapeAtPort(0);
}

void Graph::RemoveDroppedNodes() {
    graphNodes.erase(std::remove_if(graphNodes.begin(), graphNodes.end(),
                                    [](const NodePtr& node){ return node->isDropped(); }),
                     graphNodes.end());
}

void Graph::RemoveDroppedEdges() {
    graphEdges.erase(std::remove_if(graphEdges.begin(), graphEdges.end(),
                                    [](const EdgePtr& node){ return node->isDropped(); }),
                     graphEdges.end());
}

NodePtr Graph::InsertReorder(EdgePtr edge,
                             std::string layerName,
                             const MemoryDesc& inDesc,
                             const MemoryDesc& outDesc,
                             bool isOptimized,
                             const std::vector<int> & src_perm) {
    auto reorder = std::make_shared<node::Reorder>(inDesc, outDesc, layerName, context);
    reorder->setOptimized(isOptimized);
    reorder->setSrcPermutation(src_perm);

    DEBUG_LOG(reorder->getName(), " edge=", edge->name(), " isOptimized=", isOptimized);
    DEBUG_LOG("    inDesc: ", inDesc.getShape().toString(), inDesc.getPrecision().get_type_name(), " ", inDesc.serializeFormat());
    DEBUG_LOG("   outDesc: ", outDesc.getShape().toString(), outDesc.getPrecision().get_type_name(), " ", outDesc.serializeFormat());

    InsertNode(edge, reorder, true);

    // Using the method Edge::getDesc() we can check that input and output tensor descriptors are equal.
    // Due to the specificity of GraphOptimizer::MergeTransposeAndReorder() that isOptimized flag uses, we shouldn't do these checks.
    if (!isOptimized) {
        reorder->getParentEdgeAt(0)->getDesc();
        reorder->getChildEdgeAt(0)->getDesc();
    }

    return reorder;
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
    edge->getParent()->removeChildEdge(edge);
    edge->getChild()->removeParentEdge(edge);

    return InsertNode(edge->getParent(), edge->getChild(), node, iIndex, oIndex, initNode);
}

bool Graph::InsertNode(NodePtr parent, NodePtr child, NodePtr node, int parentPort, int childPort, bool initNode) {
    CreateEdge(parent, node, parentPort, 0);
    CreateEdge(node, child, 0, childPort);
    AddNode(node);

    if (initNode) {
        node->getSupportedDescriptors();
        node->initSupportedPrimitiveDescriptors();
        node->filterSupportedPrimitiveDescriptors();
        node->selectOptimalPrimitiveDescriptor();
        node->resolveInPlaceDirection();
        node->initOptimalPrimitiveDescriptor();
    }
    return true;
}

// Apply inference precision configuration
void Graph::EnforceInferencePrecision() {
    CPU_DEBUG_CAP_ENABLE(EnforceInferPrcDebug inferPrecDebug);

    const auto inferPrec = getConfig().inferencePrecision;

    if (inferPrec == ov::element::f32)
        return; // nothing to do, only precision reduction is currently allowed
#if defined(OV_CPU_ARM_ENABLE_FP16)
    if (inferPrec == ov::element::f16)
        return; // precision of configured by ov::pass::ConvertPrecision
#endif
    std::function<void(const NodePtr&, std::unordered_set<NodePtr>& skipNodes)> searchForNodesToSkip;
    searchForNodesToSkip = [&](const NodePtr& node, std::unordered_set<NodePtr>& skipNodes) -> void {
        for (size_t i = 0; i < node->getParentEdges().size(); i++) {
            const auto& parent = node->getParentEdgeAt(i)->getParent();
            if (inferPrec == ov::element::bf16) {
                /* list of node types that must be forced to be executed in BF16 precision
                * because of performance gains */
                if (one_of(parent->getType(),
                        Type::Convolution,    // conv nets
                        Type::FullyConnected, // conv / bert nets
                        Type::RNNCell,        // recurent nets
                        Type::RNNSeq,         // recurent nets
                        Type::MatMul,         // bert nets
                        Type::ROIPooling,     // object detection nets
                        Type::Interpolate))   // super resolution nets
                    continue;   // stop at significant nodes
            } else if (inferPrec == ov::element::f16) {
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

        if (one_of(node->getType(), Type::Input, Type::Output, Type::MemoryInput, Type::MemoryOutput))
            continue;
        if (node->keepOrigPrecision())
            continue;
#ifdef CPU_DEBUG_CAPS
        if (!inferPrecDebug.enabled(NameFromType(node->getType()), node->getName(), node->getOriginalLayers()))
            continue;
#endif

        for (size_t i = 0; i < node->getOriginalInputsNumber(); i++) {
            auto keepOriginalInputPrecisionAtPort = [](const NodePtr& node, const size_t inPort) {
                // keep non-float precisions
                if (node->getOriginalInputPrecisionAtPort(inPort) != ov::element::f32)
                    return true;

                const auto &parent = node->getParentEdgeAt(inPort)->getParent();
                /* Skip BF16 enforcement for nodes after Constant Inputs for maintaining precision for fusing.
                 * Element type conversion to bf16 is done automatically, if convolution follows up after Constant Inputs
                 * and activation is bf16 */
                if (parent->getType() == Type::Input && parent->isConstant() &&
                    // Concatenation node is exception because it doesn't change an accuracy for BF16 activation
                    node->getType() != Type::Concatenation)
                    return true;
                // Eltwise and Subgraph (snippets) nodes support precision conversion
                if (parent->getType() == Type::Input && one_of(node->getType(), Type::Eltwise, Type::Subgraph))
                    return true;

                // exclude Convert after Range since it may cause precision loss when integter type to LP.
                if (parent->getType() == Type::Range && node->getType() == Type::Convert) {
                    return true;
                }

                return false;
            };

            if (keepOriginalInputPrecisionAtPort(node, i))
                continue;

            DEBUG_LOG("#",
                      node->getExecIndex(),
                      " ",
                      node->getTypeStr(),
                      " : ",
                      node->getName(),
                      " input[",
                      i,
                      "] is enforced to use",
                      inferPrec);
            node->setOriginalInputPrecisionAtPort(i, inferPrec);
        }

        for (size_t i = 0; i < node->getOriginalOutputsNumber(); i++) {
            // keep non-float precisions
            if (node->getOriginalOutputPrecisionAtPort(i) != ov::element::f32)
                continue;

            // exclude Convert before Range since it may cause precision loss when integter type to LP.
            // TODO: Incorrect subgraph is generated by ONNX FE + ticket 117861.
            const auto &child = node->getChildEdgeAt(i)->getChild();
            if (child->getType() == Type::Range && node->getType() == Type::Convert)
                continue;

            DEBUG_LOG("#",
                      node->getExecIndex(),
                      " ",
                      node->getTypeStr(),
                      " : ",
                      node->getName(),
                      " output[",
                      i,
                      "] is enforced to use",
                      inferPrec);
            node->setOriginalOutputPrecisionAtPort(i, inferPrec);
        }
    }
}

std::shared_ptr<ov::Model> Graph::dump() const {
    return dump_graph_as_ie_ngraph_net(*this);
}

void Graph::SearchInternalStateNodes() {
    for (auto&& node : graphNodes) {
        if (node->getType() == Type::MemoryInput) {
            auto cur_node = std::dynamic_pointer_cast<node::MemoryStateNode>(node);
            if (!cur_node) {
                OPENVINO_THROW("Cannot cast ", node->getName(), " to MemoryStateNode");
            }
            internalStateNodes.insert({cur_node->getId(), cur_node});
        }
    }
}

}   // namespace intel_cpu
}   // namespace ov
