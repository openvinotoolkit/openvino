// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <tuple>
#include <unordered_set>
#include <limits>
#include <fstream>
#include <unordered_map>
#include <memory>
#include <utility>

#include "graph.h"
#include "graph_dumper.h"
#include "graph_optimizer.h"
#include "dnnl_extension_utils.h"
#include "extension_mngr.h"
#include "memory_solver.hpp"
#include "itt.h"
#include "infer_request.h"
#include "nodes/input.h"
#include <nodes/reorder.h>
#include "nodes/convert.h"
#include "nodes/subgraph.h"
#include "nodes/fullyconnected.h"

#include <ie_algorithm.hpp>
#include <blob_factory.hpp>
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"

#include "precision_utils.h"
#include <ie_plugin_config.hpp>

#include "utils/general_utils.h"
#include "utils/debug_capabilities.h"
#include "utils/node_dumper.h"
#include "utils/ngraph_utils.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/verbose.h"
#include "memory_desc/cpu_memory_desc_utils.h"

#include <ngraph/node.hpp>
#include <ngraph/function.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/ops.hpp>
#include <transformations/utils/utils.hpp>
#include <low_precision/low_precision.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <common/primitive_desc.hpp>
#include <common/primitive_desc_iface.hpp>
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
#   include <tbb/task_group.h>
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

void Graph::CreateGraph(const std::vector<NodePtr> &graphNodes,
                              const std::vector<EdgePtr> &graphEdges,
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

template void Graph::CreateGraph(const std::shared_ptr<const ngraph::Function>&, const GraphContext::CPtr);
template void Graph::CreateGraph(const CNNNetwork&, const GraphContext::CPtr);

void Graph::Replicate(const std::shared_ptr<const ov::Model> &subgraph) {
    this->_name = "subgraph";
    this->reuse_io_tensors = false;

    // Map data object onto producer node
    std::map<std::shared_ptr<ov::Node>, NodePtr> op2node;

    // nodes which has no consumers (output or just unused). But doesn't marked as graph output.
    // Will be stored as fake output separately.
    std::deque<ngraph::Output<ngraph::Node>> unusedOutputs;

    auto getParentOutputPort = [](const std::shared_ptr<ngraph::Node> childOp, const std::shared_ptr<ngraph::Node> parentOp,
                                  const size_t childInputPort) -> int {
        for (size_t parentPort = 0; parentPort < parentOp->get_output_size(); parentPort++) {
            if (childOp->input(childInputPort).get_tensor_ptr() == parentOp->output(parentPort).get_tensor_ptr()) {
                return static_cast<int>(parentPort);
            }
        }

        return -1;
    };

    for (const auto& op : subgraph->get_ordered_ops()) {
        const NodePtr node {Node::factory().create(op, context)};

        graphNodes.push_back(node);

        if (op->get_type_info() == ngraph::op::v0::Parameter::get_type_info_static()) {
            inputNodesMap[node->getName()] = node;
        }

        if (op->get_type_info() == ngraph::op::v0::Result::get_type_info_static()) {
            const auto prev = op->input_value(0);
            const std::string inputID = ov::op::util::get_ie_output_name(prev);

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
                ngraph::op::v0::Result::get_type_info_static(),
                ngraph::op::v3::Assign::get_type_info_static(),
                ngraph::op::v6::Assign::get_type_info_static())) {
            for (int oi = 0; oi < op->get_output_size(); oi++) {
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

    if (getConfig().enforceBF16)
        EnforceBF16();
}

void Graph::Replicate(const CNNNetwork &network) {
    OV_ITT_SCOPE_CHAIN(FIRST_INFERENCE, taskChain, itt::domains::intel_cpu_LT, "Graph::Replicate", "CNNNetwork");

    InputsDataMap inputsInfo = network.getInputsInfo();
    OutputsDataMap outputsInfo = network.getOutputsInfo();

    this->_name = network.getName();

    std::shared_ptr<const ov::Model> func = nullptr;
    // we perform model cloning and reshaping on Replicate stage to preserve input/output information
    // it help to perform a graph compilation like in static case
    // and handle dynamic batch case in inference stage with minimal code changes
    if (getConfig().isNewApi && getConfig().batchLimit > 0) {
        auto upperBoundModel = ngraph::clone_function(*network.getFunction());
        std::map<ov::Output<ov::Node>, ov::PartialShape> newInShape;
        for (const auto& in : upperBoundModel->get_parameters()) {
            auto newShape = in->get_output_partial_shape(0);
            newShape[0] = getConfig().batchLimit;
            newInShape[in] = newShape;
        }
        upperBoundModel->reshape(newInShape);

        func = upperBoundModel;
    } else {
        func = network.getFunction();
    }

    if (!func) {
        IE_THROW() << "Function pointer inside CNNNetwork is nullptr";
    }

    auto orderedOps = func->get_ordered_ops();

    // TODO [NM]: unordered_map is preferred from performance perspective. Needs hash for ngraph::Node
    std::map<std::shared_ptr<ngraph::Node>, NodePtr> op2node;
    std::deque<ngraph::Output<ngraph::Node>> unusedOutputs;  // nodes which has no consumers (output or just unused)

    auto getParentOutputPort = [](const std::shared_ptr<ngraph::Node> childOp, const std::shared_ptr<ngraph::Node> parentOp,
                                  const size_t childInputPort) -> int {
        for (size_t parentPort = 0; parentPort < parentOp->get_output_size(); parentPort++) {
            if (childOp->input(childInputPort).get_tensor_ptr() == parentOp->output(parentPort).get_tensor_ptr()) {
                return static_cast<int>(parentPort);
            }
        }

        return -1;
    };

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "AllNodes");

    // Replicate All Nodes in topological order
    for (const auto& op : orderedOps) {
        const NodePtr node(Node::factory().create(op, context));

        graphNodes.push_back(node);

        if (op->get_type_info() == ngraph::op::v0::Parameter::get_type_info_static()) {
            const auto inInfo = inputsInfo.find(node->getName());
            if (inInfo != inputsInfo.end()) {
                inputNodesMap[node->getName()] = node;
                if (node->isDynamicNode()) {
                    graphHasDynamicInput = true;
                }
            }
        }

        if (op->get_type_info() == ngraph::op::v0::Result::get_type_info_static()) {
            const auto &input = op->input_value(0);
            const auto name = ov::op::util::get_ie_output_name(input);

            if (outputsInfo.count(name) != 0) {
                outputNodesMap[name] = node;
            }
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
                ngraph::op::v0::Result::get_type_info_static(),
                ngraph::op::v3::Assign::get_type_info_static(),
                ngraph::op::v6::Assign::get_type_info_static())) {
            for (int oi = 0; oi < op->get_output_size(); oi++) {
                if (op->get_output_target_inputs(oi).empty()) {
                    unusedOutputs.push_back(op->output(oi));
                }
            }
        }
    }

    // Add stub output node for unused outputs
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

    if (getConfig().enforceBF16)
        EnforceBF16();

    auto hasSubgraphConsumers = [] (const NodePtr& node) -> bool {
        const auto & childEdges = node->getChildEdges();
        return std::any_of(childEdges.begin(), childEdges.end(),
                           [] (const EdgeWeakPtr& edge) -> bool {
                               auto edgePtr = edge.lock();
                               if (!edgePtr)
                                   return false;
                               return edgePtr->getChild()->getType() == Type::Subgraph;
                           });
    };

    // change precision for input/output nodes to avoid extra data conversion when set input/output blobs
    // also we need to change input/output precisions for consumers/producers to avoid inserting reorder
    for (auto &input : inputNodesMap) {
        const auto precToSet = normalizeToSupportedPrecision(inputsInfo.at(input.first)->getPrecision());
        input.second->setOriginalOutputPrecisionAtPort(0, precToSet);
        const auto childEdges = input.second->getChildEdgesAtPort(0);
        for (size_t i = 0; i < childEdges.size(); i++) {
            const auto child = childEdges[i]->getChild();
            if (child->getOriginalInputPrecisionAtPort(childEdges[i]->getOutputNum()) != Precision::BF16 &&
                // remove this WA when #78939 is resolved
                !hasSubgraphConsumers(child))
                child->setOriginalInputPrecisionAtPort(childEdges[i]->getOutputNum(), precToSet);
        }
    }

    for (auto &output : outputNodesMap) {
        const auto precToSet = normalizeToSupportedPrecision(outputsInfo.at(output.first)->getPrecision());
        output.second->setOriginalInputPrecisionAtPort(0, precToSet);
        const auto parentEdges = output.second->getParentEdgesAtPort(0);
        for (size_t i = 0; i < parentEdges.size(); i++) {
            const auto parent = parentEdges[i]->getParent();
            parent->setOriginalOutputPrecisionAtPort(parentEdges[i]->getInputNum(), precToSet);
        }
    }

    // Loading mean images
    for (const auto& input : inputsInfo) {
        Shape outShape;
        if (!inputNodesMap[input.first]->outputShapes.front().getRank()) {
            outShape =  Shape(SizeVector({1, 1}));
        } else {
            outShape = inputNodesMap[input.first]->outputShapes.front();
        }
        InputInfo::Ptr ii = inputsInfo[input.first];
        if (ii && ii->getPreProcess().getNumberOfChannels()) {
            _normalizePreprocMap[input.first].Load(outShape, ii);
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

    InitOptimalPrimitiveDescriptors();

    InitEdges();

    optimizer.ApplyImplSpecificGraphOptimizations(*this);
    SortTopologically();

    bool haveDynNodes = false;
    for (size_t i = 0; i < graphNodes.size(); ++i) {
        const auto& node = graphNodes[i];
        if (node->isDynamicNode()) {
            haveDynNodes = true;
            if (node->outputShapeDataDependency() ||
                // WA: for convolution plus summ(broadcast). Due to the fact that a convolution with sum use the same memory for second sum term and the output
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
    // If the input tensor is included to memory reuse, that means its memory manager is shared with other tensors in the graph, which in turn may cause data
    // loss when one of the tensor dow the graph requested mem resize, while the input data have not been yet read by the consumers. To avoid such situations
    // we disalbe io mem reuse for the case of dynamic shapes.
    if (haveDynNodes) {
        this->reuse_io_tensors = false;
    }

    Allocate();

    CreatePrimitives();

#ifndef CPU_DEBUG_CAPS
    for (auto &graphNode : graphNodes) {
        graphNode->cleanup();
    }
#endif
    ExtractConstantAndExecutableNodes();

    ExecuteConstantNodesOnly();
    status = haveDynNodes ? Status::ReadyDynamic : Status::ReadyStatic;
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
        if (node->getType() == Type::Input && _normalizePreprocMap.find(node->getName()) != _normalizePreprocMap.end()) {
            auto *inputNode = dynamic_cast<node::Input *>(node.get());
            if (inputNode)
                inputNode->withMeanImage();
        }

        OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, node->profiling.getSupportedDescriptors);
        node->getSupportedDescriptors();

        OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, node->profiling.initSupportedPrimitiveDescriptors);
        node->initSupportedPrimitiveDescriptors();

        OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, node->profiling.filterSupportedPrimitiveDescriptors);
        node->filterSupportedPrimitiveDescriptors();

#ifdef CPU_DEBUG_CAPS
        DEBUG_LOG("==================");
        for (auto & pd : node->getSupportedPrimitiveDescriptors())
            DEBUG_LOG("#", node->getExecIndex(),
                      " ", node->getName(),
                      "  SupportedPrimitiveDescriptor:\n", pd);
#endif
    }

    for (auto &node : graphNodes) {
        OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, node->profiling.selectOptimalPrimitiveDescriptor);
        node->selectOptimalPrimitiveDescriptor();
    }
}

void Graph::InitOptimalPrimitiveDescriptors() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "Graph::InitOptimalPrimitiveDescriptors");
    for (auto &node : graphNodes) {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, node->profiling.initOptimalPrimitiveDescriptor);
        node->initOptimalPrimitiveDescriptor();
        DEBUG_LOG("#", node->getExecIndex(), " ", node->getName(), "\n", *node->getSelectedPrimitiveDescriptor());
    }
}

void Graph::ExtractConstantAndExecutableNodes() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::ExtractConstantAndExecutableNodes");
    for (const auto& graphNode : graphNodes) {
        if (graphNode->isConstant()) {
            constantGraphNodes.emplace_back(graphNode);
        } else if (CPU_DEBUG_CAPS_ALWAYS_TRUE(graphNode->isExecutable()) || graphNode->isDynamicNode()) {
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

void Graph::ExecuteConstantNodesOnly() const {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::ExecuteConstantNodesOnly");
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

    for (const auto &node : constantGraphNodes) {
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
    auto status = dnnl_reorder_primitive_desc_create(&result, &srcMemDesc.data, eng.get(), &dstMemDesc.data, eng.get(),
                                                     attr.get());
    if (result) {
        dnnl_primitive_desc_destroy(result);
    }

    return dnnl_success == status;
}

void Graph::InitEdges() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::InitEdges");

    size_t numberOfEdges = graphEdges.size();

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

    auto updateEdge = [&](int& i) {
        graphEdges.erase(graphEdges.begin() + i);
        i--;
        numberOfEdges--;
    };

    for (auto i = 0; i < numberOfEdges; i++) {
        auto edge = graphEdges[i];
        auto reorderStatus = graphEdges[i]->needReorder();
        DEBUG_LOG(graphEdges[i]->name(), " reorderStatus = ", static_cast<int>(reorderStatus));
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

    size_t edge_clusters_count = edge_clusters.size();

    for (size_t i = 0; i < edge_clusters_count;) {
        auto &cluster = edge_clusters[i];
        bool erase = false;
        for (auto &edge : cluster) {
            if (edge->getStatus() == Edge::Status::NeedAllocation
                && edge->getParent()->isConstant()) {
                if (edge->getParent()->getType() == Type::Input) {
                    auto constNode = std::static_pointer_cast<node::Input>(edge->getParent());
                    edge->reuse(std::const_pointer_cast<Memory>(constNode->getMemoryPtr()));
                } else {
                    edge->externalAllocate(context->getWeightsCache());
                }
                erase = true;
            }
        }

        if (erase) {
            std::swap(edge_clusters[i], edge_clusters[edge_clusters_count - 1]);
            --edge_clusters_count;
        } else {
            ++i;
        }
    }

    edge_clusters.resize(edge_clusters_count);

    const int64_t alignment = 32;  // 32 bytes

    std::vector<MemorySolver::Box> definedBoxes;
    std::vector<MemorySolver::Box> undefinedBoxes;
    for (int i = 0; i < edge_clusters.size(); i++) {
        MemorySolver::Box box = { std::numeric_limits<int>::max(), 0, 0, i };
        int64_t boxSize = 0;
        for (auto &edge : edge_clusters[i]) {
            int e_start = edge->getParent()->execIndex;
            int e_finish = edge->getChild()->execIndex;

            if (boxSize != -1 && edge->getDesc().hasDefinedMaxSize()) {
                int64_t e_size = edge->getDesc().getMaxMemSize();  // size in bytes (from the beginning of data to the last element)
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

    MemorySolver staticMemSolver(definedBoxes);
    size_t total_size = static_cast<size_t>(staticMemSolver.solve()) * alignment;

    memWorkspace = std::make_shared<Memory>(getEngine());
    memWorkspace->Create(DnnlBlockedMemoryDesc(InferenceEngine::Precision::I8, Shape(InferenceEngine::SizeVector{total_size})));

    if (edge_clusters.empty())
        return;

    auto* workspace_ptr = static_cast<int8_t*>(memWorkspace->GetData());

    for (auto& box : definedBoxes) {
        int count = 0;
        for (auto& edge : edge_clusters[box.id]) {
            if (edge->getStatus() == Edge::Status::NeedAllocation) {
                int64_t offset = staticMemSolver.getOffset(box.id);
                // !! Fallback to individual memory allocation !!
                // if you like to check infer without reuse just call this function without arguments.
                edge->allocate(workspace_ptr + offset * alignment);  // alignment in byte

                // TODO: WA for some test (like strided_slice_test) which use tensors with
                //       shapes {0}. And it is implisitly converted into {1} tensor.
                //       Zeroing of input data allow pass tests.
                if (edge->getParent()->type == Type::Input && edge->hasDefinedMaxSize())
                    edge->getMemoryPtr()->FillZero();

                count++;
            }
        }
        IE_ASSERT(count == 1);
    }

    if (!undefinedBoxes.empty()) {
        if (!syncNodesInds.empty()) {
            //We have to extend the lifespan of thensors that are crossing a sync point border in order to save
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
                std::make_shared<DnnlMemoryMngr>(std::unique_ptr<MemoryMngrWithReuse>(new MemoryMngrWithReuse()));
            for (auto& box : group) {
                for (auto& edge : edge_clusters[box.id]) {
                    if (edge->getStatus() == Edge::Status::NeedAllocation) {
                        edge->allocate(grpMemMngr);
                    }
                }
            }
        }
    }
}

void Graph::Allocate() {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "Graph::Allocate");

    // resolve edges. Define which will be a view on others
    //   NeedAllocation - real blob
    //   NotAllocated - view on other blob, peer or in-place
    for (auto& edge : graphEdges) edge->init();

    // Allocate memory space for all edges marked with NeedAllocation
    AllocateWithReuse();

    // Resolve all other edges with status NotAllocated and in-place
    for (auto& node : graphNodes) node->resolveInPlaceEdges();

    // Check all getters. Should work.
    for (auto& edge : graphEdges) edge->validate();
}

void Graph::CreatePrimitives() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "Graph::CreatePrimitives");
    for (auto& node : graphNodes) {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, node->profiling.createPrimitive);
        DEBUG_LOG(*node);
        node->createPrimitive();
#ifdef CPU_DEBUG_CAPS
        if (node->prim) {
            auto pd_c = node->prim.get_primitive_desc();
            auto* pd = reinterpret_cast<const dnnl_primitive_desc*>(pd_c);
            DEBUG_LOG("verbose##", node->getName(), "##", pd->info(), "\n");
        }
#endif
    }
}

void Graph::PushInputData(const std::string& name, const InferenceEngine::Blob::Ptr &in) {
    if (!IsReady()) IE_THROW()<< "Wrong state. Topology not ready.";

    auto input = inputNodesMap.find(name);
    if (input != inputNodesMap.end()) {
        auto& inTensorDesc = in->getTensorDesc();
        auto node = input->second;
        auto childEdge = node->getChildEdgeAt(0);
        const auto& outDims = node->getOutputShapeAtPort(0);

        const void *ext_data_ptr = in->cbuffer();
        void *inter_data_ptr = childEdge->getMemory().GetData();

        if (ext_data_ptr != inter_data_ptr) {
            auto ext_tdesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(in->getTensorDesc());

            Memory ext_mem(getEngine());
            ext_mem.Create(ext_tdesc, ext_data_ptr, false);

            // branch for handling dynamic batch feature in new API
            if (getConfig().isNewApi && getConfig().batchLimit > 0 && ext_mem.getStaticDims()[0] != childEdge->getMemory().getStaticDims()[0]) {
                auto newDims = childEdge->getMemory().getStaticDims();
                newDims[0] = ext_mem.getStaticDims()[0];

                Memory tmpMem(getEngine());
                auto newDesc = childEdge->getMemory().getDesc().cloneWithNewDims(newDims, true);
                tmpMem.Create(newDesc, childEdge->getMemory().GetData(), false);

                tmpMem.SetData(ext_mem, false);
            } else {
                childEdge->getMemory().SetData(ext_mem, false);
            }
        }

        // todo: make sure 'name' exists in this map...
        if (_normalizePreprocMap.find(name) != _normalizePreprocMap.end()) {
            if (inTensorDesc.getPrecision() == InferenceEngine::Precision::FP32) {
                _normalizePreprocMap[name].NormalizeImage(outDims, reinterpret_cast<float *>(inter_data_ptr),
                                                          inTensorDesc.getLayout());
            } else {
                IE_THROW() << "Mean image of type " << inTensorDesc.getPrecision().name() << " is unsupported";
            }
        }
    } else {
        IE_THROW() << "Input blob for infer '" << name << "' doesn't correspond to input in network";
    }
}

void Graph::PullOutputData(BlobMap &out) {
    if (!IsReady())
        IE_THROW() << "Wrong state. Topology not ready.";

    for (auto &outputMap : outputNodesMap) {
        auto name = outputMap.first;
        auto node = outputMap.second;
        auto parentEdge = node->getParentEdgeAt(0);
        const Memory& intr_blob = parentEdge->getMemory();

        const auto ext_blob_map = out.find(name);
        const auto ext_blob = ext_blob_map->second;
        if (ext_blob_map == out.end()) {
            IE_THROW(Unexpected) << "The CPU plugin graph doesn't contain output node with name: \"" << name << "\"";
        }

        const auto actualDesc = MemoryDescUtils::convertToTensorDesc(intr_blob.getDesc());
        auto &expectedDesc = ext_blob->getTensorDesc();

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
        if (out[name]->getTensorDesc().getDims() != outDims && !isScalarOutput) {
            // WA: because input/output info initially contains non empty dims, order etc.
            // and setDims (called inside setShape) can't correct modify blocked desc for desc with blocked layout
            if (expectedDesc.getLayout() == InferenceEngine::Layout::BLOCKED) {
                expectedDesc = TensorDesc(expectedDesc.getPrecision(), expectedDesc.getLayout());
            }
            if (getConfig().isNewApi && getConfig().batchLimit > 0) {
                outDims[0] = node->batchToProcess();
            }
            out[name]->setShape(outDims);
        }

        // check for empty output blob
        if (std::any_of(outDims.begin(), outDims.end(), [](const Dim dim) {return dim == 0;})) {
            continue;
        }

        auto srcPrec = actualDesc.getPrecision();
        auto dstPrec = expectedDesc.getPrecision();

        if ((getConfig().isNewApi && !getConfig().batchLimit) && srcPrec == dstPrec && ext_blob->byteSize() != intr_blob.GetSize())
                IE_THROW() << "Output blob byte size is not equal network output byte size ("
                                   << ext_blob->byteSize() << "!=" << intr_blob.GetSize() << ").";

        void *ext_blob_ptr = ext_blob->buffer();
        void *intr_blob_ptr = intr_blob.GetData();

        // That is the same memory. No need to copy
        if (ext_blob_ptr == intr_blob_ptr) continue;

        if (actualDesc.getBlockingDesc() != expectedDesc.getBlockingDesc() && !isScalarOutput) {
            // User can initialize output via SetOutput API using tensorDesc with ANY layout.
            // For these cases we create planar memory descriptor.
            auto outBlobDesc = expectedDesc.getLayout() == InferenceEngine::Layout::ANY
                                ? DnnlBlockedMemoryDesc(expectedDesc.getPrecision(), Shape(expectedDesc.getDims()))
                                : MemoryDescUtils::convertToDnnlBlockedMemoryDesc(expectedDesc);
            Memory outBloMem(getEngine());
            outBloMem.Create(outBlobDesc, ext_blob_ptr, false);

            // branch for handling dynamic batch feature in new API
            if (getConfig().isNewApi && getConfig().batchLimit > 0 && outBloMem.getStaticDims()[0] != intr_blob.getStaticDims()[0]) {
                auto newDims = intr_blob.getStaticDims();
                newDims[0] = outBloMem.getStaticDims()[0];

                Memory tmpMem(getEngine());
                auto newDesc = intr_blob.getDesc().cloneWithNewDims(newDims, true);
                tmpMem.Create(newDesc, intr_blob.GetData(), false);

                outBloMem.SetData(tmpMem, false);
            } else {
                outBloMem.SetData(intr_blob, false);
            }
        } else {
            size_t size_to_copy = intr_blob.GetDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
            // TODO: Should we support InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT???
            // TODO [DS]: phase 2: should we support this behaviour? Looks obsolete in the dynamic shapes paradigm
            if (getConfig().batchLimit) {
                if (node->isDynamicNode() && !getConfig().isNewApi) {
                    IE_THROW(NotImplemented) << "[DS] not implemented dynamic batch for node with dynamic shape";
                }
                int MB_to_process = node->batchToProcess();
                size_to_copy = std::accumulate(outDims.begin() + 1, outDims.end(), (size_t)1, std::multiplies<size_t>()) * MB_to_process;
            }

            cpu_convert(intr_blob_ptr, ext_blob_ptr, srcPrec, dstPrec, size_to_copy);
        }
    }
}

void Graph::InferStatic(InferRequestBase* request) {
    dnnl::stream stream(getEngine());

    for (const auto& node : executableGraphNodes) {
        VERBOSE(node, getConfig().debugCaps.verbose);
        PERF(node, getConfig().collectPerfCounters);

        if (request)
            request->ThrowIfCanceled();
        ExecuteNode(node, stream);
    }
}

void Graph::InferDynamic(InferRequestBase* request) {
    dnnl::stream stream(getEngine());

    std::set<size_t> syncIndsWorkSet;
    for (const auto& nodeIndx : syncNodesInds) {
        syncIndsWorkSet.insert(nodeIndx.second);
        //since sometimes we need to run the synchronization node  alone (for example in the case of internal dynamism)
        //let's add another sync index after the sync point node
        syncIndsWorkSet.insert(nodeIndx.second + 1);
    }
    syncIndsWorkSet.insert(executableGraphNodes.size());

    std::function<void(size_t)> updateNodes;

#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    std::atomic<size_t> prepareCounter(0);
    std::vector<std::atomic<uint8_t>> waveFrontCount(executableGraphNodes.size());
    waveFrontCount.front().store(1);
    for (size_t i = 1; i < waveFrontCount.size(); ++i) {
        waveFrontCount[i].store(2);
    }

    tbb::task_group tg;
    std::function<void(size_t, size_t)> updateShapes;
    std::function<void(size_t, size_t)> updateDynParams;

    updateShapes = [&](size_t node_indx, size_t stop_indx) {
        const auto& node = executableGraphNodes[node_indx];
        prepareCounter.store(node_indx);
        if (node_indx >= stop_indx) {
            return;
        }
        if (node->isDynamicNode()) {
            node->updateShapes();
        }
        if (--waveFrontCount[node_indx] == 0) {
            tg.run([=, &updateDynParams](){ updateDynParams(node_indx, stop_indx); });
        }
        updateShapes(node_indx + 1, stop_indx);
    };

    updateDynParams = [&](size_t node_indx, size_t stop_indx) {
        const auto& node = executableGraphNodes[node_indx];
        if (node_indx >= stop_indx) {
            prepareCounter.store(node_indx);
            return;
        }
        if (node->isDynamicNode()) {
            node->updateDynamicParams();
        }
        if (node_indx + 1 < waveFrontCount.size() && --waveFrontCount[node_indx + 1] == 0) {
            tg.run([=, &updateDynParams](){ updateDynParams(node_indx + 1, stop_indx); });
        }
    };

    updateNodes = [&](size_t stopIndx) {
        auto startCounter = prepareCounter.load();
        tg.run([=, &updateShapes](){ updateShapes(startCounter, stopIndx); });
        tg.wait();
    };
#else
    size_t prepareCounter = 0;
    updateNodes = [&](size_t stopIndx) {
        for (; prepareCounter < stopIndx; ++prepareCounter) {
            const auto& node = executableGraphNodes[prepareCounter];
            if (node->isDynamicNode()) {
                node->updateShapes();
                node->updateDynamicParams();
            }
        }
    };
#endif
    size_t inferCounter = 0;

    for (auto stopIndx : syncIndsWorkSet) {
        updateNodes(stopIndx);
        for (; inferCounter < stopIndx; ++inferCounter) {
            auto& node = executableGraphNodes[inferCounter];
            VERBOSE(node, getConfig().debugCaps.verbose);
            PERF(node, getConfig().collectPerfCounters);

            if (request)
                request->ThrowIfCanceled();
            ExecuteNode(node, stream);
        }
    }
}

inline void Graph::ExecuteNode(const NodePtr& node, const dnnl::stream& stream) const {
    DUMP(node, getConfig().debugCaps, infer_count);

    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, node->profiling.execute);

    if (node->isDynamicNode()) {
        node->executeDynamic(stream);
    } else {
        node->execute(stream);
    }
    DEBUG_LOG(*node);
}

void Graph::Infer(InferRequestBase* request) {
    if (!IsReady()) {
        IE_THROW() << "Wrong state of the ov::intel_cpu::Graph. Topology is not ready.";
    }

    if (Status::ReadyDynamic == status) {
        InferDynamic(request);
    } else if (Status::ReadyStatic == status) {
        InferStatic(request);
    } else {
        IE_THROW() << "Unknown ov::intel_cpu::Graph state: " << static_cast<size_t>(status);
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

    for (int i = 0; i < graphNodes.size(); i++) {
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

    for (int i = 0; i < sorted.size(); i++) sorted[i]->execIndex = i;

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

            for (int i = 0; i < node->parentEdges.size(); i++) {
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

            for (int i = 0; i < node->childEdges.size(); i++) {
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

void Graph::GetPerfData(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const {
    unsigned i = 0;
    std::function<void(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &, const NodePtr&)>
            getPerfMapFor = [&](std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap, const NodePtr& node) {
        InferenceEngine::InferenceEngineProfileInfo &pc = perfMap[node->getName()];
        pc.execution_index = i++;
        // TODO: Why time counter is signed?
        pc.cpu_uSec = pc.realTime_uSec = (long long) node->PerfCounter().avg();
        pc.status = pc.cpu_uSec > 0 ? InferenceEngine::InferenceEngineProfileInfo::EXECUTED
                                    : InferenceEngine::InferenceEngineProfileInfo::NOT_RUN;
        std::string pdType = node->getPrimitiveDescriptorType();
        size_t typeLen = sizeof(pc.exec_type) / sizeof(pc.exec_type[0]);
        pdType.copy(pc.exec_type, typeLen, 0);
        size_t layerTypeLen = sizeof(pc.layer_type) / sizeof(pc.layer_type[0]);
        node->typeStr.copy(pc.layer_type, layerTypeLen, 0);

        for (auto& fusedNode : node->fusedWith) {
            getPerfMapFor(perfMap, fusedNode);
        }

        for (auto& mergedWith : node->mergedWith) {
            getPerfMapFor(perfMap, mergedWith);
        }
    };

    for (int i = 0; i < graphNodes.size(); i++) {
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

        for (size_t j = 0; j < children.size(); j++) {
            if (!children[j].lock())
                continue;
            auto child = children[j].lock()->getChild();
            if (!child)
                continue;

            EdgePtr &remEdge = p_edge;
            int inNum = 0;
            if (remEdge) {
                inNum = remEdge->getInputNum();
                remEdge->drop();
                RemoveEdge(remEdge);
            }
            remEdge = children[j].lock();
            int outNum = 0;
            if (remEdge) {
                outNum = remEdge->getOutputNum();
                remEdge->drop();
                RemoveEdge(remEdge);
            }
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

        for (size_t j = 0; j < children.size(); j++) {
            if (!children[j].lock())
                continue;
            auto child = children[j].lock()->getChild();
            if (!child)
                continue;

            EdgePtr &remEdge = p_edge;
            int inNum = 0;
            if (remEdge) {
                inNum = remEdge->getInputNum();
                remEdge->drop();
                RemoveEdge(remEdge);
            }
            remEdge = children[j].lock();
            int outNum = 0;
            if (remEdge) {
                outNum = remEdge->getOutputNum();
                remEdge->drop();
                RemoveEdge(remEdge);
            }
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

        EdgePtr &remEdge = p_edge;
        int inNum = 0;
        int portCandidate = 0;
        if (remEdge) {
            inNum = remEdge->getInputNum();
            portCandidate = remEdge->getOutputNum();
            remEdge->drop();
            RemoveEdge(remEdge);
        }
        int outNum = parentConv->parentEdges.size();

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
        IE_THROW() << "Graph::InsertReorder: Cannot cast to Reorder";
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
        IE_THROW() << "Cannot insert node '" << node->getName() << "' between nodes: "
                           << edge->getParent()->getName() << " and "
                           << edge->getChild()->getName() << ".";

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
        node->initOptimalPrimitiveDescriptor();
    }

    graphEdges.push_back(beforeNode);
    graphEdges.push_back(afterNode);
    graphNodes.push_back(node);
    return true;
}

// Set all non const data paths precision to BF16
void Graph::EnforceBF16() {
    // Floating point parts of FP32 + INT8 or FP32 + BIN mixed precision models will be executed in BF16 precision
    // only if enforceBF16 flag was set manually because current performance is not good enough to enable it by default
    if (!implication(context->isGraphQuantized(), getConfig().manualEnforceBF16))
        return;

    std::function<void(const NodePtr&, std::unordered_set<NodePtr>& skipNodes)> searchForNodesToSkip;
    searchForNodesToSkip = [&](const NodePtr& node, std::unordered_set<NodePtr>& skipNodes) -> void {
        for (size_t i = 0; i < node->getParentEdges().size(); i++) {
            const auto& parent = node->getParentEdgeAt(i)->getParent();

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

            const auto res = skipNodes.insert(parent);
            if (res.second) // node not visited yet
                searchForNodesToSkip(parent, skipNodes);
        }
    };

    /* Skip BF16 enforcement for tail of the graph by forming set of nodes to skip.
     * Necessary to maintain accuracy.
     * Experiments show zero peformance impact on average */
    std::unordered_set<NodePtr> nodesToSkip;
    // starting from output nodes
    for (const auto& entry : outputNodesMap) {
        const auto& node = entry.second;
        searchForNodesToSkip(node, nodesToSkip);
    }

    for (const auto& node : graphNodes) {
        if (nodesToSkip.count(node) && !node->enforceBF16evenForGraphTail)
            continue;

        if (node->getType() != Type::Input && node->getType() != Type::Output) {
            DEBUG_LOG("#", node->getExecIndex(),
                      " ", node->getName(),
                      " is enforced to use BF16\n");
            for (size_t i = 0; i < node->getOriginalInputsNumber(); i++) {
                const auto &parent = node->getParentEdgesAtPort(i)[0]->getParent();
                /* Skip BF16 enforcement for nodes after Constant Inputs for maintaining precision for fusing.
                 * Precision conversion to BF16 does automatically, if convolution follows up after Constant Inputs
                 * and if activation is BF16 */
                if (!(parent->getType() == Type::Input && parent->isConstant() &&
                    // Concatenation node is exception because it doesn't change an accuracy for BF16 activation
                      node->getType() != Type::Concatenation) &&
                    // exclude Eltwise after Input since it supports conversion to BF16
                    !(parent->getType() == Type::Input && (node->getType() == Type::Eltwise || node->getType() == Type::Subgraph)) &&
                    node->getOriginalInputPrecisionAtPort(i) == Precision::FP32)
                    node->setOriginalInputPrecisionAtPort(i, Precision::BF16);
            }

            for (size_t i = 0; i < node->getOriginalOutputsNumber(); i++) {
                if (node->getOriginalOutputPrecisionAtPort(i) == Precision::FP32)
                    node->setOriginalOutputPrecisionAtPort(i, Precision::BF16);
            }
        }
    }
}

std::shared_ptr<ngraph::Function> Graph::dump() const {
    return dump_graph_as_ie_ngraph_net(*this);
}

}   // namespace intel_cpu
}   // namespace ov
