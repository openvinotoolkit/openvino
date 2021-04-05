// Copyright (C) 2018-2021 Intel Corporation
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

#include "mkldnn_graph.h"
#include "mkldnn_graph_dumper.h"
#include "mkldnn_graph_optimizer.h"
#include "mkldnn_extension_utils.h"
#include "mkldnn_extension_mngr.h"
#include "mkldnn_memory_solver.hpp"
#include "mkldnn_itt.h"
#include "mkldnn_infer_request.h"
#include <nodes/mkldnn_input_node.h>
#include <nodes/mkldnn_reorder_node.h>
#include <nodes/mkldnn_convert_node.h>

#include <ie_algorithm.hpp>
#include <blob_factory.hpp>
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"

#include "precision_utils.h"
#include <ie_plugin_config.hpp>

#include "utils/blob_dump.h"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"

#include <ngraph/node.hpp>
#include <ngraph/function.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/ops.hpp>
#include <transformations/utils/utils.hpp>

/*****************************************************
 * Debug capability
 *  - BLOB_DUMP_PATH : Specify with existing folder name
 *    to dump intermediate blobs into it
 *  - PRINT_GRAPH_INFO : Define it to enable printing
 *    additional information to std output.
 *
 *****************************************************/
// #define BLOB_DUMP_PATH "mkldnn_dump"
// #define PRINT_GRAPH_INFO
// #define DUMP_AS_TEXT
// #define DUMP_INTERNAL_BLOBS

#ifdef BLOB_DUMP_PATH
#   define DUMP_DIR        BLOB_DUMP_PATH
#   define ENABLE_DUMP(_x) { _x ;}
#else
#   define DUMP_DIR ""
#   define ENABLE_DUMP(_x)
#endif

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

typedef std::unordered_set<MKLDNNEdgePtr> edge_cluster_t;
typedef std::vector<edge_cluster_t> edge_clusters_t;

mkldnn::engine MKLDNNGraph::eng(mkldnn::engine::kind::cpu, 0);

template<typename NET>
void MKLDNNGraph::CreateGraph(NET &net, const MKLDNNExtensionManager::Ptr& extMgr,
        MKLDNNWeightsSharing::Ptr &w_cache) {
    OV_ITT_SCOPED_TASK(MKLDNNPlugin::itt::domains::MKLDNN_LT, "CreateGraph");

    if (IsReady())
        ForgetGraphData();
    // disable caching if graph was created only once
    weightsCache = config.streamExecutorConfig._streams != 1 ? w_cache : nullptr;

    Replicate(net, extMgr);
    InitGraph();
    status = Ready;
}

//template void MKLDNNGraph::CreateGraph(const TensorIterator::Body&,
//        const MKLDNNExtensionManager::Ptr&, MKLDNNWeightsSharing::Ptr&);
template void MKLDNNGraph::CreateGraph(const CNNNetwork&,
        const MKLDNNExtensionManager::Ptr&, MKLDNNWeightsSharing::Ptr&);
//template void MKLDNNGraph::CreateGraph(CNNNetwork&,
//        const MKLDNNExtensionManager::Ptr&, MKLDNNWeightsSharing::Ptr&);

//void MKLDNNGraph::Replicate(const TensorIterator::Body &subgraph, const MKLDNNExtensionManager::Ptr& extMgr) {
//    this->_name = "subgraph";
//    this->reuse_io_tensors = false;
//
//    // Map data object onto producer layer(node)
//    std::unordered_map<Data*, std::pair<MKLDNNNodePtr, int>> data2node;
//
//    // nodes which has no consumers (output or just unused). But doesn't marked as graph output.
//    // Will be stored as fake output separately.
//    std::unordered_set<DataPtr> unused_data;
//
//    // Step 1. Replicate input nodes
//    for (const auto &input : subgraph.inputs) {
//        if (input->getPrecision() == Precision::UNSPECIFIED) continue;  // const node holder
//
//        auto creator = getCreatorLayer(input).lock();
//        if (creator == nullptr) {
//            creator.reset(new CNNLayer({input->getName(), "Input", input->getTensorDesc().getPrecision()}));
//            creator->outData.push_back(input);
//        }
//
//        const MKLDNNNodePtr node(MKLDNNNode::factory().create(creator, getEngine(), extMgr, weightsCache));
//        data2node[input.get()] = {node, 0};
//
//        graphNodes.push_back(node);
//        inputNodesMap[input->getName()] = node;
//
//        if (getInputTo(input).empty()) {
//            unused_data.insert(input);
//        }
//    }
//
//    // Step 2. Replicate all internal nodes.
//    for (const auto layer : NetPass::TIBodySortTopologically(subgraph)) {
//        const MKLDNNNodePtr node {MKLDNNNode::factory().create(layer, getEngine(), extMgr, weightsCache)};
//        graphNodes.push_back(node);
//
//        for (int port = 0; port < layer->insData.size(); port++) {
//            auto data = layer->insData[port].lock();
//
//            auto port_info = data2node[data.get()];
//            auto parent_node = port_info.first;
//            auto parent_port_idx = port_info.second;
//
//            MKLDNNEdgePtr edge(new MKLDNNEdge(parent_node, node, parent_port_idx, port));
//            node->addEdge(edge);
//            graphEdges.push_back(edge);
//        }
//        int out_port_idx = 0;
//        for (auto &out_data : layer->outData) {
//            data2node[out_data.get()] = {node, out_port_idx++};
//            if (getInputTo(out_data).empty()) {
//                unused_data.insert(out_data);
//            }
//        }
//    }
//
//    // Step 3. Add output nodes and output stubs for unused data objects.
//    for (const auto &output : subgraph.outputs) {
//        auto port_info = data2node[output.get()];
//        auto parent_node = port_info.first;
//        auto parent_port_idx = port_info.second;
//
//        CNNLayerPtr layer(new CNNLayer({"out_" + output->getName(), "Output", output->getTensorDesc().getPrecision()}));
//        layer->insData.push_back(output);
//
//        const MKLDNNNodePtr node {MKLDNNNode::factory().create(layer, getEngine(), extMgr, weightsCache)};
//
//        MKLDNNEdgePtr edge(new MKLDNNEdge(parent_node, node, parent_port_idx, 0));
//        node->addEdge(edge);
//        graphEdges.push_back(edge);
//        graphNodes.push_back(node);
//        outputNodesMap.push_back(node);
//
//        unused_data.erase(output);
//    }
//
//    // Add stub output node for unused data
//    for (auto to_stub_data : unused_data) {
//        auto port_info = data2node[to_stub_data.get()];
//        auto parent_node = port_info.first;
//        auto parent_port_idx = port_info.second;
//
//        CNNLayerPtr layer(new CNNLayer({"stub_" + to_stub_data->getName(), "Output", to_stub_data->getTensorDesc().getPrecision()}));
//        layer->insData.push_back(to_stub_data);
//
//        const MKLDNNNodePtr node(MKLDNNNode::factory().create(layer, getEngine(), extMgr, weightsCache));
//
//        MKLDNNEdgePtr edge(new MKLDNNEdge(parent_node, node, parent_port_idx, 0));
//        node->addEdge(edge);
//        graphEdges.push_back(edge);
//        graphNodes.push_back(node);
//    }
//}

void MKLDNNGraph::Replicate(const CNNNetwork &network, const MKLDNNExtensionManager::Ptr& extMgr) {
    InputsDataMap inputsInfo = network.getInputsInfo();
    OutputsDataMap outputsInfo = network.getOutputsInfo();

    std::shared_ptr<const ngraph::Function> func = network.getFunction();
    if (!func) {
        IE_THROW() << "Function pointer inside CNNNetwork is nullptr";
    }

    auto orderedOps = func->get_ordered_ops();


//    // The input layer precision has to be equal to the InputData precision
//    std::map<std::string, Precision> changedPrecision;
//    for (const auto& input : inputs) {
//        auto inputLayer = getCreatorLayer(input.second->getInputData()).lock();
//        if (inputLayer) {
//            inputLayer->precision = inputLayer->outData[0]->getTensorDesc().getPrecision();
//        }
//    }
//
//  // TODO [NM]: unordered_map is preferred from performance perspective. Needs hash for ngraph::Node
//    std::unordered_map<ngraph::Node, MKLDNNNodePtr> op2node;
    std::map<std::shared_ptr<ngraph::Node>, MKLDNNNodePtr> op2node;
    std::deque<ngraph::Output<ngraph::Node>> unusedOutputs;  // nodes which has no consumers (output or just unused)

    auto getParentPort = [](const std::shared_ptr<ngraph::Node> op, const std::shared_ptr<ngraph::Node> parentOp, const size_t port) -> int {
        for (size_t parentPort = 0; parentPort < parentOp->get_output_size(); parentPort++) {
            if (op->input(port).get_tensor_ptr() == parentOp->output(parentPort).get_tensor_ptr()) {
                return static_cast<int>(parentPort);
            }
        }

        return -1;
    };

    // Replicate All Nodes in topological order
    for (const auto& op : orderedOps) {
        const MKLDNNNodePtr node(MKLDNNNode::factory().create(op, getEngine(), extMgr, weightsCache));
        graphNodes.push_back(node);

        if (op->get_type_info() == ngraph::op::v0::Parameter::type_info) {
            if (inputsInfo.count(node->getName()) != 0) {
                inputNodesMap[node->getName()] = node;
            }
        }

        if (op->get_type_info() == ngraph::op::v0::Result::type_info) {
            const auto &input = op->input_value(0);
            NGRAPH_SUPPRESS_DEPRECATED_START
            auto name = input.get_tensor().get_name();
            NGRAPH_SUPPRESS_DEPRECATED_END
            if (name.empty()) {
                name = ngraph::op::util::create_ie_output_name(input);
            }

            if (outputsInfo.count(name) != 0) {
                outputNodesMap[name] = node;
            }
        }

        op2node[op] = node;

        for (size_t port = 0; port < op->get_input_size(); port++) {
            auto parentOp = op->get_input_node_shared_ptr(port);

//            auto data = layer->insData[port].lock();
//            auto parent_layer = getCreatorLayer(data).lock();
//            if (!parent_layer) continue;  // no parent means that it is input data node (or memory/const layer)

            auto parentNode = op2node[parentOp];

            MKLDNNEdgePtr edge(new MKLDNNEdge(parentNode, node, getParentPort(op, parentOp, port), static_cast<int>(port)));
            node->addEdge(edge);
            graphEdges.push_back(edge);
        }

        if (!MKLDNNPlugin::one_of(op->get_type_info(),
                ngraph::op::v0::Result::type_info,
                ngraph::op::v3::Assign::type_info,
                ngraph::op::v6::Assign::type_info)) {
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
        auto newResult = std::make_shared<ngraph::op::v0::Result>(unusedOutput);
        newResult->set_friendly_name(std::string("stub_") + std::to_string(unusedOutput.get_index()) + "_" + parentNode->getName());
        const MKLDNNNodePtr outNode(MKLDNNNode::factory().create(newResult, getEngine(), extMgr, weightsCache));
        MKLDNNEdgePtr edge(new MKLDNNEdge(parentNode, outNode, unusedOutput.get_index(), 0));
        outNode->addEdge(edge);
        graphEdges.push_back(edge);
        graphNodes.push_back(outNode);
    }
//
//    // Replicate input nodes
//    for (const auto& input : inputs) {
//        auto inputLayer = getCreatorLayer(input.second->getInputData()).lock();
//        inputNodesMap[input.first] = layer2node[inputLayer];
//
//        // Loading mean images
//        MKLDNNDims outDims;
//        if (!inputNodesMap[input.first]->getChildEdgeAt(0)->getDims().ndims())
//            outDims = MKLDNNDims(InferenceEngine::SizeVector(1, 1));
//        else
//            outDims = MKLDNNDims(inputNodesMap[input.first]->getChildEdgeAt(0)->getDims());
//        if (inputs.find(input.first) != inputs.end()) {
//            InputInfo::Ptr ii = inputs[input.first];
//            if (ii && ii->getPreProcess().getNumberOfChannels()) {
//                _meanImages[input.first].Load(outDims, ii);
//            }
//        }
//    }
}

void MKLDNNGraph::InitGraph() {
    MKLDNNGraphOptimizer optimizer;

    SortTopologically();
    InitNodes();

    optimizer.ApplyCommonGraphOptimizations(*this);
    SortTopologically();

    InitDescriptors();

    InitOptimalPrimitiveDescriptors();

    InitEdges();

    optimizer.ApplyImplSpecificGraphOptimizations(*this);
    SortTopologically();

    Allocate();

    CreatePrimitives();

//    if (!config.dumpToDot.empty())
//        dumpToDotFile(config.dumpToDot + "_init.dot");

#ifndef DUMP_INTERNAL_BLOBS
    for (auto &graphNode : graphNodes) {
        graphNode->cleanup();
    }
#endif

#if !defined(NDEBUG) && defined(PRINT_GRAPH_INFO)
    for (auto &graphNode : graphNodes) {
        std::cout << "name: " << graphNode->getName() << " [ ";
        if (graphNode->parentEdges.size() > 0) {
            auto prnt_out_desc = graphNode->parentEdges[0].lock()->getOutputDesc();
            std::cout << "in: " << prnt_out_desc.getPrecision().name()
                      << "/l=" << prnt_out_desc.getLayout()
                    << "; ";
        }
        if (graphNode->childEdges.size() > 0) {
            auto chld_in_desc = graphNode->childEdges[0].lock()->getInputDesc();
            std::cout << "out: " << chld_in_desc.getPrecision().name()
                      << "/l=" << chld_in_desc.getLayout();
        }
        std::cout << " ]"  << std::endl;
    }
#endif

    ExecuteConstantNodesOnly();
}

void MKLDNNGraph::InitNodes() {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNN_LT, "MKLDNNGraph::InitNodes");
    for (auto &node : graphNodes) {
        node->init();
    }
}

void MKLDNNGraph::InitDescriptors() {
    OV_ITT_TASK_CHAIN(taskChain, MKLDNNPlugin::itt::domains::MKLDNN_LT, "InitDescriptors", "Prepare");

    for (auto &node : graphNodes) {
        if (node->getType() == Input && _meanImages.find(node->getName()) != _meanImages.end()) {
            auto *inputNode = dynamic_cast<MKLDNNInputNode *>(node.get());
            if (inputNode)
                inputNode->withMeanImage();
        }
        OV_ITT_TASK_NEXT(taskChain, node->profiling.getSupportedDescriptors);
        node->getSupportedDescriptors();

        OV_ITT_TASK_NEXT(taskChain, node->profiling.initSupportedPrimitiveDescriptors);
        node->initSupportedPrimitiveDescriptors();

        OV_ITT_TASK_NEXT(taskChain, node->profiling.filterSupportedPrimitiveDescriptors);
        node->filterSupportedPrimitiveDescriptors();
    }

    for (auto &node : graphNodes) {
        OV_ITT_TASK_NEXT(taskChain, node->profiling.selectOptimalPrimitiveDescriptor);
        node->selectOptimalPrimitiveDescriptor();
    }
}

void MKLDNNGraph::InitOptimalPrimitiveDescriptors() {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, "MKLDNNGraph::InitOptimalPrimitiveDescriptors");
    for (auto &node : graphNodes) {
        OV_ITT_SCOPED_TASK(itt::domains::MKLDNN_LT, node->profiling.initOptimalPrimitiveDescriptor);
        node->initOptimalPrimitiveDescriptor();
    }
}

void MKLDNNGraph::ExecuteConstantNodesOnly() {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNN_LT, "MKLDNNGraph::ExecuteConstantNodesOnly");
    mkldnn::stream stream(eng);

    using shared_memory_ptr = MKLDNNWeightsSharing::MKLDNNSharedMemory::Ptr;

    auto acquireSharedOutputs = [this](MKLDNNNodePtr & graphNode) {
        std::vector<shared_memory_ptr> outputs;
        bool hasLocalAllocatedEdges = false;
        bool hasExternalInvalidEdges = false;

        for (size_t i = 0; i < graphNode->getChildEdges().size(); ++i) {
            auto edgePtr = graphNode->getChildEdgeAt(i);
            if (edgePtr) {
                if (edgePtr->isUseExternalMemory()) {
                    auto ptr = weightsCache->get(edgePtr->name());
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

    for (auto &graphNode : graphNodes) {
        if (!graphNode->isConstant())
            continue;

        if (weightsCache) {
            auto sharedOutputs = acquireSharedOutputs(graphNode);

            if (std::get<0>(sharedOutputs) || std::get<1>(sharedOutputs)) {
                graphNode->execute(stream);

                for (auto & output : std::get<2>(sharedOutputs))
                    output->valid(true);
            }
        } else {
            graphNode->execute(stream);
        }
    }
}

static bool isReorderAvailable(const TensorDesc& parentDesc, const TensorDesc& childDesc, const mkldnn::engine& eng) {
    memory::desc dstMemDesc = MKLDNNMemoryDesc(childDesc);
    memory::desc srcMemDesc = MKLDNNMemoryDesc(parentDesc);
    mkldnn::primitive_attr attr;

    dnnl_primitive_desc_t result = nullptr;
    auto status = dnnl_reorder_primitive_desc_create(&result, &srcMemDesc.data, eng.get(), &dstMemDesc.data, eng.get(),
                                                     attr.get());
    if (result) {
        mkldnn_primitive_desc_destroy(result);
    }

    return mkldnn_success == status;
}

void MKLDNNGraph::InitEdges() {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNN_LT, "MKLDNNGraph::InitEdges");

    size_t numberOfEdges = graphEdges.size();

    std::unordered_set<std::string> uniqueLayerNames;
    for (auto node : graphNodes) {
        uniqueLayerNames.insert(node->getName());
    }

    for (auto i = 0; i < numberOfEdges; i++) {
        if (graphEdges[i]->needReorder()) {
            auto edge = graphEdges[i];
            bool insertReorder = true;

            // Check if there is a reorder that supports the type conversion
            if (edge->getInputDesc().getPrecision() != edge->getOutputDesc().getPrecision() &&
                !isReorderAvailable(edge->getInputDesc(), edge->getOutputDesc(), this->getEngine())) {
                IE_THROW() << "[NM] Not implemented";
//                //If we are here, then we need to insert Convert, because there are no reorders that support such type conversion
//                std::string convertName = edge->getParent()->getName() + "_" +
//                                          edge->getInputDesc().getPrecision().name() + "_" + edge->getOutputDesc().getPrecision().name();
//
//                CNNLayerPtr convert(new CNNLayer(LayerParams{convertName, "Convert", edge->getInputDesc().getPrecision()}));
//                auto convertNode = std::make_shared<MKLDNNConvertNode>(convert, this->getEngine(), this->weightsCache);
//                convertNode->setDescs(edge->getInputDesc(), edge->getOutputDesc());
//                InsertNode(edge, convertNode, true);
//
//                //Check if reorder is still needed
//                if (convertNode->getChildEdgeAt(0)->needReorder()) {
//                    edge = convertNode->getChildEdgeAt(0);
//                } else {
//                    insertReorder = false;
//                }
            }

            if (insertReorder) {
                std::string basicLayerName = edge->getParent()->getName() + "_" +
                                             MKLDNNExtensionUtils::getReorderArgs(edge->getInputDesc(), edge->getOutputDesc()) + "_" +
                                             edge->getChild()->getName();
                std::string layerName = basicLayerName;
                int idx = 0;
                while (uniqueLayerNames.find(layerName) != uniqueLayerNames.end()) {
                    idx++;
                    layerName = basicLayerName + "_" + std::to_string(idx);
                }
                uniqueLayerNames.insert(layerName);
                InsertReorder(edge, layerName, edge->getInputDesc(), edge->getOutputDesc());
            }
            graphEdges.erase(graphEdges.begin() + i);
            i--;
            numberOfEdges--;
        }
    }
}

static inline bool isConstOutput(MKLDNNEdgePtr edge) {
    return edge->getParent()->isConstant() && !edge->getChild()->isConstant();
}

static edge_clusters_t findEdgeClusters(const std::vector<MKLDNNEdgePtr> & graphEdges) {
    typedef std::unordered_map<MKLDNNEdgePtr, size_t> edge_cluster_idx_map_t;

    edge_clusters_t edge_clusters;
    edge_cluster_idx_map_t edge_cluster_indices;

    for (auto &edge : graphEdges) {
        auto edge_it = edge_cluster_indices.find(edge);

        if (edge_it != edge_cluster_indices.end())
            continue;   // edge is visited

        size_t cluster_idx = edge_clusters.size();
        MKLDNNEdgePtr last_shared_edge = nullptr;

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

void MKLDNNGraph::AllocateWithReuse() {
    edge_clusters_t edge_clusters = findEdgeClusters(graphEdges);

    size_t edge_clusters_count = edge_clusters.size();

    for (size_t i = 0; i < edge_clusters_count;) {
        auto &cluster = edge_clusters[i];
        bool erase = false;
        for (auto &edge : cluster) {
            if (edge->getStatus() == MKLDNNEdge::Status::NeedAllocation
                && edge->getParent()->isConstant()) {
                edge->externalAllocate(weightsCache);
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

    std::vector<MemorySolver::Box> boxes(edge_clusters.size());
    for (int i = 0; i < edge_clusters.size(); i++) {
        MemorySolver::Box &box = boxes[i];
        box = { std::numeric_limits<int>::max(), 0, 0, i };
        for (auto &edge : edge_clusters[i]) {
            int e_start = edge->getParent()->execIndex;
            int e_finish = edge->getChild()->execIndex;

            const BlockingDesc block_desk = edge->getDesc().getBlockingDesc();

            int64_t e_size = block_desk.getOffsetPadding() + 1;  // size in bytes (from begin of data to last element)
            for (int j = 0; j < block_desk.getBlockDims().size(); j++)
                e_size += (block_desk.getBlockDims()[j] - 1) * block_desk.getStrides()[j];

            // In some cases computational formula above doesn't work properly (e.g. for OhIw8o4i layout).
            // This WA allows to limit the size of allocated memory from below.
            // TODO: need to properly investigate the root cause of incorrect computations
            int64_t min_size = 1;
            for (int64_t dim : block_desk.getBlockDims()) {
                min_size *= dim;
            }
            e_size = std::max(e_size, min_size);

            e_size *= edge->getDesc().getPrecision() == Precision::BIN ? 1 : edge->getDesc().getPrecision().size();

            box.start = std::min(e_start, box.start);
            box.finish = std::max(e_finish, box.finish);
            box.size =  std::max(e_size, box.size);
        }

        // Constant data are filled once on load.
        // So we need it untouchable during all execution time
        // -1 is a place holder for a max timestamp.
        bool isConst = false, isOutput = false, isInput = false;
        for (auto &edge : edge_clusters[i]) {
            isConst  |= isConstOutput(edge);
            isOutput |= edge->getChild()->getType() == Output;
            isInput  |= edge->getParent()->getType() == Input;
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

        box.size = div_up(box.size, alignment);
    }

    MemorySolver memSolver(boxes);
    size_t total_size = static_cast<size_t>(memSolver.solve()) * alignment;

    memWorkspace = std::make_shared<MKLDNNMemory>(eng);
    memWorkspace->Create(MKLDNNMemoryDesc(TensorDesc(Precision::I8, {total_size}, Layout::C)));

    if (edge_clusters.empty())
        return;

    auto* workspace_ptr = static_cast<int8_t*>(memWorkspace->GetData());

    for (int i = 0; i < edge_clusters.size(); i++) {
        int count = 0;
        for (auto &edge : edge_clusters[i]) {
            if (edge->getStatus() == MKLDNNEdge::Status::NeedAllocation) {
                int64_t offset = memSolver.getOffset(i);
                // !! Fallback to individual memory allocation !!
                // if you like to check infer without reuse just call this function without arguments.
                edge->allocate(workspace_ptr + offset * alignment);  // alignment in byte

                // TODO: WA for some test (like strided_slice_test) which use tensors with
                //       shapes {0}. And it is implisitly converted into {1} tensor.
                //       Zeroing of input data allow pass tests.
                if (edge->getParent()->type == Input)
                    edge->getMemoryPtr()->FillZero();

                count++;
            }
        }
        IE_ASSERT(count == 1);
    }
}

void MKLDNNGraph::Allocate() {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNN_LT, "MKLDNNGraph::Allocate");

    // resolve edges. Define which will be a view on others
    //   NeedAllocation - real blob
    //   NotAllocated - view on other blob, peer or in-place
    for (auto& edge : graphEdges) edge->init();

    // Allocate memory space for all edges marked with NeedAllocation
    AllocateWithReuse();

    // Resolve all other edges with status NotAllocated or in-place
    for (auto& node : graphNodes) node->resolveNotAllocatedEdges();

    // Check all getters. Should work.
    for (auto& edge : graphEdges) edge->validate();
}

void MKLDNNGraph::CreatePrimitives() {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, "MKLDNNGraph::CreatePrimitives");
    for (auto& node : graphNodes) {
        OV_ITT_SCOPED_TASK(itt::domains::MKLDNN_LT, node->profiling.createPrimitive);
        node->createPrimitive();
    }
}

void MKLDNNGraph::PushInputData(const std::string& name, const InferenceEngine::Blob::Ptr &in) {
    if (!IsReady()) IE_THROW()<< "Wrong state. Topology not ready.";

    auto input = inputNodesMap.find(name);
    if (input != inputNodesMap.end()) {
        MKLDNNDims outDims = input->second->getChildEdgeAt(0)->getDims();

        const void *ext_data_ptr = in->cbuffer();
        void *inter_data_ptr = input->second->getChildEdgeAt(0)->getMemory().GetData();

        if (ext_data_ptr != inter_data_ptr) {
            auto ext_tdesc = MKLDNNMemoryDesc {in->getTensorDesc()};

            auto ext_mem = MKLDNNMemory(eng);
            ext_mem.Create(ext_tdesc, ext_data_ptr, false);

            input->second->getChildEdgeAt(0)->getMemory().SetData(ext_mem, 0, false);
        }

        // todo: make sure 'name' exists in this map...
        if (_meanImages.find(name) != _meanImages.end()) {
            if (in->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
                _meanImages[name].Subtract(outDims, reinterpret_cast<float *>(inter_data_ptr), in->getTensorDesc().getLayout());
            } else {
                IE_THROW() << "Mean image of type " << in->getTensorDesc().getPrecision().name() << " is unsupported";
            }
        }
    } else {
        IE_THROW() << "Input blob for infer '" << name << "' doesn't correspond to input in network";
    }
}

void MKLDNNGraph::PullOutputData(BlobMap &out) {
    if (!IsReady())
        IE_THROW() << "Wrong state. Topology not ready.";

    for (auto &outputMap : outputNodesMap) {
        auto name = outputMap.first;
        auto node = outputMap.second;
        // remove out_ from node name
//        std::string name = node->getName().substr(4);
        const MKLDNNMemory& intr_blob = node->getParentEdgeAt(0)->getMemory();
        if (out.find(name) == out.end()) {
            // TODO [NM]: Do we really need this path?
            // TODO: Create blob from MemoryDesc
            Blob::Ptr outBlob = make_shared_blob<float>({Precision::FP32, node->getParentEdgeAt(0)->getDims().ToSizeVector(),
                                                         TensorDesc::getLayoutByDims(node->getParentEdgeAt(0)->getDims().ToSizeVector())},
                                                        reinterpret_cast<float*>(intr_blob.GetData()));
            out[name] = outBlob;
        }

        Blob::Ptr &ext_blob = out[name];

        // TODO: Why we allow allocation of output memory inside Infer call??
        // Suggestion is to disable this behaviour
        if (ext_blob->buffer() == nullptr) {
            ext_blob->allocate();
        }

        auto srcPrec = MKLDNNExtensionUtils::DataTypeToIEPrecision(intr_blob.GetDataType());
        auto dstPrec = ext_blob->getTensorDesc().getPrecision();
        if (srcPrec == dstPrec && ext_blob->byteSize() != intr_blob.GetSize())
                IE_THROW() << "Output blob byte size is not equal network output byte size ("
                                   << ext_blob->byteSize() << "!=" << intr_blob.GetSize() << ").";
        if (ext_blob->size() != intr_blob.GetElementsCount())
            IE_THROW() << "Output blob number of elements is not equal network output number of elements ("
                               << ext_blob->size() << "!=" << intr_blob.GetElementsCount() << ").";

        void *ext_blob_ptr = ext_blob->buffer();
        void *intr_blob_ptr = intr_blob.GetData();

        // That is the same memory. No need to copy
        if (ext_blob_ptr == intr_blob_ptr) continue;

        int MB = intr_blob.GetDims()[0];
        int MB_to_process = node->batchToProcess();
        // TODO: Should we support InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT???
        if (config.batchLimit)
            MB_to_process = std::min<int>(config.batchLimit, MB_to_process);
        size_t size_to_copy = intr_blob.GetElementsCount() * MB_to_process / MB;

        cpu_convert(intr_blob_ptr, ext_blob_ptr, srcPrec, dstPrec, size_to_copy);
    }
}

void MKLDNNGraph::Infer(MKLDNNInferRequest* request, int batch) {
    if (!IsReady()) {
        IE_THROW() << "Wrong state. Topology is not ready.";
    }

    mkldnn::stream stream(eng);

    for (int i = 0; i < graphNodes.size(); i++) {
        if (request != nullptr) {
            request->ThrowIfCanceled();
        }

        PERF(graphNodes[i]);

        if (batch > 0)
            graphNodes[i]->setDynamicBatchLim(batch);

        ENABLE_DUMP(do_before(DUMP_DIR, graphNodes[i]));

        if (!graphNodes[i]->isConstant()) {
            OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, graphNodes[i]->profiling.execute);
            graphNodes[i]->execute(stream);
        }
        ENABLE_DUMP(do_after(DUMP_DIR, graphNodes[i]));
    }

    if (infer_count != -1) infer_count++;
}

void MKLDNNGraph::VisitNode(MKLDNNNodePtr node, std::vector<MKLDNNNodePtr>& sortedNodes) {
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

void MKLDNNGraph::SortTopologically() {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNN_LT, "MKLDNNGraph::SortTopologically");

    std::vector<MKLDNNNodePtr> unsorted;
    std::vector<MKLDNNNodePtr> sorted;

    for (int i = 0; i < graphNodes.size(); i++) {
        MKLDNNNodePtr node = graphNodes[i];

        node->permanent = false;
        node->temporary = false;

        unsorted.push_back(node);
    }

    while (!unsorted.empty()) {
        MKLDNNNodePtr node = unsorted.at(0);
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
            int port_num = node->inDims.size();
            std::vector<MKLDNNEdgePtr> res(port_num);

            for (int i = 0; i < node->parentEdges.size(); i++) {
                auto edge = node->getParentEdgeAt(i);
                int port = edge->getOutputNum();
                if (!res[port])
                    res[port] = edge;
                else
                    res.push_back(edge);
            }
            node->parentEdges = {res.begin(), res.end()};
        }
        {
            int port_num = node->outDims.size();
            std::vector<MKLDNNEdgePtr> res(port_num);

            for (int i = 0; i < node->childEdges.size(); i++) {
                auto edge = node->getChildEdgeAt(i);
                int port = edge->getInputNum();
                if (!res[port])
                    res[port] = edge;
                else
                    res.push_back(edge);
            }
            node->childEdges = {res.begin(), res.end()};
        }
    }
}

void MKLDNNGraph::GetPerfData(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const {
    unsigned i = 0;
    std::function<void(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &, const MKLDNNNodePtr&)>
            getPerfMapFor = [&](std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap, const MKLDNNNodePtr& node) {
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

    for (int i = 1; i < graphNodes.size(); i++) {
        getPerfMapFor(perfMap, graphNodes[i]);
    }

//    if (!config.dumpToDot.empty()) dumpToDotFile(config.dumpToDot + "_perf.dot");
}

void MKLDNNGraph::setConfig(const Config &cfg) {
    config = cfg;
}

void MKLDNNGraph::setProperty(const std::map<std::string, std::string>& properties) {
    config.readProperties(properties);
}

Config MKLDNNGraph::getProperty() const {
    return config;
}

void MKLDNNGraph::getInputBlobs(InferenceEngine::BlobMap &resp) {
    for (auto &it : inputNodesMap) {
// TODO [NM]: Do we still need this?
//        MKLDNNInputNode* node = dynamic_cast<MKLDNNInputNode*>(it.second.get());
//        if (!node || node->isConstant())
//            continue;
        resp[it.first] = it.second->getChildEdgeAt(0)->getBlob();
    }
}

void MKLDNNGraph::getOutputBlobs(InferenceEngine::BlobMap &resp) {
    for (auto &it : outputNodesMap) {
        resp[it.first] = it.second->getParentEdgeAt(0)->getBlob();
    }
}

void MKLDNNGraph::DropNode(const MKLDNNNodePtr &node) {
    auto removeEdge = [](MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == edge) {
                edges.erase(it);
                return;
            }
        }
    };

    auto childs = node->childEdges;
    auto parents = node->parentEdges;

    for (size_t i = 0; i < parents.size(); i++) {
        auto p_edge = parents[i].lock();
        if (!p_edge) continue;
        auto parent = p_edge->getParent();
        if (!parent) continue;

        for (size_t j = 0; j < childs.size(); j++) {
            if (!childs[j].lock())
                continue;
            auto child = childs[j].lock()->getChild();
            if (!child)
                continue;

            MKLDNNEdgePtr &remEdge = p_edge;
            int inNum = 0;
            if (remEdge) {
                inNum = remEdge->getInputNum();
                remEdge->drop();
                removeEdge(*this, remEdge);
            }
            remEdge = childs[j].lock();
            int outNum = 0;
            if (remEdge) {
                outNum = remEdge->getOutputNum();
                remEdge->drop();
                removeEdge(*this, remEdge);
            }
            MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, child, inNum, outNum));
            graphEdges.push_back(newEdge);
            parent->addEdge(newEdge);
        }
    }
}

void MKLDNNGraph::DropDWConvNode(const MKLDNNNodePtr &node) {
    auto removeEdge = [](MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == edge) {
                edges.erase(it);
                return;
            }
        }
    };

    auto childs = node->childEdges;
    auto parents = node->parentEdges;

    auto parentConvEdge = parents[0].lock();
    if (!parentConvEdge) return;
    auto parentConv = parentConvEdge->getParent();
    if (!parentConv) return;

    for (size_t i = 0; i < 1; i++) {
        auto p_edge = parents[i].lock();
        if (!p_edge) continue;
        auto parent = p_edge->getParent();
        if (!parent) continue;

        for (size_t j = 0; j < childs.size(); j++) {
            if (!childs[j].lock())
                continue;
            auto child = childs[j].lock()->getChild();
            if (!child)
                continue;

            MKLDNNEdgePtr &remEdge = p_edge;
            int inNum = 0;
            if (remEdge) {
                inNum = remEdge->getInputNum();
                remEdge->drop();
                removeEdge(*this, remEdge);
            }
            remEdge = childs[j].lock();
            int outNum = 0;
            if (remEdge) {
                outNum = remEdge->getOutputNum();
                remEdge->drop();
                removeEdge(*this, remEdge);
            }
            MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, child, inNum, outNum));
            graphEdges.push_back(newEdge);
            parent->addEdge(newEdge);
        }
    }

    for (size_t i = 1; i < parents.size(); i++) {
        auto p_edge = parents[i].lock();
        if (!p_edge) continue;
        auto parent = p_edge->getParent();
        if (!parent) continue;

        MKLDNNEdgePtr &remEdge = p_edge;
        int inNum = 0;
        if (remEdge) {
            inNum = remEdge->getInputNum();
            remEdge->drop();
            removeEdge(*this, remEdge);
        }
        int outNum = parentConv->parentEdges.size();

        MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, parentConv, inNum, outNum));
        graphEdges.push_back(newEdge);
        parent->addEdge(newEdge);
        parentConv->inDims.push_back(newEdge->getDims());
    }
}

void MKLDNNGraph::RemoveDroppedNodes() {
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

void MKLDNNGraph::RemoveDroppedEdges() {
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

MKLDNNNodePtr MKLDNNGraph::InsertReorder(MKLDNNEdgePtr edge, std::string layerName, const TensorDesc& inDesc, const TensorDesc& outDesc,
                                bool isOptimized, InferenceEngine::Blob::Ptr scales) {
    MKLDNNNodePtr newReorder(new MKLDNNReorderNode(layerName, getEngine(), weightsCache));
    auto *reorderPtr = dynamic_cast<MKLDNNReorderNode *>(newReorder.get());
    if (reorderPtr == nullptr) {
        IE_THROW() << "MKLDNNGraph::InsertReorder: Cannot cast to MKLDNNReorderNode";
    }
    reorderPtr->setDescs(inDesc, outDesc);
    reorderPtr->_scales = scales;
    reorderPtr->setOptimized(isOptimized);

    InsertNode(edge, newReorder, true);

    // Using the method MKLDNNEdge::getDesc() we can check that input and output tensor descriptors are equal.
    // Due to the specificity of MKLDNNGraphOptimizer::MergeTransposeAndReorder() that isOptimized flag uses, we shouldn't do these checks.
    if (!isOptimized) {
        newReorder->getParentEdgeAt(0)->getDesc();
        newReorder->getChildEdgeAt(0)->getDesc();
    }

    return newReorder;
}

void MKLDNNGraph::do_before(const std::string &dir, const MKLDNNNodePtr &node) {
    auto exec_order = std::to_string(node->execIndex);
    std::string nodeName = node->name;
    std::replace(nodeName.begin(), nodeName.end(), '\\', '_');
    std::replace(nodeName.begin(), nodeName.end(), '/', '_');
    std::replace(nodeName.begin(), nodeName.end(), ' ', '_');
    std::replace(nodeName.begin(), nodeName.end(), ':', '-');

    auto num_ports = node->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size();
    for (size_t i = 0; i < num_ports; i++) {
        auto prEdge = node->getParentEdgeAt(i);
        auto pr = prEdge->getParent();

        std::string file_name = nodeName;
        if (infer_count != -1) file_name += "_iter" + std::to_string(infer_count);
        file_name += "_in" + std::to_string(i) + ".ieb";
        if (file_name.size() > 240)
            file_name = file_name.substr(file_name.size() - 240);


        auto dump_file = dir + "/#" + exec_order + "_" + file_name;
        TensorDesc desc = prEdge->getDesc();
        if (desc.getPrecision() == Precision::BIN)
            continue;

        BlobDumper dumper(prEdge->getBlob());
        if (pr->ext_scales) dumper.withScales(pr->ext_scales);
#ifdef DUMP_AS_TEXT
        dumper.dumpAsTxt(dump_file);
#else
        dumper.dump(dump_file);
#endif
    }

#ifdef DUMP_INTERNAL_BLOBS
    for (size_t i = 0; i < node->internalBlobs.size(); i++) {
        const auto& blb = node->internalBlobs[i];
        auto dump_file = dir + "/#" + exec_order + "_" +  nodeName + "_blb" + std::to_string(i) + ".ieb";
        TensorDesc desc = blb->getTensorDesc();
        if (desc.getPrecision() == Precision::BIN)
            continue;
        BlobDumper dumper(blb);
#ifdef DUMP_AS_TEXT
        dumper.dumpAsTxt(dump_file);
#else
        dumper.dump(dump_file);
#endif
    }
#endif
}

void MKLDNNGraph::do_after(const std::string &dir, const MKLDNNNodePtr &node) {
    auto exec_order = std::to_string(node->execIndex);
    auto nodeName = node->name;
    std::replace(nodeName.begin(), nodeName.end(), '\\', '_');
    std::replace(nodeName.begin(), nodeName.end(), '/', '_');
    std::replace(nodeName.begin(), nodeName.end(), ' ', '_');
    std::replace(nodeName.begin(), nodeName.end(), ':', '-');

    auto num_ports = node->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size();
    for (size_t i = 0; i < num_ports; i++) {
        auto childEdge = node->getChildEdgeAt(i);

        std::string file_name = nodeName;
        if (infer_count != -1) file_name += "_iter" + std::to_string(infer_count);
        file_name += "_out" + std::to_string(i) + ".ieb";
        if (file_name.size() > 240)
            file_name = file_name.substr(file_name.size() - 240);

        auto dump_file = dir + "/#" + exec_order + "_" + file_name;
        std::cout << "try : " << dump_file << std::endl;

        TensorDesc desc = childEdge->getDesc();
        if (desc.getPrecision() == Precision::BIN)
            continue;

        BlobDumper dumper(childEdge->getBlob());
        if (node->ext_scales) dumper.withScales(node->ext_scales);

#ifdef DUMP_AS_TEXT
        dumper.dumpAsTxt(dump_file);
#else
        dumper.dump(dump_file);
#endif
    }
}

InferenceEngine::CNNNetwork MKLDNNGraph::dump() const {
    return dump_graph_as_ie_ngraph_net(*this);
}

bool MKLDNNGraph::InsertNode(MKLDNNEdgePtr edge, MKLDNNNodePtr node, bool initNode) {
    auto oIndex = edge->getOutputNum();
    auto iIndex = edge->getInputNum();
    if (iIndex < 0 || oIndex < 0)
        IE_THROW() << "Cannot insert node '" << node->getName() << "' between nodes: "
                           << edge->getParent()->getName() << " and "
                           << edge->getChild()->getName() << ".";

    edge->drop();

    return InsertNode(edge->getParent(), edge->getChild(), node, iIndex, oIndex, initNode);
}

bool MKLDNNGraph::InsertNode(MKLDNNNodePtr parent, MKLDNNNodePtr child, MKLDNNNodePtr node, int parentPort, int childPort, bool initNode) {
    MKLDNNEdgePtr beforeNode(new MKLDNNEdge(parent, node, parentPort, 0));
    MKLDNNEdgePtr afterNode(new MKLDNNEdge(node, child, 0, childPort));

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
