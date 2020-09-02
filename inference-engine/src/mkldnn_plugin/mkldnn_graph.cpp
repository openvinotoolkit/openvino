// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <string>
#include <map>
#include <vector>
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
#include <nodes/mkldnn_input_node.h>
#include <nodes/mkldnn_reorder_node.h>

#include <legacy/graph_tools.hpp>
#include <ie_algorithm.hpp>
#include <blob_factory.hpp>
#include <legacy/net_pass.h>
#include <legacy/details/ie_cnn_network_tools.h>
#include "nodes/common/cpu_memcpy.h"

#include "precision_utils.h"
#include <ie_plugin_config.hpp>
#include "low_precision_transformations/transformer.hpp"

#include "utils/blob_dump.h"

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

template<typename NET>
void MKLDNNGraph::ApplyUnrollPasses(NET &net) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, "MKLDNNGraph::ApplyUnrollPasses");

    NetPass::CombineRNNSeq(net);
    bool ti_proc_ok = NetPass::UnrollRNN_if(net, [] (const RNNCellBase &rnn) -> bool {
        if (rnn.clip != 0.0f)
            return true;
        if ((rnn.cellType == RNNCellBase::GRU || rnn.cellType == RNNCellBase::GRU_LBR) &&
            rnn.activations != std::vector<std::string> {"sigmoid", "tanh"})
            return true;
        if (rnn.cellType == RNNCellBase::LSTM &&
            rnn.activations != std::vector<std::string> {"sigmoid", "tanh", "tanh"})
            return true;
        return false;
    });
    if (!ti_proc_ok)
        THROW_IE_EXCEPTION << "Plugin doesn't support Tensor Iterator in pure form. "
                              "None TI optimization pattern has been applied successfully";
}

template void MKLDNNGraph::ApplyUnrollPasses(TensorIterator::Body&);
template void MKLDNNGraph::ApplyUnrollPasses(ICNNNetwork&);

template<typename NET>
void MKLDNNGraph::CreateGraph(const NET &net, const MKLDNNExtensionManager::Ptr& extMgr,
        MKLDNNWeightsSharing::Ptr &w_cache) {
    if (IsReady())
        ForgetGraphData();
    // disable caching if graph was created only once
    weightsCache = config.streamExecutorConfig._streams != 1 ? w_cache : nullptr;

    Replicate(net, extMgr);
    InitGraph();
    status = Ready;
}

template void MKLDNNGraph::CreateGraph(const TensorIterator::Body&,
        const MKLDNNExtensionManager::Ptr&, MKLDNNWeightsSharing::Ptr&);
template void MKLDNNGraph::CreateGraph(const ICNNNetwork&,
        const MKLDNNExtensionManager::Ptr&, MKLDNNWeightsSharing::Ptr&);
template void MKLDNNGraph::CreateGraph(const CNNNetwork&,
        const MKLDNNExtensionManager::Ptr&, MKLDNNWeightsSharing::Ptr&);

void MKLDNNGraph::Replicate(const TensorIterator::Body &subgraph, const MKLDNNExtensionManager::Ptr& extMgr) {
    this->_name = "subgraph";
    this->reuse_io_tensors = false;

    std::unordered_map<CNNLayerPtr, MKLDNNNodePtr> layer2node;
    std::unordered_set<DataPtr> unused_data;  // nodes which has no consumers (output or just unused)

    auto _parent_port = [] (const DataPtr &data) -> int {
        auto parent = getCreatorLayer(data).lock();
        for (int i = 0; parent->outData.size(); i++)
            if (data == parent->outData[i])
                return i;
        return -1;
    };

    auto _child_port = [] (const DataPtr &data, const CNNLayerPtr &layer) -> int {
        for (int i = 0; layer->insData.size(); i++)
            if (data == layer->insData[i].lock())
                return i;
        return -1;
    };


    // Replicate All Nodes in topological order
    for (const auto layer : NetPass::TIBodySortTopologically(subgraph)) {
        CNNLayerPtr _layer = layer;

        const MKLDNNNodePtr node(MKLDNNNode::CreateNode(_layer, getEngine(), extMgr, weightsCache));
        graphNodes.push_back(node);
        layer2node[layer] = node;

        for (int port = 0; port < layer->insData.size(); port++) {
            auto data = layer->insData[port].lock();
            auto parent_layer = getCreatorLayer(data).lock();
            if (!parent_layer) continue;  // no parent means that it is input data node (or memory/const layer)

            auto parent_node = layer2node[parent_layer];

            MKLDNNEdgePtr edge(new MKLDNNEdge(parent_node, node, _parent_port(data), port));
            node->addEdge(edge);
            graphEdges.push_back(edge);
        }
        for (auto &out_data : layer->outData) {
            if (getInputTo(out_data).empty()) {
                unused_data.insert(out_data);
            }
        }
    }

    for (const auto &output : subgraph.outputs) {
        auto parent_layer = getCreatorLayer(output).lock();
        auto parent_node = layer2node[parent_layer];

        CNNLayerPtr layer(new CNNLayer({"out_" + output->getName(), "Output", output->getTensorDesc().getPrecision()}));
        layer->insData.push_back(output);

        const MKLDNNNodePtr node(MKLDNNNode::CreateNode(layer, getEngine(), extMgr, weightsCache));

        MKLDNNEdgePtr edge(new MKLDNNEdge(parent_node, node, _parent_port(output), 0));
        node->addEdge(edge);
        graphEdges.push_back(edge);

        graphNodes.push_back(node);
        outputNodes.push_back(node);

        unused_data.erase(output);
    }

    // Add stub output node for unused data
    for (auto to_stub_data : unused_data) {
        auto parent_layer = getCreatorLayer(to_stub_data).lock();
        auto parent_node = layer2node[parent_layer];

        CNNLayerPtr layer(new CNNLayer({"stub_" + parent_layer->name, "Output", to_stub_data->getTensorDesc().getPrecision()}));
        layer->insData.push_back(to_stub_data);

        const MKLDNNNodePtr node(MKLDNNNode::CreateNode(layer, getEngine(), extMgr, weightsCache));

        MKLDNNEdgePtr edge(new MKLDNNEdge(parent_node, node, _parent_port(to_stub_data), 0));
        node->addEdge(edge);
        graphEdges.push_back(edge);
        graphNodes.push_back(node);
    }

    // Replicate input nodes
    for (const auto &input : subgraph.inputs) {
        if (input->getName() == "const_holder") continue;

        CNNLayerPtr layer(new CNNLayer({"in_" + input->getName(), "Input", input->getTensorDesc().getPrecision()}));
        layer->outData.push_back(input);

        const MKLDNNNodePtr node(MKLDNNNode::CreateNode(layer, getEngine(), extMgr, weightsCache));

        for (auto p : getInputTo(input)) {
            auto consumer = p.second;
            MKLDNNEdgePtr edge(new MKLDNNEdge(node, layer2node[consumer], 0, _child_port(input, consumer)));
            node->addEdge(edge);
            graphEdges.push_back(edge);
        }

        graphNodes.push_back(node);
        inputNodes[input->getName()] = node;
    }
}

void MKLDNNGraph::Replicate(const ICNNNetwork &network, const MKLDNNExtensionManager::Ptr& extMgr) {
    InputsDataMap inputs;
    network.getInputsInfo(inputs);
    if (inputs.empty()) {
        THROW_IE_EXCEPTION << "MKLDNNGraph::CreateGraph: No inputs for the topology";
    }

    this->_name = network.getName();

    // The input layer precision has to be equal to the InputData precision
    std::map<std::string, Precision> changedPrecision;
    for (const auto& input : inputs) {
        auto inputLayer = getCreatorLayer(input.second->getInputData()).lock();
        if (inputLayer) {
            inputLayer->precision = inputLayer->outData[0]->getTensorDesc().getPrecision();
        }
    }

    std::unordered_map<CNNLayerPtr, MKLDNNNodePtr> layer2node;
    std::unordered_set<DataPtr> unused_data;  // nodes which has no consumers (output or just unused)

    auto _parent_port = [] (const DataPtr &data) -> int {
        auto parent = getCreatorLayer(data).lock();
        for (int i = 0; parent->outData.size(); i++)
            if (data == parent->outData[i])
                return i;
        return -1;
    };

    // Replicate All Nodes in topological order
    for (const auto layer : CNNNetSortTopologically(network)) {
        CNNLayerPtr _layer = layer;
        if (layer->type == "Memory" && layer->GetParamAsString("index") == "1") {
            auto memoryId = layer->GetParamAsString("id");
            Precision portPrecision = layer->outData[0]->getTensorDesc().getPrecision();
            _layer.reset(new CNNLayer({layer->name + "/id=" + memoryId, "MemoryInput", portPrecision}));
            _layer->params = layer->params;
            _layer->outData = layer->outData;
        }

        const MKLDNNNodePtr node(MKLDNNNode::CreateNode(_layer, getEngine(), extMgr, weightsCache));
        graphNodes.push_back(node);
        layer2node[layer] = node;

        if (layer->params.count("originalLayersNames")) {
            node->originalLayers = layer->params["originalLayersNames"];
        }

        for (int port = 0; port < layer->insData.size(); port++) {
            auto data = layer->insData[port].lock();
            auto parent_layer = getCreatorLayer(data).lock();
            if (!parent_layer) continue;  // no parent means that it is input data node (or memory/const layer)

            auto parent_node = layer2node[parent_layer];

            MKLDNNEdgePtr edge(new MKLDNNEdge(parent_node, node, _parent_port(data), port));
            node->addEdge(edge);
            graphEdges.push_back(edge);
        }
        for (auto &out_data : layer->outData) {
            if (getInputTo(out_data).empty()) {
                unused_data.insert(out_data);
            }
        }
    }

    std::map<std::string, DataPtr> outputs;
    network.getOutputsInfo(outputs);

    for (const auto &output : outputs) {
        const auto data = output.second;

        auto parent_layer = getCreatorLayer(data).lock();
        auto parent_node = layer2node[parent_layer];

        CNNLayerPtr layer(new CNNLayer({"out_" + output.first, "Output", data->getTensorDesc().getPrecision()}));
        layer->insData.push_back(data);

        const MKLDNNNodePtr node(MKLDNNNode::CreateNode(layer, getEngine(), extMgr, weightsCache));

        MKLDNNEdgePtr edge(new MKLDNNEdge(parent_node, node, _parent_port(data), 0));
        node->addEdge(edge);
        graphEdges.push_back(edge);

        graphNodes.push_back(node);
        outputNodes.push_back(node);

        unused_data.erase(data);
    }

    // Add stub output node for unused data
    for (auto to_stub_data : unused_data) {
        auto parent_layer = getCreatorLayer(to_stub_data).lock();
        auto parent_node = layer2node[parent_layer];

        CNNLayerPtr layer(new CNNLayer({"stub_" + parent_layer->name, "Output", to_stub_data->getTensorDesc().getPrecision()}));
        layer->insData.push_back(to_stub_data);

        const MKLDNNNodePtr node(MKLDNNNode::CreateNode(layer, getEngine(), extMgr, weightsCache));

        MKLDNNEdgePtr edge(new MKLDNNEdge(parent_node, node, _parent_port(to_stub_data), 0));
        node->addEdge(edge);
        graphEdges.push_back(edge);
        graphNodes.push_back(node);
    }

    // Replicate input nodes
    for (const auto& input : inputs) {
        auto inputLayer = getCreatorLayer(input.second->getInputData()).lock();
        inputNodes[input.first] = layer2node[inputLayer];

        // Loading mean images
        MKLDNNDims outDims;
        if (!inputNodes[input.first]->getChildEdgeAt(0)->getDims().ndims())
            outDims = MKLDNNDims(InferenceEngine::SizeVector(1, 1));
        else
            outDims = MKLDNNDims(inputNodes[input.first]->getChildEdgeAt(0)->getDims());
        if (inputs.find(input.first) != inputs.end()) {
            InputInfo::Ptr ii = inputs[input.first];
            if (ii && ii->getPreProcess().getNumberOfChannels()) {
                _meanImages[input.first].Load(outDims, ii);
            }
        }
    }
}

void MKLDNNGraph::InitGraph() {
    MKLDNNGraphOptimizer optimizer;

    SortTopologically();
    InitNodes();
    optimizer.ApplyCommonGraphOptimizations(*this);
    SortTopologically();

    InitDescriptors();

    for (auto &node : graphNodes) {
        node->initOptimalPrimitiveDescriptor();
    }
    InitEdges();

    optimizer.ApplyImplSpecificGraphOptimizations(*this);

    SortTopologically();

    Allocate();

    CreatePrimitives();

    // Do it before cleanup. Because it will lose original layers information
    for (auto &graphNode : graphNodes) {
        auto nodeType = graphNode->getType();
        if (nodeType == Reorder || nodeType == Output) continue;

        if (graphNode->getOriginalLayers().empty()) {
            graphNode->addOriginalLayer(graphNode->getCnnLayer());
        }

        if (graphNode->getFusedWith().size() || graphNode->getMergeWith().size()) {
            // Original layer names
            std::vector<MKLDNNNodePtr> internal = graphNode->getFusedWith();
            auto &merged = graphNode->getMergeWith();
            internal.insert(internal.end(), merged.begin(), merged.end());

            for (auto &sub_node : internal) {
                graphNode->addOriginalLayer(sub_node->getCnnLayer());
            }
        }
    }
    if (!config.dumpToDot.empty())
        dumpToDotFile(config.dumpToDot + "_init.dot");

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

    mkldnn::stream stream = mkldnn::stream(stream::kind::eager);
    for (auto &graphNode : graphNodes) {
        if (!graphNode->isConstant())
            continue;
        graphNode->execute(stream);
    }
}

void MKLDNNGraph::InitNodes() {
    for (auto &node : graphNodes) {
        node->init();
    }
}

void MKLDNNGraph::InitDescriptors() {
    for (auto &node : graphNodes) {
#if defined (COMPILED_CPU_MKLDNN_INPUT_NODE)
        if (node->getType() == Input && _meanImages.find(node->getName()) != _meanImages.end()) {
            auto *inputNode = dynamic_cast<MKLDNNInputNode *>(node.get());
            if (inputNode)
                inputNode->withMeanImage();
        }
#endif
        node->getSupportedDescriptors();

        node->initSupportedPrimitiveDescriptors();
        node->filterSupportedPrimitiveDescriptors();
    }

    for (auto &node : graphNodes) {
        node->selectOptimalPrimitiveDescriptor();
    }
}

void MKLDNNGraph::InitEdges() {
    auto reorderArgs = [](const InferenceEngine::TensorDesc &parentDesc, const InferenceEngine::TensorDesc &childDesc) {
        std::string inArgs, outArgs;
        if (parentDesc.getPrecision() != childDesc.getPrecision()) {
            inArgs += (inArgs.empty() ? "" : "_") + std::string(parentDesc.getPrecision().name());
            outArgs += (outArgs.empty() ? "" : "_") + std::string(childDesc.getPrecision().name());
        }
        if (MKLDNNMemoryDesc(parentDesc).getFormat() != MKLDNNMemoryDesc(childDesc).getFormat()) {
            inArgs += (inArgs.empty() ? "" : "_") + MKLDNNMemory::formatToString(MKLDNNMemoryDesc(parentDesc).getFormat());
            outArgs += (outArgs.empty() ? "" : "_") + MKLDNNMemory::formatToString(MKLDNNMemoryDesc(childDesc).getFormat());
        }
        return inArgs + "_" + outArgs;
    };
    size_t numberOfEdges = graphEdges.size();

    std::unordered_set<std::string> uniqueLayerNames;
    for (auto node : graphNodes) {
        uniqueLayerNames.insert(node->getCnnLayer()->name);
    }

    for (auto i = 0; i < numberOfEdges; i++) {
        if (graphEdges[i]->needReorder()) {
#if defined (COMPILED_CPU_MKLDNN_REORDER_NODE)
            auto &edge = graphEdges[i];
            std::string basicLayerName = edge->getParent()->getName() + "_" +
                                         reorderArgs(edge->getInputDesc(), edge->getOutputDesc()) + "_" +
                                         edge->getChild()->getName();
            std::string layerName = basicLayerName;
            int idx = 0;
            while (uniqueLayerNames.find(layerName) != uniqueLayerNames.end()) {
                idx++;
                layerName = basicLayerName + "_" + std::to_string(idx);
            }
            uniqueLayerNames.insert(layerName);
            CNNLayerPtr layer(new CNNLayer({layerName,
                                            "Reorder",
                                            edge->getInputDesc().getPrecision()}));
            MKLDNNNodePtr newReorder(new MKLDNNReorderNode(layer, getEngine(), weightsCache));
            auto *reorderPtr = dynamic_cast<MKLDNNReorderNode *>(newReorder.get());
            if (reorderPtr) {
                reorderPtr->setDescs(edge->getInputDesc(), edge->getOutputDesc());
            }

            auto oIndex = edge->getOutputNum();
            auto iIndex = edge->getInputNum();
            if (iIndex < 0 || oIndex < 0)
                THROW_IE_EXCEPTION << "Cannot create reorder for nodes: "
                                   << edge->getParent()->getName() << " and "
                                   << edge->getChild()->getName() << ".";

            edge->drop();

            MKLDNNEdgePtr beforeNode(new MKLDNNEdge(edge->getParent(), newReorder, iIndex, 0));
            MKLDNNEdgePtr afterNode(new MKLDNNEdge(newReorder, edge->getChild(), 0, oIndex));

            // Add edge for beforeNode
            beforeNode->getChild()->parentEdges.push_back(beforeNode);
            edge->getParent()->childEdges.push_back(beforeNode);

            // Add edge for afterNode
            afterNode->getParent()->childEdges.push_back(afterNode);
            edge->getChild()->parentEdges.push_back(afterNode);

            newReorder->getSupportedDescriptors();
            newReorder->initSupportedPrimitiveDescriptors();
            newReorder->selectOptimalPrimitiveDescriptor();

            graphEdges.push_back(beforeNode);
            graphEdges.push_back(afterNode);

            graphNodes.push_back(newReorder);
            graphEdges.erase(graphEdges.begin() + i);
            i--;
            numberOfEdges--;
#else
            THROW_IE_EXCEPTION << "CPU Plugin doesn't contains reorder layer";
#endif
        }
    }
}

static inline bool isConstOutput(MKLDNNEdgePtr edge) {
    return edge->getParent()->isConstant() && !edge->getChild()->isConstant();
}

void MKLDNNGraph::AllocateWithReuse() {
    std::vector<std::vector<MKLDNNEdgePtr>> edge_clasters;

    // detect edge clusters which are view on one.
    for (auto &edge : graphEdges) {
        MKLDNNEdgePtr par = (edge->getStatus() == MKLDNNEdge::Status::NotAllocated)
                            ? edge->getSharedEdge()
                            : nullptr;
        if (par) {
            bool found = false;
            for (auto &claster : edge_clasters) {
                for (auto &element : claster) {
                    if (element == par) {
                        if (std::find(claster.begin(), claster.end(), edge) == claster.end())
                            claster.push_back(edge);
                        found = true;
                        break;
                    }
                }
            }
            if (!found)
                edge_clasters.push_back({par, edge});
        } else {
            bool found = false;
            for (auto &claster : edge_clasters) {
                for (auto &element : claster) {
                    if (element == edge) {
                        found = true;
                        break;
                    }
                }
            }
            if (!found)
                edge_clasters.push_back({edge});
        }
    }

    //======= WA. getSharedEdge() returns not identical edges ============
    //  Will try to merge clasters with matched edges
    for (auto &edge : graphEdges) {
        std::vector<decltype(&edge_clasters[0])> to_merge;

        for (auto &claster : edge_clasters)
            if (std::find(claster.begin(), claster.end(), edge) != claster.end())
                to_merge.push_back(&claster);

        if (to_merge.size() > 1) {
            // Merge clasters
            auto base_classter = to_merge[0];
            for (int i = 1; i < to_merge.size(); i++) {
                base_classter->insert(base_classter->end(),
                                      to_merge[i]->begin(), to_merge[i]->end());
                to_merge[i]->clear();
            }

            // remove duplicates in merged claster
            std::sort(base_classter->begin(), base_classter->end());
            base_classter->erase(std::unique(base_classter->begin(), base_classter->end()),
                                 base_classter->end() );

            // remove empty clasters
            edge_clasters.erase(std::remove_if(edge_clasters.begin(), edge_clasters.end(),
                                               [] ( std::vector<MKLDNNEdgePtr> &cls) { return cls.empty(); }),
                                edge_clasters.end());
        }
    }
    //======= End of WA ============

    const int64_t alignment = 32;  // 32 bytes

    std::vector<MemorySolver::Box> boxes(edge_clasters.size());
    for (int i = 0; i < edge_clasters.size(); i++) {
        MemorySolver::Box &box = boxes[i];
        box = { std::numeric_limits<int>::max(), 0, 0, i };
        for (auto &edge : edge_clasters[i]) {
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
        for (auto &edge : edge_clasters[i]) {
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
    auto* workspace_ptr = static_cast<int8_t*>(memWorkspace->GetData());

    for (int i = 0; i < edge_clasters.size(); i++) {
        int count = 0;
        for (auto &edge : edge_clasters[i]) {
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
        node->createPrimitive();
    }
}

void MKLDNNGraph::PushInputData(const std::string& name, const InferenceEngine::Blob::Ptr &in) {
    if (!IsReady()) THROW_IE_EXCEPTION<< "Wrong state. Topology not ready.";

    auto input = inputNodes.find(name);
    if (input != inputNodes.end()) {
        MKLDNNDims outDims = input->second->getChildEdgeAt(0)->getDims();

        const void *ext_data_ptr = in->cbuffer();
        void *inter_data_ptr = input->second->getChildEdgeAt(0)->getMemory().GetData();

        if (ext_data_ptr != inter_data_ptr) {
            auto l = in->getTensorDesc().getLayout();
            if (l == CHW && input->second->getChildEdgeAt(0)->getDims().ndims() == 4)
                l = NCHW;

            input->second->getChildEdgeAt(0)->getMemory().SetData(
                    MKLDNNExtensionUtils::IEPrecisionToDataType(in->getTensorDesc().getPrecision()),
                    MKLDNNMemory::Convert(l), ext_data_ptr, in->byteSize(), false);
        }

        // todo: make sure 'name' exists in this map...
        if (_meanImages.find(name) != _meanImages.end()) {
            if (in->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
                _meanImages[name].Subtract(outDims, reinterpret_cast<float *>(inter_data_ptr), in->getTensorDesc().getLayout());
            } else {
                THROW_IE_EXCEPTION << "Mean image of type " << in->getTensorDesc().getPrecision().name() << " is unsupported";
            }
        }
    } else {
        THROW_IE_EXCEPTION << "Input blob for infer '" << name << "' doesn't correspond to input in network";
    }
}

void MKLDNNGraph::PullOutputData(BlobMap &out) {
    if (!IsReady())
        THROW_IE_EXCEPTION << "Wrong state. Topology not ready.";

    for (MKLDNNNodePtr &node : outputNodes) {
        // remove out_ from node name
        std::string name = node->getName().substr(4);
        const MKLDNNMemory& intr_blob = node->getParentEdgeAt(0)->getMemory();
        if (out.find(name) == out.end()) {
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

        if (ext_blob->byteSize() != intr_blob.GetSize())
            THROW_IE_EXCEPTION << "Output blob size is not equal network output size ("
                               << ext_blob->size() << "!=" << intr_blob.GetSize()/sizeof(float) << ").";

        void *ext_blob_ptr = ext_blob->buffer();
        void *intr_blob_ptr = intr_blob.GetData();

        // That is the same memory. No need to copy
        if (ext_blob_ptr == intr_blob_ptr) continue;

        int MB = intr_blob.GetDims()[0];
        int MB_to_process = node->batchToProcess();
        // TODO: Should we support InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT???
        if (config.batchLimit)
            MB_to_process = std::min<int>(config.batchLimit, MB_to_process);
        size_t size_to_copy = intr_blob.GetSize() * MB_to_process / MB;

        cpu_memcpy_s(ext_blob_ptr, ext_blob->byteSize(), intr_blob_ptr, size_to_copy);
    }
}

void MKLDNNGraph::Infer(int batch) {
    if (!IsReady()) {
        THROW_IE_EXCEPTION << "Wrong state. Topology is not ready.";
    }

    mkldnn::stream stream = mkldnn::stream(stream::kind::eager);
    for (int i = 0; i < graphNodes.size(); i++) {
        PERF(graphNodes[i]);

        if (batch > 0)
            graphNodes[i]->setDynamicBatchLim(batch);

        ENABLE_DUMP(do_before(DUMP_DIR, graphNodes[i]));

        if (!graphNodes[i]->isConstant()) {
            OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, graphNodes[i]->profilingTask);
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

    if (!config.dumpToDot.empty()) dumpToDotFile(config.dumpToDot + "_perf.dot");
}

void MKLDNNGraph::setConfig(const Config &cfg) {
    config = cfg;
}

void MKLDNNGraph::setProperty(const std::map<std::string, std::string>& properties) {
    config.readProperties(properties);
}

Config MKLDNNGraph::getProperty() {
    return config;
}

void MKLDNNGraph::getInputBlobs(InferenceEngine::BlobMap &resp) {
#if defined (COMPILED_CPU_MKLDNN_INPUT_NODE)
    for (auto &it : inputNodes) {
        MKLDNNInputNode* node = dynamic_cast<MKLDNNInputNode*>(it.second.get());
        if (!node || node->isConstant())
            continue;
        resp[it.first] = node->getChildEdgeAt(0)->getBlob();
    }
#endif
}

void MKLDNNGraph::getOutputBlobs(InferenceEngine::BlobMap &resp) {
    for (auto &it : outputNodes) {
        std::string name = it->getName().substr(4);
        resp[name] = it->getParentEdgeAt(0)->getBlob();
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

void MKLDNNGraph::dumpToDotFile(std::string file) const {
    std::ofstream dot;
    dot.open(file);
    if (!dot.is_open()) THROW_IE_EXCEPTION << "CPU Plugin cannot create dot file " << file << ".";

    dump_graph_as_dot(*this, dot);
    dot.close();
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

InferenceEngine::ICNNNetwork::Ptr MKLDNNGraph::dump() const {
    return dump_graph_as_ie_ngraph_net(*this);
}
