// Copyright (C) 2018 Intel Corporation
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
#include "details/caseless.hpp"

#include "mkldnn_graph.h"
#include "mkldnn_graph_optimizer.h"
#include <debug.h>
#include <nodes/mkldnn_input_node.h>
#include <nodes/mkldnn_reorder_node.h>
#include <nodes/mkldnn_depthwise_node.h>
#include <nodes/mkldnn_conv_node.h>

#include "mkldnn_extension_utils.h"
#include "mkldnn_extension_mngr.h"
#include "mkldnn/omp_manager.h"
#include <graph_tools.hpp>
#include <cpp_interfaces/ie_executor_manager.hpp>
#include "ie_algorithm.hpp"
#include "memory_solver.hpp"
#include "mkldnn_infer_request.h"
#include "mkldnn_async_infer_request.h"
#include <blob_factory.hpp>
#include <ie_util_internal.hpp>
#include <net_pass.h>

#include <mkldnn_graph_dumper.h>

#include <data_stats.h>
#include "cnn_network_int8_normalizer.hpp"
#include "ie_memcpy.h"

#define XBYAK_NO_OP_NAMES
#define XBYAK_UNDEF_JNL
#include "../../thirdparty/mkl-dnn/src/cpu/xbyak/xbyak_util.h"

#include "cnn_network_stats_impl.hpp"

#include "utils/blob_dump.h"

/*****************************************************
 * Dump capability
 * Specify path to dump folder in BLOB_DUMP_PATH
 *****************************************************/
// #define BLOB_DUMP_PATH "dump"

#ifdef BLOB_DUMP_PATH
#   define DUMP_DIR        BLOB_DUMP_PATH
#   define ENABLE_DUMP(_x) { _x ;}
#else
#   define DUMP_DIR ""
#   define ENABLE_DUMP(_x)
#endif

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace MKLDNNPlugin::cpu;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

void MKLDNNGraph::CreateGraph(const ICNNNetwork &network, const MKLDNNExtensionManager::Ptr& extMgr) {
    if (IsReady()) {
        ForgetGraphData();
    }

    // go over the inputs and create input primitives
    InputsDataMap inputs;
    network.getInputsInfo(inputs);
    if (inputs.empty()) {
        THROW_IE_EXCEPTION << "MKLDNNGraph::CreateGraph: No inputs for the topology";
    }

    // The input layer precision has to be equal to the InputData precision
    for (const auto& input : inputs) {
        auto inputLayer = input.second->getInputData()->getCreatorLayer().lock();
        if (inputLayer) inputLayer->precision = inputLayer->outData[0]->precision;
    }

    for (const auto& input : inputs) {
        auto inputLayer = input.second->getInputData()->getCreatorLayer().lock();
        if (!inputLayer) {
            // For v1 parser
            inputLayer.reset(new CNNLayer({input.second->getInputData()->getName(),
                                           "Input",
                                           input.second->getInputData()->getPrecision()}));

            inputLayer->outData.push_back(input.second->getInputData());
        }

        const MKLDNNNodePtr inputNode = MKLDNNNodePtr(MKLDNNNode::CreateNode(inputLayer, getEngine(), extMgr));

        graphNodes.push_back(inputNode);
        inputNodes[input.first] = inputNode;
        std::vector<ParsedLayer> queueLayers;

        for (const auto &layer : input.second->getInputData()->getInputTo()) {
            queueLayers.push_back({inputNode, layer.second, 0});
        }

        while (!queueLayers.empty()) {
            ParseNode(queueLayers[0].cnnLayer, queueLayers[0].parent, extMgr, queueLayers[0].outIdx, queueLayers);
            queueLayers.erase(queueLayers.begin());
        }

        // Loading mean images
        MKLDNNDims outDims(inputNode->getChildEdgeAt(0)->getDims());
        if (inputs.find(input.first) != inputs.end()) {
            InputInfo::Ptr ii = inputs[input.first];
            if (ii && ii->getPreProcess().getNumberOfChannels()) {
                _meanImages[input.first].Load(outDims, ii);
            }
        }
    }

    auto allInputs = CNNNetGetAllInputLayers(network);
    for (const auto& input : allInputs) {
        auto isRealInput = std::find_if(std::begin(inputs), std::end(inputs), [&](InputsDataMap::value_type& inputInfo){
            return inputInfo.second->getInputData()->getName() == input->name;
        });
        if (isRealInput != std::end(inputs)) {
            continue;
        }

        MKLDNNNodePtr inputNode;
        CaselessEq<std::string> eq;

        if (eq(input->type, "Memory")) {
            auto memoryId = input->GetParamAsString("id");
            CNNLayerPtr layer(new CNNLayer({input->name + "/id=" + memoryId, "MemoryInput", input->precision}));
            layer->params = input->params;
            layer->outData = input->outData;

            inputNode = MKLDNNNodePtr(MKLDNNNode::CreateNode(layer, getEngine(), extMgr));
        } else if (eq(input->type, "Const")) {
            inputNode = MKLDNNNodePtr(MKLDNNNode::CreateNode(input, getEngine(), extMgr));
        }
        graphNodes.push_back(inputNode);

        std::vector<ParsedLayer> queueLayers;
        size_t count_out = 0;
        for (auto &&outData : input->outData) {
            for (auto &&layer : outData->getInputTo()) {
                queueLayers.push_back({inputNode, layer.second, count_out});
            }
            count_out++;
        }

        while (!queueLayers.empty()) {
            ParseNode(queueLayers[0].cnnLayer, queueLayers[0].parent, extMgr, queueLayers[0].outIdx, queueLayers);
            queueLayers.erase(queueLayers.begin());
        }
    }

    std::map<std::string, DataPtr> output;
    network.getOutputsInfo(output);

    for (auto it = output.begin(); it != output.end(); ++it) {
        const DataPtr& outputDataPtr = it->second;

        MKLDNNNodePtr node = FindNodeWithName(outputDataPtr->getCreatorLayer().lock()->name);
        if (!node)
            THROW_IE_EXCEPTION << "Cannot find output layer " << outputDataPtr->getCreatorLayer().lock()->name;

        const std::string name = "out_" + it->first;

        CNNLayerPtr layer(new CNNLayer({name, "Output", outputDataPtr->getCreatorLayer().lock()->outData[0]->getPrecision()}));
        layer->insData.push_back(outputDataPtr);
        MKLDNNNodePtr outputLayer(new MKLDNNInputNode(layer, getEngine()));
        MKLDNNEdgePtr edgePtr(new MKLDNNEdge(node, outputLayer));
        graphEdges.push_back(edgePtr);

        const std::vector<MKLDNNEdgeWeakPtr>& childEdges = node->getChildEdges();
        size_t insertBeforeChildEdgeIndex = childEdges.size();
        if (!childEdges.empty()) {
            bool outputDataIndexWasFound = false;
            size_t outputDataIndex = 0;
            for (size_t i = 0; i < node->getCnnLayer()->outData.size(); ++i) {
                const DataPtr& otherOutputDataPtr = node->getCnnLayer()->outData[i];
                if (otherOutputDataPtr->name == it->first) {
                    outputDataIndexWasFound = true;
                    outputDataIndex = i;
                }
            }
            IE_ASSERT(outputDataIndexWasFound) << "Node " << node->getName() << " doesn't have output data '" << it->first << "'";

            std::unordered_map<Data*, size_t> nodeOutputDataIndexByData;
            const CNNLayerPtr& nodeLayer = node->getCnnLayer();
            for (size_t dataIndex = 0; dataIndex < nodeLayer->outData.size(); ++dataIndex) {
                nodeOutputDataIndexByData.emplace(nodeLayer->outData[dataIndex].get(), dataIndex);
            }

            auto getOutputDataIndex = [&](const MKLDNNEdgePtr& childEdge) -> size_t {
                const InferenceEngine::CNNLayerPtr& childNodeLayer = childEdge->getChild()->getCnnLayer();
                for (const DataWeakPtr& childNodeInsertWeakData : childNodeLayer->insData) {
                    const DataPtr childNodeInsertData = childNodeInsertWeakData.lock();
                    if (!childNodeInsertData) {
                        continue;
                    }

                    const auto indexIt = nodeOutputDataIndexByData.find(childNodeInsertData.get());
                    if (indexIt != nodeOutputDataIndexByData.end()) {
                        return indexIt->second;
                    }
                }

                IE_ASSERT(false) << "Node has child edge without insert data";
            };

            for (size_t childEdgeIndex = 0; childEdgeIndex < childEdges.size(); ++childEdgeIndex) {
                const MKLDNNEdgePtr childEdge = childEdges[childEdgeIndex].lock();
                if (!childEdge) {
                    continue;
                }

                const size_t edgeOutputDataIndex = getOutputDataIndex(childEdge);
                if (outputDataIndex < edgeOutputDataIndex) {
                    insertBeforeChildEdgeIndex = childEdgeIndex;
                    break;
                }
            }
        }

        if (insertBeforeChildEdgeIndex < childEdges.size()) {
            outputLayer->addEdge(edgePtr, 0, insertBeforeChildEdgeIndex, true);
        } else {
            outputLayer->addEdge(edgePtr, 0, node->getChildEdges().size());
        }

        graphNodes.push_back(outputLayer);
        outputNodes.push_back(outputLayer);
    }

    MKLDNNGraphOptimizer optimizer;
    optimizer.ApplyCommonGraphOptimizations(*this);
    SortTopologically();

    InitNodes();

    for (auto &node : graphNodes) {
        node->initOptimalPrimitiveDescriptor();
    }
    InitEdges();

    optimizer.ApplyImplSpecificGraphOptimizations(*this);

    SortTopologically();

    Allocate();

    CreatePrimitives();

    // Will do it before cleanup. Because it will lose original layers information
    if (!config.dumpToDot.empty()) dumpToDotFile(config.dumpToDot + "_init.dot");

    for (auto &graphNode : graphNodes) {
        graphNode->cleanup();
    }

    for (auto &graphNode : graphNodes) {
#ifndef NDEBUG
        std::cout << "name: " << graphNode->getName() << " [ ";
#endif
        if (graphNode->parentEdges.size() > 0) {
            auto prnt = graphNode->parentEdges[0].lock();
#ifndef NDEBUG
            std::cout << "in: " << prnt->getOutputDesc().getPrecision().name() << "/l="
                    << prnt->getOutputDesc().getLayout()
                    << "; ";
#endif
        }
        if (graphNode->childEdges.size() > 0) {
            auto chld = graphNode->childEdges[0].lock();
#ifndef NDEBUG
            std::cout << "out: " << chld->getInputDesc().getPrecision().name() << "/l="
                    << chld->getInputDesc().getLayout();
#endif
        }
#ifndef NDEBUG
        std::cout << " ]"  << std::endl;
#endif
    }


    mkldnn::stream stream = mkldnn::stream(stream::kind::eager);
    for (auto &graphNode : graphNodes) {
        if (!graphNode->isConstant())
            continue;
        graphNode->execute(stream);
    }

    status = Ready;
}

void MKLDNNGraph::ParseNode(const CNNLayerPtr& cnnLayer, MKLDNNNodePtr& parent,
                            const MKLDNNExtensionManager::Ptr& extMgr, size_t outIdx,
                            std::vector<ParsedLayer>& queuelayers) {
    if (cnnLayer->precision != Precision::FP32 &&
        cnnLayer->precision != Precision::I8 &&
        cnnLayer->precision != Precision::U8) {
        THROW_IE_EXCEPTION << "The plugin does not support " << cnnLayer->precision;
    }

    MKLDNNNodePtr node = FindNodeWithName(cnnLayer->name);
    bool exists = false;
    if (node) {
        exists = true;
    } else {
        node.reset(MKLDNNNode::CreateNode(cnnLayer, getEngine(), extMgr));
    }

    if (parent) {
        MKLDNNEdgePtr edgePtr;
        size_t shift = 0;
        if (outIdx >= parent->getChildEdges().size() || !parent->getChildEdges()[outIdx].lock()) {
            edgePtr.reset(new MKLDNNEdge(parent, node));
            graphEdges.push_back(edgePtr);
        } else {
            edgePtr = parent->getChildEdgeAt(outIdx);
            if (edgePtr->getChild() != node) {
                edgePtr.reset(new MKLDNNEdge(parent, node));
                graphEdges.push_back(edgePtr);
                shift = parent->getChildEdges().size();
            }
        }


        size_t pIndex = node->getParentEdges().size();
        if (parent->getCnnLayer() != nullptr) {
            for (size_t idx = 0; idx < cnnLayer->insData.size(); idx++) {
                auto cnnLayerIN = cnnLayer->insData[idx].lock();
                if (cnnLayerIN &&
                    parent->getCnnLayer()->outData.size() > outIdx &&
                    cnnLayerIN.get() == parent->getCnnLayer()->outData[outIdx].get()) {
                    pIndex = idx;
                    break;
                }
            }
            node->addEdge(edgePtr, pIndex, outIdx + shift);
            if (cnnLayer->insData.size() > 1) {
                for (size_t idx = 1; idx < cnnLayer->insData.size(); idx++) {
                    if (cnnLayer->insData[idx].lock() == cnnLayer->insData[idx - 1].lock()) {
                        node->addEdge(edgePtr, pIndex + idx, outIdx + shift + idx);
                    }
                }
            }
        } else {
            for (size_t idx = 0; idx < cnnLayer->insData.size(); idx++) {
                if (cnnLayer->insData[idx].lock()->getName() == parent->getName()) {
                    pIndex = static_cast<int>(idx);
                    break;
                }
            }
            node->addEdge(edgePtr, pIndex, outIdx + shift);
        }
    }

    if (exists)
        return;

    if (cnnLayer->blobs.find("ext-scale") != cnnLayer->blobs.end())
        node->ext_scales = cnnLayer->blobs["ext-scale"];

    graphNodes.push_back(node);

    size_t count_out = 0;
    std::vector<ParsedLayer> remaining;
    for (const auto &layer : cnnLayer->outData) {
        bool first = true;
        for (const auto &data : layer->getInputTo()) {
            if (first) {
                queuelayers.push_back({node, data.second, count_out});
                first = false;
            } else {
                // TODO: Just to hide bug with port ordering.
                //       At first step we visit only first connection
                //       at port. As second we will visit all remaining.
                //
                // Not first connection to the port are stored here
                remaining.push_back({node, data.second, count_out});
            }
        }
        count_out++;
    }
    queuelayers.insert(queuelayers.end(), remaining.begin(), remaining.end());
}

void MKLDNNGraph::InitNodes() {
    for (auto &node : graphNodes) {
        if (node->getType() == Input && _meanImages.find(node->getName()) != _meanImages.end()) {
            auto *inputNode = dynamic_cast<MKLDNNInputNode *>(node.get());
            if (inputNode)
                inputNode->withMeanImage();
        }
        node->getSupportedDescriptors();

        node->initSupportedPrimitiveDescriptors();
    }

    for (auto &node : graphNodes) {
        node->selectOptimalPrimitiveDescriptor();
    }
}

void MKLDNNGraph::InitEdges() {
    auto reorderArgs = [](InferenceEngine::TensorDesc parentDesc, InferenceEngine::TensorDesc childDesc) {
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
    for (auto i = 0; i < numberOfEdges; i++) {
        if (graphEdges[i]->needReorder()) {
            std::string layerName = graphEdges[i]->getParent()->getName() + "_" +
                    reorderArgs(graphEdges[i]->getInputDesc(), graphEdges[i]->getOutputDesc()) + "_" +
                    graphEdges[i]->getChild()->getName();
            CNNLayerPtr layer(new CNNLayer({layerName,
                                            "Reorder",
                                            graphEdges[i]->getInputDesc().getPrecision()}));
            MKLDNNNodePtr newReorder(new MKLDNNReorderNode(layer, getEngine()));
            auto *reorderPtr = dynamic_cast<MKLDNNReorderNode *>(newReorder.get());
            if (reorderPtr) {
                reorderPtr->setDescs(graphEdges[i]->getInputDesc(), graphEdges[i]->getOutputDesc());
            }
            MKLDNNEdgePtr beforeNode(new MKLDNNEdge(graphEdges[i]->getParent(), newReorder));
            beforeNode->setDims(graphEdges[i]->getDims());
            MKLDNNEdgePtr afterNode(new MKLDNNEdge(newReorder, graphEdges[i]->getChild()));
            afterNode->setDims(graphEdges[i]->getDims());

            auto oIndexes = graphEdges[i]->getAllOutputNums();
            auto iIndexes = graphEdges[i]->getAllInputNums();
            if (iIndexes[0] < 0 || oIndexes[0] < 0)
                THROW_IE_EXCEPTION << "Cannot create reorder for nodes: "
                                   << graphEdges[i]->getParent()->getName() << " and "
                                   << graphEdges[i]->getChild()->getName() << ".";

            // Add edge for beforeNode
            beforeNode->getChild()->parentEdges.push_back(beforeNode);
            for (auto iIndex : iIndexes) graphEdges[i]->getParent()->childEdges[iIndex] = beforeNode;

            // Add edge for afterNode
            afterNode->getParent()->childEdges.push_back(afterNode);
            for (auto oIndex : oIndexes) graphEdges[i]->getChild()->parentEdges[oIndex] = afterNode;

            newReorder->getSupportedDescriptors();
            newReorder->initSupportedPrimitiveDescriptors();
            newReorder->selectOptimalPrimitiveDescriptor();

            beforeNode->getDesc();
            graphEdges.push_back(beforeNode);
            afterNode->getDesc();
            graphEdges.push_back(afterNode);

            graphNodes.push_back(newReorder);
            graphEdges.erase(graphEdges.begin() + i);
            i--;
            numberOfEdges--;
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
                        claster.push_back(edge);
                        found = true;
                        break;
                    }
                }
            }
            if (!found) edge_clasters.push_back({par, edge});

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
            if (!found) edge_clasters.push_back({edge});
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

    const int alignment = 16;  // 64 bytes or 16 floats

    std::vector<MemorySolver::Box> boxes(edge_clasters.size());
    for (int i = 0; i < edge_clasters.size(); i++) {
        MemorySolver::Box &box = boxes[i];
        box = { std::numeric_limits<int>::max(), 0, 0, i };
        for (auto &edge : edge_clasters[i]) {
            int e_start = edge->getParent()->execIndex;
            int e_finish = edge->getChild()->execIndex;

            const BlockingDesc block_desk = edge->getDesc().getBlockingDesc();

            int e_size = block_desk.getOffsetPadding() + 1;  // size in elements (from begin of data to last element)
            for (int j = 0; j < block_desk.getBlockDims().size(); j++)
                e_size += (block_desk.getBlockDims()[j] - 1) * block_desk.getStrides()[j];

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

            // WA. MemoryOutput will keep data in that edge
            // So need to make it immortal..
            isConst |= edge->getParent()->getType() == MemoryInput;
        }

        if (isInput  | isConst) box.start = 0;
        if (isOutput | isConst) box.finish = -1;

        box.size = div_up(box.size, alignment);
    }

    MemorySolver memSolver(boxes);
    size_t total_size = memSolver.solve() * alignment;

    memWorkspace.reset(new MKLDNNMemory(eng));
    memWorkspace->Create(MKLDNNMemoryDesc(TensorDesc(Precision::FP32, {total_size}, Layout::C)));
    float* workspace_ptr = static_cast<float*>(memWorkspace->GetData());

    for (int i = 0; i < edge_clasters.size(); i++) {
        int count = 0;
        for (auto &edge : edge_clasters[i]) {
            if (edge->getStatus() == MKLDNNEdge::Status::NeedAllocation) {
                int offset = memSolver.getOffset(i);
                // !! Fallback to individual memory allocation !!
                // if you like to check infer without reuse just call this function without arguments.
                edge->allocate(workspace_ptr + offset * alignment);  // alignment in float
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
                _meanImages[name].Subtract(outDims, reinterpret_cast<float *>(inter_data_ptr));
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
            SizeVector dims = node->getParentEdgeAt(0)->getDims().ToSizeVector();
            std::reverse(dims.begin(), dims.end());  // Blobs dims are in reverse order (legacy of OpenVX :-( )
            ext_blob->Resize(dims);
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

        ie_memcpy(ext_blob_ptr, ext_blob->byteSize(), intr_blob_ptr, size_to_copy);
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
            IE_PROFILING_AUTO_SCOPE_TASK(graphNodes[i]->profilingTask)
            graphNodes[i]->execute(stream);
        }

        ENABLE_DUMP(do_after(DUMP_DIR, graphNodes[i]));
    }
}

MKLDNNNodePtr MKLDNNGraph::FindNodeWithName(const std::string& name) const {
    if (inputNodes.empty()) {
        return std::shared_ptr<MKLDNNNode>();
    }

    const auto children = graphNodes;
    const auto node = std::find_if(children.begin(), children.end(),
                             [&name](MKLDNNNodePtr const& item) {
                                 return item->getName() == name;
                             });

    return (node == children.end() ? std::shared_ptr<MKLDNNNode>() : *node);
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
}

void MKLDNNGraph::GetPerfData(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const {
    std::function<void(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &, const MKLDNNNodePtr&)>
            getPerfMapFor = [&](std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap, const MKLDNNNodePtr& node) {
        InferenceEngine::InferenceEngineProfileInfo &pc = perfMap[node->getName()];
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
    for (auto &it : inputNodes) {
        MKLDNNInputNode* node = dynamic_cast<MKLDNNInputNode*>(it.second.get());
        if (!node || node->isConstant())
            continue;
        resp[it.first] = node->getChildEdgeAt(0)->getBlob();
    }
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
    for (size_t i = 0; i < node->parentEdges.size(); i++) {
        if (!node->parentEdges[i].lock())
            continue;
        auto parent = node->parentEdges[i].lock()->getParent();
        if (!parent)
            continue;

        for (size_t j = 0; j < node->childEdges.size(); j++) {
            if (!node->childEdges[j].lock())
                continue;
            auto child = node->childEdges[j].lock()->getChild();
            if (!child)
                continue;

            MKLDNNEdgePtr remEdge = node->parentEdges[i].lock();
            int inNum = 0;
            if (remEdge) {
                inNum = remEdge->getInputNum();
                node->removeEdge(remEdge);
                removeEdge(*this, remEdge);
            }
            inNum += j;
            remEdge = node->childEdges[j].lock();
            int outNum = 0;
            if (remEdge) {
                outNum = remEdge->getOutputNum();
                node->removeEdge(remEdge);
                removeEdge(*this, remEdge);
            }
            MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, child));
            this->GetEdges().push_back(newEdge);
            parent->addEdge(newEdge, outNum, inNum);
        }
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
    std::replace(nodeName.begin(), nodeName.end(), '/', '_');

    auto num_ports = node->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size();
    for (size_t i = 0; i < num_ports; i++) {
        auto prEdge = node->getParentEdgeAt(i);
        auto pr = prEdge->getParent();

        auto dump_file = dir + "/#" + exec_order + "_" +  nodeName + "_in" + std::to_string(i) + ".ieb";
        TensorDesc desc = prEdge->getDesc();
        Blob::Ptr blob = make_blob_with_precision(desc, prEdge->getMemoryPtr()->GetData());

        BlobDumper dumper(blob);
        if (pr->ext_scales) dumper.withScales(pr->ext_scales);
        dumper.dump(dump_file);
    }
}

void MKLDNNGraph::do_after(const std::string &dir, const MKLDNNNodePtr &node) {
    auto exec_order = std::to_string(node->execIndex);
    auto nodeName = node->name;
    std::replace(nodeName.begin(), nodeName.end(), '/', '_');

    auto num_ports = node->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size();
    for (size_t i = 0; i < num_ports; i++) {
        auto childEdge = node->getChildEdgeAt(i);

        auto dump_file = dir + "/#" + exec_order + "_" +  nodeName + "_out" + std::to_string(i) + ".ieb";
        TensorDesc desc = childEdge->getDesc();
        Blob::Ptr blob = make_blob_with_precision(desc, childEdge->getMemoryPtr()->GetData());

        BlobDumper dumper(blob);
        if (node->ext_scales) dumper.withScales(node->ext_scales);

        dumper.dump(dump_file);
    }
}

bool MKLDNNExecNetwork::CanProcessDynBatch(const InferenceEngine::ICNNNetwork &network) const {
    InputsDataMap inputs;
    network.getInputsInfo(inputs);

    CNNLayerSet inputLayers;
    std::unordered_set<CNNLayer *> allLayers;

    if (inputs.empty())
        return false;

    auto & secondLayers = inputs.begin()->second->getInputData()->getInputTo();
    if (secondLayers.empty())
        return false;

    bool check_result = true;
    details::UnorderedDFS(allLayers, secondLayers.begin()->second, [&](CNNLayerPtr layer) {
        auto type = TypeFromName(layer->type);
        // This is WA for Tile layer
        auto tileLayer = dynamic_cast<TileLayer *>(layer.get());
        if (tileLayer && tileLayer->axis)
            return;

        if (type != Input &&
            type != Output &&
            type != Convolution &&
            type != Deconvolution &&
            type != Activation &&
            type != Depthwise &&
            type != Lrn &&
            type != Pooling &&
            type != FullyConnected &&
            type != Gemm &&
            type != SoftMax &&
            type != Split &&
            type != Concatenation &&
            type != Power &&
            type != Eltwise &&
            type != Crop &&
            type != BatchNormalization &&
            type != Copy) {
            check_result = false;
        }
    }, false);

    return check_result;
}

InferenceEngine::InferRequestInternal::Ptr
MKLDNNExecNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                          InferenceEngine::OutputsDataMap networkOutputs) {
    if (graphs.size() > 1)  // streams uses special requests that are not connected to graphs
        return std::make_shared<MKLDNNGraphlessInferRequest>(networkInputs, networkOutputs);
    else
        return std::make_shared<MKLDNNInferRequest>(networkInputs, networkOutputs);
}

MKLDNNExecNetwork::MKLDNNExecNetwork(const InferenceEngine::ICNNNetwork &network,
                                     const Config &cfg,
                                     const MKLDNNExtensionManager::Ptr& extMgr) : extensionManager(extMgr) {
    ICNNNetworkStats* pstats = nullptr;
    StatusCode s = network.getStats(&pstats, nullptr);
    // we are cloning network if we have statistics and we can transform network
    // in other case we pass original network. Especially because LSTM networks
    // are not cloned properly
    details::CNNNetworkImplPtr clonedNetwork;
    if (s == StatusCode::OK && pstats && !pstats->isEmpty()) {
        CNNNetworkInt8Normalizer cnnorm;
        clonedNetwork = cloneNet(network);
        cnnorm.NormalizeNetwork(*clonedNetwork, *pstats);
    }
    bool ti_proc_ok = !NetPass::CombineLSTMSeq(network) ? NetPass::UnrollTI(network) : true;
    if (!ti_proc_ok)
        THROW_IE_EXCEPTION << "Plugin doesn't support Tensor Iterator in pure form. "
                              "None TI optimization pattern has been applied successfully";


    if (cfg.batchLimit > 1) {
        // check topology for applicability
        if (!CanProcessDynBatch(clonedNetwork ? *clonedNetwork : network)) {
            THROW_IE_EXCEPTION << "MKLDNNGraph::CreateGraph: such topology cannot be compiled for dynamic batch!";
        }
    }
    // check whether any (affinity-related) envs are set and if user requested thread binding
    const bool bPinningRequested = !check_env_variables() && cfg.useThreadBinding;
    // general #threads logic
    const int env_threads = parallel_get_env_threads();
    // for streams need all (logical) cores, while single-stream case just physical cores (better for servers), as usual
    const int hw_cores = cfg.throughputStreams > 1 ? parallel_get_max_threads() : getNumberOfCPUCores();
    const int threads = cfg.threadsNum ? cfg.threadsNum : (env_threads ? env_threads : hw_cores);
    const int threads_per_stream = std::max(1, threads/cfg.throughputStreams);

    // graph(s) initialization in taskExecutor threads (streams), in parallel (in case of streams)
    std::vector<Task::Ptr> tasks;

    for (int n = 0; n < cfg.throughputStreams; n++) {
        MKLDNNGraph::Ptr _graph = std::make_shared<MKLDNNGraph>();
        graphs.push_back(_graph);
        auto task = std::make_shared<InferenceEngine::Task>([=, &cfg, &network]() {
            _graph->CreateArena(threads_per_stream);

            if (bPinningRequested) {
                _graph->CreateObserver(n, threads_per_stream);
            }

            _graph->setConfig(cfg);
            _graph->CreateGraph(clonedNetwork ? *clonedNetwork : network, extensionManager);
            if (cfg.throughputStreams > 1)  // for streams, each worker thread has it's own graph
                MKLDNNPlugin::MultiWorkerTaskExecutor::ptrContext.ptrGraph = _graph;
        });
        tasks.push_back(task);
    }

    if (cfg.throughputStreams > 1) {
        // special executor with as many threads as requested #streams, each with it's own initialization task
        _taskExecutor = std::make_shared<MultiWorkerTaskExecutor>(tasks);
    } else {
        if (cfg.exclusiveAsyncRequests) {
            // special case when all InferRequests are muxed into a single queue
            ExecutorManager *executorManager = ExecutorManager::getInstance();
            _taskExecutor = executorManager->getExecutor(TargetDeviceInfo::name(TargetDevice::eCPU));
        }
        _taskExecutor->startTask(tasks[0]);
        Task::Status sts = tasks[0]->wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    }
    for (auto t : tasks)
        t->checkException();
}

void MKLDNNExecNetwork::setProperty(const std::map<std::string, std::string> &properties) {
    for (auto g : graphs)
        g->setProperty(properties);
}

void MKLDNNExecNetwork::CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) {
    auto syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    auto asyncRequestImpl = std::make_shared<MKLDNNAsyncInferRequest>(syncRequestImpl, _taskExecutor,
                                                                      _taskSynchronizer, _callbackExecutor);
    asyncRequest.reset(new InferRequestBase<MKLDNNAsyncInferRequest>(asyncRequestImpl),
                       [](IInferRequest *p) { p->Release(); });

    asyncRequestImpl->SetPointerToPublicInterface(asyncRequest);

    if (graphs.size() == 1) {  // single-stream (legacy/hetero) case - single graph for all requests
        auto mkldnnSyncRequest = dynamic_cast<MKLDNNInferRequest *>(syncRequestImpl.get());
        if (!mkldnnSyncRequest)
            THROW_IE_EXCEPTION << " Cannot get mkldnn sync request.";
        mkldnnSyncRequest->SetGraph(graphs[0]);
    }
}
