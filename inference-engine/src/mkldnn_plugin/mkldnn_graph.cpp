// Copyright (C) 2018-2019 Intel Corporation
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
#include <details/ie_cnn_network_tools.h>

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
    if (IsReady())
        ForgetGraphData();

    Replicate(network, extMgr);
    InitGraph();
    status = Ready;
}

void MKLDNNGraph::Replicate(const ICNNNetwork &network, const MKLDNNExtensionManager::Ptr& extMgr) {
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

    std::unordered_map<CNNLayerPtr, MKLDNNNodePtr> layer2node;

    auto _parent_port = [] (const DataPtr &data) -> int {
        auto parent = data->creatorLayer.lock();
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
            _layer.reset(new CNNLayer({layer->name + "/id=" + memoryId, "MemoryInput", layer->precision}));
            _layer->params = layer->params;
            _layer->outData = layer->outData;
        }

        const MKLDNNNodePtr node(MKLDNNNode::CreateNode(_layer, getEngine(), extMgr));
        graphNodes.push_back(node);
        layer2node[layer] = node;

        for (int port = 0; port < layer->insData.size(); port++) {
            auto data = layer->insData[port].lock();
            auto parent_layer = data->creatorLayer.lock();
            if (!parent_layer) continue;  // no parent means that it is input data node (or memory/const layer)

            auto parent_node = layer2node[parent_layer];

            MKLDNNEdgePtr edge(new MKLDNNEdge(parent_node, node, _parent_port(data), port));
            node->addEdge(edge);
            graphEdges.push_back(edge);
        }
    }

    std::map<std::string, DataPtr> outputs;
    network.getOutputsInfo(outputs);

    for (const auto &output : outputs) {
        const auto data = output.second;

        auto parent_layer = data->creatorLayer.lock();
        auto parent_node = layer2node[parent_layer];

        CNNLayerPtr layer(new CNNLayer({"out_" + output.first, "Output", data->precision}));
        layer->insData.push_back(data);

        const MKLDNNNodePtr node(MKLDNNNode::CreateNode(layer, getEngine(), extMgr));

        MKLDNNEdgePtr edge(new MKLDNNEdge(parent_node, node, _parent_port(data), 0));
        node->addEdge(edge);
        graphEdges.push_back(edge);

        graphNodes.push_back(node);
        outputNodes.push_back(node);
        layer2node[layer] = node;
    }

    // Replicate input nodes
    for (const auto& input : inputs) {
        auto inputLayer = input.second->getInputData()->getCreatorLayer().lock();
        inputNodes[input.first] = layer2node[inputLayer];

        // Loading mean images
        MKLDNNDims outDims(inputNodes[input.first]->getChildEdgeAt(0)->getDims());
        if (inputs.find(input.first) != inputs.end()) {
            InputInfo::Ptr ii = inputs[input.first];
            if (ii && ii->getPreProcess().getNumberOfChannels()) {
                _meanImages[input.first].Load(outDims, ii);
            }
        }
    }
}

void MKLDNNGraph::InitGraph() {
    SortTopologically();
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

    // Do it before cleanup. Because it will lose original layers information
    for (auto &graphNode : graphNodes) {
        auto nodeType = graphNode->getType();
        if (nodeType == Reorder || nodeType == Output) continue;

        graphNode->addOriginalLayer(graphNode->getCnnLayer());
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

    for (auto &graphNode : graphNodes) {
        graphNode->cleanup();
    }

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
    for (auto i = 0; i < numberOfEdges; i++) {
        if (graphEdges[i]->needReorder()) {
            auto &edge = graphEdges[i];
            std::string layerName = edge->getParent()->getName() + "_" +
                                    reorderArgs(edge->getInputDesc(), edge->getOutputDesc()) + "_" +
                                    edge->getChild()->getName();
            CNNLayerPtr layer(new CNNLayer({layerName,
                                            "Reorder",
                                            edge->getInputDesc().getPrecision()}));
            MKLDNNNodePtr newReorder(new MKLDNNReorderNode(layer, getEngine()));
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

            // WA. MemoryOutput will keep data in that edge
            // So need to make it immortal..
            isConst |= edge->getParent()->getType() == MemoryInput;
        }

        if (isInput  | isConst) box.start = 0;
        if (isOutput | isConst) box.finish = -1;

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
    std::replace(nodeName.begin(), nodeName.end(), ':', '_');

    auto num_ports = node->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size();
    for (size_t i = 0; i < num_ports; i++) {
        auto prEdge = node->getParentEdgeAt(i);
        auto pr = prEdge->getParent();

        auto dump_file = dir + "/#" + exec_order + "_" +  nodeName + "_in" + std::to_string(i) + ".ieb";
        TensorDesc desc = prEdge->getDesc();
        if (desc.getPrecision() == Precision::BIN)
            return;
        Blob::Ptr blob = make_blob_with_precision(desc, prEdge->getMemoryPtr()->GetData());

        BlobDumper dumper(blob);
        if (pr->ext_scales) dumper.withScales(pr->ext_scales);
#ifdef DUMP_AS_TEXT
        dumper.dumpAsTxt(dump_file);
#else
        dumper.dump(dump_file);
#endif
    }
}

void MKLDNNGraph::do_after(const std::string &dir, const MKLDNNNodePtr &node) {
    auto exec_order = std::to_string(node->execIndex);
    auto nodeName = node->name;
    std::replace(nodeName.begin(), nodeName.end(), '\\', '_');
    std::replace(nodeName.begin(), nodeName.end(), '/', '_');
    std::replace(nodeName.begin(), nodeName.end(), ' ', '_');
    std::replace(nodeName.begin(), nodeName.end(), ':', '_');

    auto num_ports = node->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size();
    for (size_t i = 0; i < num_ports; i++) {
        auto childEdge = node->getChildEdgeAt(i);

        auto dump_file = dir + "/#" + exec_order + "_" +  nodeName + "_out" + std::to_string(i) + ".ieb";
        TensorDesc desc = childEdge->getDesc();
        if (desc.getPrecision() == Precision::BIN)
            return;
        Blob::Ptr blob = make_blob_with_precision(desc, childEdge->getMemoryPtr()->GetData());

        BlobDumper dumper(blob);
        if (node->ext_scales) dumper.withScales(node->ext_scales);

#ifdef DUMP_AS_TEXT
        dumper.dumpAsTxt(dump_file);
#else
        dumper.dump(dump_file);
#endif
    }
}

InferenceEngine::ICNNNetwork::Ptr MKLDNNGraph::dump() const {
    return dump_graph_as_ie_net(*this);
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
    // we are cloning network if we have statistics and we can transform network.
    auto clonedNetwork = cloneNet(network);

    if (s == StatusCode::OK && pstats && !pstats->isEmpty()) {
        CNNNetworkInt8Normalizer cnnorm;
        cnnorm.NormalizeNetwork(*clonedNetwork, *pstats);
    }

    bool ti_proc_ok = !NetPass::CombineRNNSeq(*clonedNetwork) ? NetPass::UnrollTI(*clonedNetwork) : true;
    ti_proc_ok &= NetPass::UnrollRNN_if(*clonedNetwork, [] (const RNNCellBase &rnn) -> bool {
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


    if (cfg.batchLimit > 1) {
        // check topology for applicability
        if (!CanProcessDynBatch(*clonedNetwork)) {
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
            _graph->CreateGraph(*clonedNetwork, extensionManager);
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

void MKLDNNExecNetwork::GetExecGraphInfo(InferenceEngine::ICNNNetwork::Ptr &graphPtr) {
    graphPtr = graphs[0]->dump();
}
