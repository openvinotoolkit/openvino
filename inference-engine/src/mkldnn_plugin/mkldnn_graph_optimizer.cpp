// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_graph_optimizer.h"

#include "mkldnn_extension_utils.h"
#include "nodes/mkldnn_reshape_node.h"
#include "nodes/mkldnn_activation_node.h"
#include "nodes/mkldnn_pooling_node.h"
#include "nodes/mkldnn_eltwise_node.h"
#include "nodes/mkldnn_depthwise_node.h"
#include "nodes/mkldnn_concat_node.h"
#include "nodes/mkldnn_reorder_node.h"
#include "nodes/mkldnn_conv_node.h"
#include "nodes/mkldnn_bin_conv_node.h"
#include "nodes/mkldnn_quantize_node.h"
#include "nodes/mkldnn_mvn_node.h"
#include "nodes/mkldnn_resample_node.h"

#include <blob_factory.hpp>
#include <ie_layers_internal.hpp>

// WA for xbyak.h
#ifdef _WIN32
# ifndef _WINSOCKAPI_
#  define _WINSOCKAPI_
# endif
# ifndef _WINSOCK2API_
#  define _WINSOCK2API_
#endif
#endif
#include <cpu_isa_traits.hpp>

#include <string>
#include <list>
#include <memory>
#include <set>
#include <algorithm>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNGraphOptimizer::MKLDNNGraphOptimizer() {}

void MKLDNNGraphOptimizer::ApplyCommonGraphOptimizations(MKLDNNGraph &graph) {
    MergeTwoEqualScaleShifts(graph);
    graph.RemoveDroppedNodes();

    MergeSigmoidAndMultiplyToSwish(graph);
    graph.RemoveDroppedNodes();

    MergeConversions(graph);
    graph.RemoveDroppedNodes();

    FuseBroadcastAndEltwise(graph);
    graph.RemoveDroppedNodes();

    FuseClampAndQuantize(graph);
    graph.RemoveDroppedNodes();

    FuseScaleShiftAndQuantize(graph);
    graph.RemoveDroppedNodes();

    MergeGroupConvolution(graph);
    graph.RemoveDroppedNodes();

    FuseConvolutionAndZeroPoints(graph);
    graph.RemoveDroppedNodes();

#if defined (COMPILED_CPU_MKLDNN_DEPTHWISE_NODE)
    FuseConvolutionAndDepthwise(graph);
    graph.RemoveDroppedNodes();
#endif

#if defined(COMPILED_CPU_MKLDNN_ACTIVATION_NODE)
    FuseConvolutionAndActivation(graph);
    graph.RemoveDroppedNodes();
#endif

#if defined (COMPILED_CPU_MKLDNN_DEPTHWISE_NODE)
    FuseConvolutionAndDepthwise(graph);
    graph.RemoveDroppedNodes();
#endif

    FuseConvolutionAndQuantize(graph);
    graph.RemoveDroppedNodes();

    graph.SortTopologically();
    graph.RemoveDroppedEdges();

#if defined (COMPILED_CPU_MKLDNN_DEPTHWISE_NODE)
    FuseConvolutionAndDepthwise(graph);
    graph.RemoveDroppedNodes();
#endif

    FusePoolingAndQuantize(graph);
    graph.RemoveDroppedNodes();

    graph.SortTopologically();
    graph.RemoveDroppedEdges();

    FuseConvolutionAndDWConvolution(graph);
    graph.RemoveDroppedNodes();

#if defined(COMPILED_CPU_MKLDNN_QUANTIZE_NODE)
    FuseBinaryConvolutionAndQuantize(graph);
    graph.RemoveDroppedNodes();
#endif

    FuseBatchNormWithScale(graph);
    graph.RemoveDroppedNodes();

    RemoveIdentityOperator(graph);
    graph.RemoveDroppedNodes();

#if defined(COMPILED_CPU_MKLDNN_ELTWISE_NODE)
    FuseConvolutionSumAndConvolutionSumActivation(graph);
    graph.RemoveDroppedNodes();
#endif

    FuseConvolutionAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseFullyConnectedAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseMVNAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseResampleAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseNormalizeAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseEltwiseAndSimple(graph);
    graph.RemoveDroppedNodes();

    graph.RemoveDroppedEdges();
}

void MKLDNNGraphOptimizer::ApplyImplSpecificGraphOptimizations(MKLDNNGraph &graph) {
    RemoveIOScaleShifts(graph);
    graph.RemoveDroppedNodes();

#if defined (COMPILED_CPU_MKLDNN_REORDER_NODE)
    DropDoubleReorders(graph);
    graph.RemoveDroppedNodes();

    DropConvertReorder(graph);
    graph.RemoveDroppedNodes();
#endif

    graph.RemoveDroppedEdges();
}

void MKLDNNGraphOptimizer::MergeConversions(MKLDNNGraph& graph) {
    for (auto node : graph.GetNodes()) {
        // Input with at least 2 Convertions
        if (!IsOneOf(node->getType(), { Input }) || node->getChildEdges().size() < 2 ||
            !IsOneOf(node->getChildEdgeAt(0)->getChild()->getType(), { Convert })) {
            continue;
        }
        auto& input = node;

        // Convertions of same the type with Concat as a child
        for (size_t i = 0; i < input->getChildEdges().size(); i++) {
            auto convInEdge = input->getChildEdgeAt(i);
            auto conv = convInEdge->getChild();
            auto convOutEdge = conv->getChildEdgeAt(i);
            auto convInDims = convInEdge->getDims();
            auto convOutDims = convOutEdge->getDims();
            Precision convOutPrecision = conv->getCnnLayer()->precision;

            for (size_t j = i + 1; j < input->getChildEdges().size();) {
                auto childEdge = input->getChildEdgeAt(j);
                auto child = childEdge->getChild();

                if (child->getCnnLayer()->precision != convOutPrecision ||
                    child->getChildEdgeAt(0)->getDims() != convOutDims ||
                    childEdge->getDims() != convInDims ||
                    child->getChildEdges().size() != 1) {
                    j++;
                    continue;
                }

                auto childChildEdge = child->getChildEdgeAt(0);
                auto childChild = childChildEdge->getChild();
                int idxChild = childChildEdge->getOutputNum();

                child->remove();
                graph.DropNode(child);

                MKLDNNEdgePtr newEdge(new MKLDNNEdge(conv, childChild, 0, idxChild));
                graph.GetEdges().push_back(newEdge);
                conv->addEdge(newEdge);
            }
        }
    }
}

void MKLDNNGraphOptimizer::FuseConvolutionAndZeroPoints(MKLDNNGraph &graph) {
    auto removeEdge = [](MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == edge) {
                edges.erase(it);
                return;
            }
        }
    };

    auto& graphNodes = graph.GetNodes();

    auto isSutableConvNode = [](MKLDNNNodePtr node) {
        if (node->getType() != Convolution)
            return false;

        if (node->getParentEdges().size() < 2)
            return false;

        auto* convLayer = dynamic_cast<ConvolutionLayer*>(node->getCnnLayer().get());
        if (convLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot get convolution layer " << node->getName();

        return true;
    };

    auto initializeInputZeroPoints = [](MKLDNNNodePtr node, MKLDNNNodePtr parent0) {
        auto* convNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
        if (convNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get convolution node " << node->getName();

        int IC = node->getParentEdgesAtPort(0)[0]->getDims()[1];
        int OC = node->getChildEdgesAtPort(0)[0]->getDims()[1];

        if (parent0->getType() == Eltwise) {
            auto * eltwiseLayer = dynamic_cast<EltwiseLayer*>(parent0->getCnnLayer().get());
            if (eltwiseLayer == nullptr)
                THROW_IE_EXCEPTION << "Cannot get eltwise layer " << node->getName();

            if (eltwiseLayer->_operation != EltwiseLayer::Sub)
                return false;

            if (parent0->getParentEdges().size() != 2)
                return false;

            if (parent0->getParentEdgesAtPort(1)[0]->getParent()->getCnnLayer()->type == "Const") {
                auto arg0 = parent0->getParentEdgesAtPort(1)[0]->getParent();
                if (arg0->getCnnLayer()->outData[0]->getPrecision() != Precision::U8)
                    return false;

                if (parent0->getParentEdgesAtPort(1)[0]->getDims()[1] != 1 &&
                    parent0->getParentEdgesAtPort(1)[0]->getDims()[1] != IC)
                    return false;

                auto arg1 = parent0->getParentEdgesAtPort(0)[0]->getParent();
                if (arg1->getCnnLayer()->outData[0]->getPrecision() != Precision::U8)
                    return false;

                auto zeroPointsBlob = dynamic_cast<TBlob<uint8_t>*>(arg0->getCnnLayer()->blobs["custom"].get());
                auto zeroPointsData = zeroPointsBlob->buffer().as<uint8_t*>();

                for (int j = 0; j < parent0->getParentEdgesAtPort(1)[0]->getDims()[1]; j++) {
                    convNode->inputZeroPoints.push_back(zeroPointsData[j]);
                }
            } else {
                return false;
            }
        } else {
            return false;
        }

        if (convNode->outputCompensation.empty()) {
            convNode->outputCompensation.resize(OC);
        }

        return true;
    };

    auto initializeWeightsZeroPoints = [](MKLDNNNodePtr node, MKLDNNNodePtr parent0) {
        auto* convNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
        if (convNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get convolution node " << node->getName();

        int OC = node->getChildEdgesAtPort(0)[0]->getDims()[1];

        if (parent0->getType() == Eltwise) {
            auto * eltwiseLayer = dynamic_cast<EltwiseLayer*>(parent0->getCnnLayer().get());
            if (eltwiseLayer == nullptr)
                THROW_IE_EXCEPTION << "Cannot get eltwise layer " << node->getName();

            if (eltwiseLayer->_operation != EltwiseLayer::Sub)
                return false;

            if (parent0->getParentEdges().size() != 2)
                return false;

            if (parent0->getParentEdgesAtPort(1)[0]->getParent()->getCnnLayer()->type == "Const") {
                auto arg0 = parent0->getParentEdgesAtPort(1)[0]->getParent();
                if (arg0->getCnnLayer()->outData[0]->getPrecision() != Precision::I8)
                    return false;

                if (parent0->getParentEdgesAtPort(1)[0]->getDims()[0] != 1 &&
                    parent0->getParentEdgesAtPort(1)[0]->getDims()[0] != OC)
                    return false;

                auto arg1 = parent0->getParentEdgesAtPort(0)[0]->getParent();
                if (arg1->getCnnLayer()->outData[0]->getPrecision() != Precision::I8)
                    return false;

                auto zeroPointsBlob = dynamic_cast<TBlob<int8_t>*>(arg0->getCnnLayer()->blobs["custom"].get());
                auto zeroPointsData = zeroPointsBlob->buffer().as<int8_t*>();

                for (int j = 0; j < parent0->getParentEdgesAtPort(1)[0]->getDims()[0]; j++) {
                    convNode->weightsZeroPoints.push_back(static_cast<float>(zeroPointsData[j]));
                }
            } else {
                return false;
            }
        } else {
            return false;
        }

        return true;
    };

    auto initializeOutputCompensation = [](MKLDNNNodePtr node) {
        auto* convNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
        if (convNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get convolution node " << node->getName();

        auto * convLayer = dynamic_cast<ConvolutionLayer*>(convNode->getCnnLayer().get());
        if (convLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot get eltwise layer " << node->getName();

        for (int i = 0; i < convLayer->insData.size(); i++)
            if (convLayer->insData[i].lock() == nullptr)
                THROW_IE_EXCEPTION << "Node '"<< node->getName() << "' has invalid input data with index " << i;

        if (convNode->inputZeroPoints.empty())
            return;

        auto weightsLayer = getCreatorLayer(convLayer->insData[1].lock()).lock();
        if (weightsLayer->type != "Const") {
            weightsLayer = getCreatorLayer(weightsLayer->insData[0].lock()).lock();
        }

        auto weightsBlob = dynamic_cast<TBlob<int8_t>*>(weightsLayer->blobs["custom"].get());
        auto weightsPtr = weightsBlob->buffer().as<int8_t*>();

        ptrdiff_t G = convLayer->_group;
        ptrdiff_t OC = weightsLayer->outData[0]->getDims()[0] / G;
        ptrdiff_t IC = weightsLayer->outData[0]->getDims()[1];
        ptrdiff_t KD = weightsLayer->outData[0]->getDims().size() == 5 ? weightsLayer->outData[0]->getDims()[2] : 1;
        ptrdiff_t KH = weightsLayer->outData[0]->getDims()[weightsLayer->outData[0]->getDims().size() - 2];
        ptrdiff_t KW = weightsLayer->outData[0]->getDims()[weightsLayer->outData[0]->getDims().size() - 1];

        for (size_t g = 0; g < G; g++) {
            for (size_t oc = 0; oc < OC; oc++) {
                int32_t a = 0;
                for (size_t ic = 0; ic < IC; ic++) {
                    for (size_t kd = 0; kd < KD; kd++) {
                        for (size_t kh = 0; kh < KH; kh++) {
                            for (size_t kw = 0; kw < KW; kw++) {
                                size_t widx = g * OC * IC * KD * KH * KW +
                                              oc * IC * KD * KH * KW +
                                              ic * KD * KH * KW +
                                              kd * KH * KW +
                                              kh * KW +
                                              kw;

                                auto w = static_cast<int32_t>(weightsPtr[widx]);

                                auto izp = !convNode->inputZeroPoints.empty() ? static_cast<int32_t>(convNode->inputZeroPoints[g * IC + ic]) : 0;
                                a += w * izp;

                                auto wzp = !convNode->weightsZeroPoints.empty() ? static_cast<int32_t>(convNode->weightsZeroPoints[g * OC + oc]) : 0;
                                a -= wzp * izp;
                            }
                        }
                    }
                }
                convNode->outputCompensation[g * OC + oc] = -a;
            }
        }
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto conv = graphNodes[i];
        if (!isSutableConvNode(conv)) continue;

        auto dataEltwise = conv->getParentEdgesAtPort(0)[0]->getParent();
        if (initializeInputZeroPoints(conv, dataEltwise)) {
            auto p_edge = dataEltwise->getParentEdgesAtPort(1)[0];
            removeEdge(graph, p_edge);

            graph.DropNode(dataEltwise);
        }

        auto weightsEltwise = conv->getParentEdgesAtPort(1)[0]->getParent();
        if (initializeWeightsZeroPoints(conv, weightsEltwise)) {
            auto p_edge = weightsEltwise->getParentEdgesAtPort(1)[0];
            removeEdge(graph, p_edge);

            graph.DropNode(weightsEltwise);
        }

        initializeOutputCompensation(conv);
    }
}

void MKLDNNGraphOptimizer::MergeGroupConvolution(MKLDNNGraph &graph) {
    for (auto node : graph.GetNodes()) {
        // Split with at least 2 Convolutions
        if (!IsOneOf(node->getType(), {Split}) || node->getChildEdges().size() < 2 ||
                !IsOneOf(node->getChildEdgeAt(0)->getChild()->getType(), {Convolution})) {
            continue;
        }
        bool canBeMerged = true;

        auto& split = node;

        auto convInEdge = split->getChildEdgeAt(0);
        auto conv = convInEdge->getChild();
        auto convOutEdge = conv->getChildEdgeAt(0);

        auto convType = conv->getType();
        auto convInDims = convInEdge->getDims();
        auto convOutDims = convOutEdge->getDims();

        // Convolutions of same the type with Concat as a child
        for (size_t i = 1; i < split->getChildEdges().size(); i++) {
            auto childEdge = split->getChildEdgeAt(i);
            auto child = childEdge->getChild();
            Type type = child->getType();

            if (convType != type || child->getChildEdgeAt(0)->getChild()->getType() != Concatenation ||
                    convOutDims != child->getChildEdgeAt(0)->getDims() || child->getChildEdges().size() != 1 ||
                    convInDims != childEdge->getDims()) {
                canBeMerged = false;
                break;
            }
        }

        if (!canBeMerged) continue;

        // TODO: Rewrite topology optimizer at all. it should be clean and understandable
        auto concat = conv->getChildEdgeAt(0)->getChild();
        // Merge and remove Convolution
        while (split->getChildEdges().size() > 1) {
            auto peerInEdge = split->getChildEdgeAt(1);
            auto peer = peerInEdge->getChild();
            conv->mergeWith(peer);
            convInDims[1] += (peerInEdge->getDims())[1];
            convOutDims[1] += (peer->getChildEdgeAt(0)->getDims())[1];
            peer->remove();
        }
        conv->inDims[0] = convInDims;
        conv->outDims[0] = convOutDims;

        conv->fuseWith(split);
        conv->fuseWith(concat);

        graph.DropNode(split);
        graph.DropNode(concat);
    }
}

//  WA: We need it until LP transformations will not optimize this pattern inside
void MKLDNNGraphOptimizer::MergeTwoEqualScaleShifts(MKLDNNGraph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableScaleShiftNode = [](MKLDNNNodePtr node) {
        if (node->getType() != Depthwise)
            return false;

        auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode *>(node.get());
        if (depthwiseNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << node->getName() << " to Depthwise node";

        if (depthwiseNode->getChildEdges().size() != 1)
            return false;

        if (depthwiseNode->getAlgorithm() != depthwise_scale_shift || depthwiseNode->isBroadcast())
            return false;

        return true;
    };

    auto isEqualScaleShiftNodes = [](MKLDNNNodePtr node1, MKLDNNNodePtr node2) {
        auto *depthwiseNode1 = dynamic_cast<MKLDNNDepthwiseNode *>(node1.get());
        auto *depthwiseNode2 = dynamic_cast<MKLDNNDepthwiseNode *>(node2.get());

        auto depthwiseLayer1 = depthwiseNode1->getCnnLayer();
        auto depthwiseLayer2 = depthwiseNode2->getCnnLayer();

        Blob::Ptr scalesBlob1 = depthwiseLayer1->blobs["weights"];
        Blob::Ptr shiftsBlob1 = depthwiseLayer1->blobs["biases"];
        Blob::Ptr scalesBlob2 = depthwiseLayer2->blobs["weights"];
        Blob::Ptr shiftsBlob2 = depthwiseLayer2->blobs["biases"];
        if (scalesBlob1 == nullptr || shiftsBlob1 == nullptr || scalesBlob2 == nullptr || shiftsBlob2 == nullptr)
            return false;

        if (scalesBlob1->size() != shiftsBlob1->size() || scalesBlob2->size() != shiftsBlob2->size()
            || scalesBlob1->size() != scalesBlob2->size()) return false;

        const float *scalesBufferPtr1 = scalesBlob1->buffer().as<float *>();
        const float *shiftsBufferPtr1 = shiftsBlob1->buffer().as<float *>();
        const float *scalesBufferPtr2 = scalesBlob2->buffer().as<float *>();
        const float *shiftsBufferPtr2 = shiftsBlob2->buffer().as<float *>();

        for (int i = 0; i < scalesBlob1->size(); i++)
            if (scalesBufferPtr1[i] != scalesBufferPtr2[i] || shiftsBufferPtr1[i] != shiftsBufferPtr2[i])
                return false;

        return true;
    };

    auto MergeScaleShiftNodes = [&](MKLDNNNodePtr childNode1, MKLDNNNodePtr childNode2) {
        auto parentNode = childNode2->getParentEdgeAt(0)->getParent();
        auto ccNode2 = childNode2->getChildEdgeAt(0)->getChild();
        graph.DropNode(childNode2);

        MKLDNNEdgePtr remEdge;
        for (auto edge : parentNode->getChildEdges()) {
            if (edge.lock()->getChild() == ccNode2) {
                remEdge = edge.lock();
                break;
            }
        }
        if (remEdge == nullptr)
            THROW_IE_EXCEPTION << "Edge was not found";
        remEdge->drop();
        graph.GetEdges().erase(std::remove(graph.GetEdges().begin(), graph.GetEdges().end(), remEdge), graph.GetEdges().end());

        if (childNode1->getChildEdgeAt(0)->getChild() != ccNode2) {
            auto iIndex = childNode1->getChildEdgeAt(0)->getInputNum();
            auto oIndex = remEdge->getOutputNum();
            MKLDNNEdgePtr newEdge(new MKLDNNEdge(childNode1, ccNode2, iIndex, oIndex));
            childNode1->addEdge(newEdge);
            graph.GetEdges().push_back(newEdge);
        }
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parentNode = graphNodes[i];
        if (parentNode->getChildEdges().size() != 2) continue;

        auto childNode1 = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableScaleShiftNode(childNode1)) continue;

        auto childNode2 = parentNode->getChildEdgeAt(1)->getChild();
        if (!isSutableScaleShiftNode(childNode2)) continue;

        if (!isEqualScaleShiftNodes(childNode1, childNode2)) continue;

        MergeScaleShiftNodes(childNode1, childNode2);
    }
}

void MKLDNNGraphOptimizer::MergeSigmoidAndMultiplyToSwish(MKLDNNGraph& graph) {
    auto& graphNodes = graph.GetNodes();
    std::vector<MKLDNNNodePtr> newNodes;

    MKLDNNNodePtr parentNode;
    MKLDNNNodePtr activationNode, eltwiseNode;
    MKLDNNEdgePtr remEdge;

    auto areSutableChildNodes = [&]() {
        auto childNode1 = parentNode->getChildEdgeAt(0)->getChild();
        auto childNode2 = parentNode->getChildEdgeAt(1)->getChild();

        if (childNode1->getType() == Activation && childNode2->getType() == Eltwise) {
            activationNode = childNode1;
            eltwiseNode = childNode2;
            remEdge = parentNode->getChildEdgeAt(1);
        } else if (childNode1->getType() == Eltwise && childNode2->getType() == Activation) {
            activationNode = childNode2;
            eltwiseNode = childNode1;
            remEdge = parentNode->getChildEdgeAt(0);
        } else {
            return false;
        }

        if (activationNode->getParentEdges().size() != 1 || activationNode->getChildEdges().size() != 1)
            return false;

        if (eltwiseNode->getParentEdges().size() != 2)
            return false;

        if (activationNode->getChildEdgeAt(0)->getChild() != eltwiseNode)
            return false;

        auto *activationNodePtr = dynamic_cast<MKLDNNActivationNode *>(activationNode.get());
        if (activationNodePtr == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << activationNode->getName() << " to Activation node";
        if (activationNodePtr->getAlgorithm() != eltwise_logistic)
            return false;

        auto *eltwiseNodePtr = dynamic_cast<MKLDNNEltwiseNode *>(eltwiseNode.get());
        if (eltwiseNodePtr == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << eltwiseNode->getName() << " to Eltwise node";
        auto *eltwiseLayer = dynamic_cast<EltwiseLayer*>(eltwiseNode->getCnnLayer().get());
        if (eltwiseLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot get eltwise layer " << eltwiseNode->getName();
        if (eltwiseLayer->_operation != EltwiseLayer::Prod)
            return false;

        return true;
    };

    auto MergeToSwish = [&]() {
        //  1. Remove edge Parent-Eltwise
        remEdge->drop();
        graph.GetEdges().erase(std::remove(graph.GetEdges().begin(), graph.GetEdges().end(), remEdge), graph.GetEdges().end());

        //  2. Remove Sigmoid node and edges Parent-Sigmoid and Sigmoid-Eltwise
        graph.DropNode(activationNode);
        remEdge = parentNode->getChildEdgeAt(0);
        auto oIndex = remEdge->getOutputNum();
        auto iIndex = remEdge->getInputNum();
        remEdge->drop();
        graph.GetEdges().erase(std::remove(graph.GetEdges().begin(), graph.GetEdges().end(), remEdge), graph.GetEdges().end());

        //  3. Create Swish node
        CNNLayerPtr swishLayer(new CNNLayer(*activationNode->getCnnLayer().get()));
        swishLayer->name = activationNode->getName() + "_Swish";
        swishLayer->type = "Swish";
        MKLDNNNodePtr swishNode(new MKLDNNActivationNode(swishLayer, graph.getEngine(), graph.weightsCache));

        //  4. Create edges Parent-Swish and Swish-Eltwise, connect to Swish node, add edges to graph
        MKLDNNEdgePtr beforeSwishEdge(new MKLDNNEdge(parentNode, swishNode, iIndex, 0));
        MKLDNNEdgePtr afterSwishEdge(new MKLDNNEdge(swishNode, eltwiseNode, 0, oIndex));
        swishNode->addEdge(beforeSwishEdge);
        swishNode->addEdge(afterSwishEdge);
        graph.GetEdges().push_back(beforeSwishEdge);
        graph.GetEdges().push_back(afterSwishEdge);
        newNodes.push_back(swishNode);

        //  5. Remove Eltwise node
        graph.DropNode(eltwiseNode);
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        parentNode = graphNodes[i];
        if (parentNode->getChildEdges().size() != 2)
            continue;

        if (!areSutableChildNodes()) continue;

        MergeToSwish();
    }
    for (int i = 0; i < newNodes.size(); i++) {
        graph.GetNodes().push_back(newNodes[i]);
    }
}

void MKLDNNGraphOptimizer::FuseBatchNormWithScale(MKLDNNGraph &graph) {
    auto &graphNodes = graph.GetNodes();

    for (int i = 0; i < graphNodes.size(); i++) {
        const auto& bn = graphNodes[i];
        if (bn->getType() == BatchNormalization) {
            const auto& outputNodes = graph.GetOutputNodes();
            const std::string node_name = bn->getName();
            // Check that the node is not output node
            if (std::find_if(outputNodes.begin(), outputNodes.end(),
                            [&node_name](const MKLDNNNodePtr& x) {
                                return x->getName() == node_name;}) == outputNodes.end()) {
                if (bn->getChildEdges().size() == 1) {
                    auto child = bn->getChildEdgeAt(0)->getChild();
                    if (child->type == Depthwise && child->getCnnLayer()->type == "ScaleShift") {
                        bn->fuseWith(child);
                        graph.DropNode(child);
                    }
                }
            }
        }
    }
}

#if defined(COMPILED_CPU_MKLDNN_ACTIVATION_NODE)
void MKLDNNGraphOptimizer::FuseConvolutionAndActivation(MKLDNNGraph &graph) {
    auto isOneOf = [&](mkldnn::algorithm alg, std::vector<mkldnn::algorithm> algs) {
        for (auto a : algs) {
            if (alg == a) {
                return true;
            }
        }
        return false;
    };

    auto& graphNodes = graph.GetNodes();

    auto isFusingSupported = [&](MKLDNNNodePtr conv, MKLDNNNodePtr activation) {
        if (!activation->getCnnLayer())
            return false;

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(activation.get());

        return activationNode &&
            (activationNode->getAlgorithm() == eltwise_relu ||
            (conv->getCnnLayer()->precision == Precision::FP32 &&
             isOneOf(activationNode->getAlgorithm(), {eltwise_elu, eltwise_logistic, eltwise_bounded_relu, eltwise_clamp,
                                                      eltwise_swish, eltwise_mish})));
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        if (graphNodes[i]->getType() == Convolution || graphNodes[i]->getType() == BinaryConvolution) {
            auto conv = graphNodes[i];

            auto fuse = [&] (MKLDNNNodePtr relu) {
                conv->fuseWith(relu);
            };

            if (conv->getChildEdges().size() == 1) {
                auto ch1 = conv->getChildEdgeAt(0)->getChild();

                if (isFusingSupported(conv, ch1)) {
                    fuse(ch1);

                    if (ch1->getChildEdges().size() == 1) {
                        auto ch2 = ch1->getChildEdgeAt(0)->getChild();

                        if (isFusingSupported(conv, ch2)) {
                            fuse(ch2);
                            graph.DropNode(ch2);
                        }
                    }
                    graph.DropNode(ch1);
                } else {
                    if (ch1->type == Pooling) {
                        auto pool = ch1;

                        auto* pLayer = dynamic_cast<PoolingLayer *>(pool->getCnnLayer().get());
                        if (pLayer == nullptr)
                            THROW_IE_EXCEPTION << "Cannot get pooling layer " << pool->getName();
                        bool is_max_pool = pLayer->_type == PoolingLayer::PoolType::MAX;

                        if (is_max_pool && pool->getChildEdges().size() == 1) {
                            auto ch2 = pool->getChildEdgeAt(0)->getChild();
                            if (isFusingSupported(conv, ch2)) {
                                fuse(ch2);
                                graph.DropNode(ch2);
                            }
                        }
                    }
                }
            }
        }
    }
}

void MKLDNNGraphOptimizer::FuseFullyConnectedAndSimpleOperation(MKLDNNGraph &graph) {
    auto isOneOf = [&](mkldnn::algorithm alg, std::vector<mkldnn::algorithm> algs) {
        for (auto a : algs) {
            if (alg == a) {
                return true;
            }
        }
        return false;
    };

    auto removeEdge = [](MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == edge) {
                edges.erase(it);
                return;
            }
        }
    };

    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == FullyConnected &&
               node->getChildEdges().size() == 1;
    };

    auto isSutableChildNode = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        if (!childNode->getCnnLayer())
            return false;

        if (childNode->getType() == Quantize) {
            auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(childNode.get());
            if (quantizeNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get quantize layer " << childNode->getName();

            if (parentNode->getParentEdgesAtPort(0)[0]->getDims().ndims() != 3) {
                return !quantizeNode->isBinarization();
            } else {
                return (quantizeNode->isInputLowBroadcast() && quantizeNode->isInputHighBroadcast() &&
                        quantizeNode->isOutputLowBroadcast() && quantizeNode->isOutputHighBroadcast() &&
                        !quantizeNode->isBinarization());
            }
        } else if (childNode->getType() == Depthwise) {
            auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode*>(childNode.get());
            if (depthwiseNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get depthwise layer " << childNode->getName();

            if (parentNode->getParentEdgesAtPort(0)[0]->getDims().ndims() != 3) {
                return ((depthwiseNode->getAlgorithm() == mkldnn::algorithm::depthwise_scale_shift &&
                         depthwiseNode->isWithBiases()) ||
                        (depthwiseNode->getAlgorithm() == mkldnn::algorithm::depthwise_prelu));
            } else {
                const auto &depthwiseLayer = depthwiseNode->getCnnLayer();
                if (depthwiseLayer == nullptr)
                    THROW_IE_EXCEPTION << "Cannot get scale shift layer " << depthwiseNode->getName();

                if (depthwiseNode->getAlgorithm() != mkldnn::algorithm::depthwise_scale_shift)
                    return false;

                Blob::Ptr scalesBlob = depthwiseLayer->blobs["weights"];
                if (scalesBlob == nullptr)
                    return false;

                Blob::Ptr shiftsBlob = depthwiseLayer->blobs["biases"];
                if (shiftsBlob == nullptr)
                    return false;

                const float* scalesBufferPtr = scalesBlob->buffer().as<float*>();
                const float* shiftsBufferPtr = shiftsBlob->buffer().as<float*>();

                if (scalesBlob->size() != shiftsBlob->size())
                    return false;

                for (int i = 1; i < scalesBlob->size(); i++)
                    if (scalesBufferPtr[0] != scalesBufferPtr[i])
                        return false;

                for (int i = 1; i < shiftsBlob->size(); i++)
                    if (shiftsBufferPtr[0] != shiftsBufferPtr[i])
                        return false;

                return true;
            }
        } else if (childNode->getType() == Activation) {
            auto* activationNode = dynamic_cast<MKLDNNActivationNode*>(childNode.get());
            if (activationNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get activation layer " << childNode->getName();

            return isOneOf(activationNode->getAlgorithm(), {eltwise_relu, eltwise_gelu, eltwise_elu, eltwise_logistic, eltwise_bounded_relu, eltwise_clamp});
        }

        return false;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

        parentNode->fuseWith(childNode);

        if (childNode->getType() == Quantize) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == FullyConnected)
                    continue;

                removeEdge(graph, p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}
#endif

#if defined (COMPILED_CPU_MKLDNN_DEPTHWISE_NODE)
void MKLDNNGraphOptimizer::FuseConvolutionAndDepthwise(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        bool isSutableConv = (node->getType() == Convolution) &&
                             node->getCnnLayer()->precision == Precision::FP32;
        bool isSutableBinConv = node->getType() == BinaryConvolution;
        return (isSutableConv || isSutableBinConv) && node->getChildEdges().size() == 1;
    };

    auto isSutableChildNode = [](MKLDNNNodePtr node) {
        if (node->getType() != Depthwise)
            return false;

        if (!node->getCnnLayer())
            return false;

        auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode *>(node.get());
        if (depthwiseNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get depthwise node " << node->getName();
        return ((depthwiseNode->getAlgorithm() == mkldnn::algorithm::depthwise_scale_shift && depthwiseNode->isWithBiases()) ||
                (depthwiseNode->getAlgorithm() == mkldnn::algorithm::depthwise_prelu));
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto conv = graphNodes[i];
        if (!isSutableParentNode(conv)) continue;

        auto depthwise0 = conv->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(depthwise0)) continue;

        conv->fuseWith(depthwise0);

        if (depthwise0->getChildEdges().size() == 1) {
            auto depthwise1 = depthwise0->getChildEdgeAt(0)->getChild();

            if (isSutableChildNode(depthwise1)) {
                conv->fuseWith(depthwise1);
                graph.DropNode(depthwise1);
            }
        }

        graph.DropNode(depthwise0);
    }
}
#endif

void MKLDNNGraphOptimizer::FuseConvolutionAndDWConvolution(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isConvolutionNode = [](MKLDNNNodePtr node) {
        return node->getType() == Convolution;
    };

    auto isBinaryConvolutionNode = [](MKLDNNNodePtr node) {
        return node->getType() == BinaryConvolution;
    };

    auto is1x1Convolution = [](ConvolutionLayer* layer) {
        return layer->_kernel[X_AXIS] == 1 && layer->_kernel[Y_AXIS] == 1;
    };

    auto isSutableParentConvolution = [&](MKLDNNNodePtr node) {
        if (isBinaryConvolutionNode(node)) {
            auto *layer = dynamic_cast<BinaryConvolutionLayer *>(node->getCnnLayer().get());
            if (layer == nullptr)
                THROW_IE_EXCEPTION << "Cannot get convolution layer " << node->getName();

            bool isSupportedParams = layer->_group == 1;
            if (!isSupportedParams) return false;
        } else {
            auto *layer = dynamic_cast<ConvolutionLayer *>(node->getCnnLayer().get());
            if (layer == nullptr)
                THROW_IE_EXCEPTION << "Cannot get convolution layer " << node->getName();

            auto* parentConvolutionNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
            if (parentConvolutionNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get convolution node " << node->getName();

            if (!parentConvolutionNode->weightsZeroPoints.empty())
                return false;

            bool isSupportedParams =
                    layer->_group == 1 &&
                    ((is1x1Convolution(layer) && layer->_stride[X_AXIS] == 1 && layer->_stride[Y_AXIS] == 1) || !is1x1Convolution(layer)) &&
                    (layer->outData[0].get()->getPrecision() == Precision::FP32 || layer->outData[0].get()->getPrecision() == Precision::U8) &&
                    node->getChildEdgeAt(0)->getDims().ndims() == 4;
            if (!isSupportedParams) return false;
        }

        return node->getChildEdges().size() == 1 && isConvolutionNode(node->getChildEdgeAt(0)->getChild());
    };

    auto isSutableChildConvolution = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        auto* childLayer = dynamic_cast<ConvolutionLayer*>(childNode->getCnnLayer().get());
        if (childLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot get convolution layer " << childNode->getName();

        if (!isBinaryConvolutionNode(parentNode)) {
            auto* parentLayer = dynamic_cast<ConvolutionLayer*>(parentNode->getCnnLayer().get());
            if (parentLayer == nullptr)
                THROW_IE_EXCEPTION << "Cannot get convolution layer " << parentNode->getName();

            if (parentLayer->outData[0].get()->getPrecision() != childLayer->outData[0].get()->getPrecision())
                return false;

            if (parentLayer->precision != childLayer->precision)
                return false;

            auto parentOutputPrecision = !parentNode->fusedWith.empty()
                    ? parentNode->fusedWith[parentNode->fusedWith.size() - 1]->getCnnLayer()->outData[0].get()->getPrecision()
                    : parentNode->getCnnLayer()->outData[0].get()->getPrecision();

            auto childOutputPrecision = !childNode->fusedWith.empty()
                    ? childNode->fusedWith[childNode->fusedWith.size() - 1]->getCnnLayer()->outData[0].get()->getPrecision()
                    : childNode->getCnnLayer()->outData[0].get()->getPrecision();

            if (parentOutputPrecision != childOutputPrecision)
                return false;
        }

        auto* childConvolutionNode = dynamic_cast<MKLDNNConvolutionNode*>(childNode.get());
        if (childConvolutionNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get convolution node " << childNode->getName();

        if (!childConvolutionNode->inputZeroPoints.empty() || !childConvolutionNode->weightsZeroPoints.empty())
            return false;

        bool withBias = (childLayer->_biases != nullptr && childLayer->_biases->size() != 0) ||
                        childConvolutionNode->getBaseIntputsNumber() == 3;

        auto allPads = getPaddings(*childLayer);
        bool isSupportedParams = childLayer->_out_depth == childLayer->_group &&
                                 childLayer->_out_depth != 1 &&
                                 childLayer->_kernel[X_AXIS] == 3 && childLayer->_kernel[Y_AXIS] == 3 &&
                                 allPads.begin[X_AXIS] == 1 && allPads.begin[Y_AXIS] == 1 &&
                                 childLayer->_dilation[X_AXIS] == 1 && childLayer->_dilation[Y_AXIS] == 1 &&
                                 withBias &&
                                 childNode->getChildEdgeAt(0)->getDims().ndims() == 4;

        return isSupportedParams;
    };

    auto isFusingWorthwhile = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        if (isBinaryConvolutionNode(parentNode)) {
            return true;
        }

        auto* layer = dynamic_cast<ConvolutionLayer*>(childNode->getCnnLayer().get());
        if (layer == nullptr)
            THROW_IE_EXCEPTION << "Cannot get convolution layer " << childNode->getName();

        auto inDims = childNode->inDims[0];
        auto outDims = childNode->outDims[0];
        int elemSize = MKLDNNExtensionUtils::sizeOfDataType(MKLDNNExtensionUtils::IEPrecisionToDataType(layer->precision));

        int L3_cache_size = mkldnn_get_cache_size(3, false);
        int dw_conv_input_size = inDims[0] * inDims[1] * inDims[2] * inDims[3] * elemSize;
        int dw_conv_output_size = outDims[0] * outDims[1]* outDims[2] * outDims[3] * elemSize;

        auto* parentConvolutionNode = dynamic_cast<MKLDNNConvolutionNode*>(parentNode.get());
        if (parentConvolutionNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get convolution node " << parentNode->getName();

        bool isInt8 = parentConvolutionNode->canBeExecutedInInt8();
        bool isAVX512NotSupported = !mkldnn::impl::cpu::mayiuse(impl::cpu::cpu_isa_t::avx512_common);

        return isInt8 ? isAVX512NotSupported : (dw_conv_input_size + dw_conv_output_size > L3_cache_size / 2);
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        if (!isConvolutionNode(graphNodes[i]) && !isBinaryConvolutionNode(graphNodes[i])) continue;

        auto parentConvNode = graphNodes[i];
        if (!isSutableParentConvolution(parentConvNode)) continue;

        auto childConvNode = parentConvNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildConvolution(parentConvNode, childConvNode)) continue;

        if (!isFusingWorthwhile(parentConvNode, childConvNode)) continue;

        parentConvNode->fuseWith(childConvNode);

        for (auto node : childConvNode->getFusedWith())
            parentConvNode->fuseWith(node);
        childConvNode->clearFusedWith();

        graph.DropDWConvNode(childConvNode);
    }
}

#if defined(COMPILED_CPU_MKLDNN_QUANTIZE_NODE)
void MKLDNNGraphOptimizer::FuseConvolutionAndQuantize(MKLDNNGraph &graph) {
    auto removeEdge = [](MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == edge) {
                edges.erase(it);
                return;
            }
        }
    };

    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        bool isSutableBinConv = node->getType() == Convolution;

        if (isSutableBinConv) {
            auto *convLayer = dynamic_cast<ConvolutionLayer *>(node->getCnnLayer().get());
            if (convLayer == nullptr)
                THROW_IE_EXCEPTION << "Cannot get convolution layer " << node->getName();

            return isSutableBinConv && node->getChildEdges().size() == 1;
        } else {
            return false;
        }
    };

    auto isSutableChildNode = [](MKLDNNNodePtr node) {
        if (!node->getCnnLayer())
            return false;

        if (node->getType() != Quantize)
            return false;

        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(node.get());
        if (quantizeNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get quantize layer " << node->getName();

        return !quantizeNode->isBinarization();
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSutableParentNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(child)) continue;

        parent->fuseWith(child);

        auto parents = child->parentEdges;
        for (size_t j = 0; j < parents.size(); j++) {
            auto p_edge = parents[j].lock();
            if (p_edge->getParent()->getType() == Convolution)
                continue;

            removeEdge(graph, p_edge);
        }

        graph.DropNode(child);
    }
}

void MKLDNNGraphOptimizer::FuseConvolutionAndSimpleOperation(MKLDNNGraph &graph) {
    auto isOneOf = [&](mkldnn::algorithm alg, std::vector<mkldnn::algorithm> algs) {
        for (auto a : algs) {
            if (alg == a) {
                return true;
            }
        }
        return false;
    };

    auto removeEdge = [](MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == edge) {
                edges.erase(it);
                return;
            }
        }
    };

    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Convolution &&
               node->getChildEdges().size() == 1 &&
               node->getCnnLayer()->precision == Precision::FP32;
    };

    auto isSutableChildNode = [&](MKLDNNNodePtr node) {
        if (!node->getCnnLayer())
            return false;

        if (node->getType() == Quantize) {
            auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(node.get());
            if (quantizeNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get quantize layer " << node->getName();

            return !quantizeNode->isBinarization();
        } else if (node->getType() == Depthwise) {
            auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode*>(node.get());
            if (depthwiseNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get depthwise layer " << node->getName();

            return ((depthwiseNode->getAlgorithm() == mkldnn::algorithm::depthwise_scale_shift && depthwiseNode->isWithBiases()) ||
                    (depthwiseNode->getAlgorithm() == mkldnn::algorithm::depthwise_prelu));
        } else if (node->getType() == Activation) {
            auto* activationNode = dynamic_cast<MKLDNNActivationNode*>(node.get());
            if (activationNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get activation layer " << node->getName();

            return isOneOf(activationNode->getAlgorithm(), {eltwise_relu, eltwise_elu, eltwise_logistic, eltwise_bounded_relu,
                                                            eltwise_clamp, eltwise_swish, eltwise_mish});
        }

        return false;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(childNode)) {
            parent++;
            continue;
        }

        parentNode->fuseWith(childNode);

        if (childNode->getType() == Quantize) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Convolution)
                    continue;

                removeEdge(graph, p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseBinaryConvolutionAndQuantize(MKLDNNGraph &graph) {
    auto removeEdge = [](MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == edge) {
                edges.erase(it);
                return;
            }
        }
    };

    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        bool isSutableBinConv = node->getType() == BinaryConvolution;
        return isSutableBinConv && node->getChildEdges().size() == 1;
    };

    auto isSutableChildNode = [](MKLDNNNodePtr node) {
        if (!node->getCnnLayer())
            return false;

        if (node->getType() != Quantize)
            return false;

        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(node.get());
        if (quantizeNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get quantize layer " << node->getName();

        return quantizeNode->isBinarization();
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSutableParentNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(child)) continue;

        parent->fuseWith(child);

        auto parents = child->parentEdges;
        for (size_t i = 0; i < parents.size(); i++) {
            auto p_edge = parents[i].lock();
            if (p_edge->getParent()->getType() == BinaryConvolution)
                continue;

            removeEdge(graph, p_edge);
        }

        graph.DropNode(child);
    }
}

void MKLDNNGraphOptimizer::FusePoolingAndQuantize(MKLDNNGraph &graph) {
    auto removeEdge = [](MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == edge) {
                edges.erase(it);
                return;
            }
        }
    };

    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        bool isSutablePooling = node->getType() == Pooling;

        if (isSutablePooling) {
            auto *poolingLayer = dynamic_cast<PoolingLayer *>(node->getCnnLayer().get());
            if (poolingLayer == nullptr)
                THROW_IE_EXCEPTION << "Cannot get Pooling layer " << node->getName();

            return node->getChildEdges().size() == 1 && poolingLayer->_type == PoolingLayer::AVG;
        } else {
            return false;
        }
    };

    auto isSutableChildNode = [](MKLDNNNodePtr node) {
        if (!node->getCnnLayer())
            return false;

        if (node->getType() != Quantize)
            return false;

        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(node.get());
        if (quantizeNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get quantize layer " << node->getName();

        return !quantizeNode->isBinarization();
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSutableParentNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(child)) continue;

        parent->fuseWith(child);

        auto parents = child->parentEdges;
        for (size_t i = 0; i < parents.size(); i++) {
            auto p_edge = parents[i].lock();
            if (p_edge->getParent()->getType() == Pooling)
                continue;

            removeEdge(graph, p_edge);
        }

        graph.DropNode(child);
    }
}
#endif

/**
 *  Check if there is a data dependency between parent and child
 *  BFS starting from parent and comparing with child
 *
 * @param parent head of BFS
 * @param child node we try to find
 * @return True if child is one of data supplier
 */
static bool is_data_dependency(const std::shared_ptr<MKLDNNNode> &parent,
                               const std::shared_ptr<MKLDNNNode> &child) {
    std::set<MKLDNNNode*> visited;
    std::list<MKLDNNNode*> nextLayers {parent.get()};

    for (; !nextLayers.empty();) {
        auto layer = *nextLayers.begin();
        if (layer == child.get()) return true;
        for (auto oe : layer->getChildEdges()) {
            auto nn = oe.lock()->getChild();
            if (visited.find(nn.get()) == visited.end()) {
                nextLayers.push_back(nn.get());
                visited.insert(nn.get());
            }
        }
        nextLayers.pop_front();
    }
    return false;
}

/*
 *  Before:
 *
 *        ***             ***                   ***             ***
 *         |               |                     |               |
 *    +========+       +========+           +========+       +========+
 *    |  any   |       | conv 2 |           |  any   |       | conv 2 |
 *    +========+       +========+           +========+       +========+
 *         |               |                     |               |
 *      +=====================+               +=====================+
 *      |         Sum         |      or       |         Sum         |
 *      +=====================+               +=====================+
 *                 |                                     |
 *         +===============+                            ***
 *         |     Relu      |
 *         +===============+
 *                 |
 *                ***
 *
 *  After:
 *
 *        ***             ***
 *         |               |
 *    +========+       +========+
 *    |  any   |-------|        |
 *    +========+       | conv2  |
 *                     |   +    |
 *                     |  sum   |
 *                     |   +    |
 *                     | [relu] |
 *                     |        |
 *                     +========+
 *                         |
 *                 +-------+
 *                 |
 *                ***
 */

#if defined(COMPILED_CPU_MKLDNN_ELTWISE_NODE)
void MKLDNNGraphOptimizer::FuseConvolutionSumAndConvolutionSumActivation(MKLDNNGraph &graph) {
    std::vector<MKLDNNNodePtr> &graphNodes = graph.GetNodes();

    auto isOneOf = [&](mkldnn::algorithm alg, std::vector<mkldnn::algorithm> algs) {
        for (auto a : algs) {
            if (alg == a) {
                return true;
            }
        }
        return false;
    };

    auto isFusingSupported = [&](MKLDNNNodePtr conv, MKLDNNNodePtr activation) {
        if (!activation->getCnnLayer())
            return false;

#if defined(COMPILED_CPU_MKLDNN_ACTIVATION_NODE)
        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(activation.get());

        return activationNode &&
            (activationNode->getAlgorithm() == eltwise_relu ||
            (conv->getCnnLayer()->precision == Precision::FP32 &&
             isOneOf(activationNode->getAlgorithm(), {eltwise_elu, eltwise_logistic, eltwise_bounded_relu, eltwise_clamp,
                                                      eltwise_swish, eltwise_mish})));
#else
        return false;
#endif
    };

    for (auto &graphNode : graphNodes) {
        if (graphNode->getType() != Eltwise)
            continue;

        if (!std::dynamic_pointer_cast<MKLDNNEltwiseNode>(graphNode)->isSum()) continue;
        if (!std::dynamic_pointer_cast<MKLDNNEltwiseNode>(graphNode)->isUnitScales()) continue;
        if (std::dynamic_pointer_cast<MKLDNNEltwiseNode>(graphNode)->isWithBroadcast()) continue;

        // TODO: Enlarge to several inputs
        bool isSutableNode = graphNode->getParentEdges().size() == 2;
        if (!isSutableNode)
            continue;

        auto parent1 = graphNode->getParentEdgeAt(0)->getParent();
        auto parent2 = graphNode->getParentEdgeAt(1)->getParent();

        bool isSutableParent1 = parent1->getType() == Convolution || parent1->getType() == BinaryConvolution;
        bool isSutableParent2 = parent2->getType() == Convolution || parent2->getType() == BinaryConvolution;

        auto* parentNode1 = dynamic_cast<MKLDNNConvolutionNode *>(parent1.get());
        if (parentNode1) {
            if (!parentNode1->canBeExecutedInInt8()) {
                isSutableParent1 = isSutableParent1 && parentNode1->getFusedWith().empty();
            }
        }

        auto* parentNode2 = dynamic_cast<MKLDNNConvolutionNode *>(parent2.get());
        if (parentNode2) {
            if (!parentNode2->canBeExecutedInInt8()) {
                isSutableParent2 = isSutableParent2 && parentNode2->getFusedWith().empty();
            }
        }

        if (!isSutableParent1 && !isSutableParent2)
            continue;

        auto mergedConv = isSutableParent1 ? parent1 : parent2;
        auto peerNode = isSutableParent1 ? parent2 : parent1;
        if (isSutableParent1 && isSutableParent2) {
            if ((peerNode->getType() == Convolution || peerNode->getType() == BinaryConvolution) &&
                mergedConv->getChildEdges().size() != 1) {
                mergedConv = parent2;
                peerNode = parent1;
            }
        }
        if (peerNode->isConstant())
            continue;
        auto sum = graphNode;
        auto lastNode = sum;

        bool fuse_allowed = mergedConv->getChildEdges().size() == 1;
        for (size_t j = 0; fuse_allowed && j < mergedConv->getParentEdges().size(); j++)
            if (mergedConv->getParentEdgeAt(j)->getParent() == peerNode)
                fuse_allowed = false;

        // Fused Conv+Sum prim will be used inplace. That's mean that input blob will
        // be overwritten. Should verify that all other consumer already read it and
        // we can spoil input data.
        // TODO: rewrite once we add "Inplace" reporting mechanism
        for (auto & edge : peerNode->getChildEdges()) {
            if (!fuse_allowed)
                break;
            fuse_allowed &= is_data_dependency(edge.lock()->getChild(), sum);
        }
        if (!fuse_allowed) continue;

        if (graphNode->getChildEdges().size() == 1 &&
                isFusingSupported(graphNode, graphNode->getChildEdgeAt(0)->getChild())) {
            auto relu_shared = graphNode->getChildEdgeAt(0)->getChild();
            lastNode = relu_shared;
            mergedConv->fuseWith(sum);
        }

        mergedConv->fuseWith(lastNode);

        if (mergedConv->fusedWith.size() > 0 &&
           (mergedConv->fusedWith[0]->getType() == Convolution || mergedConv->fusedWith[0]->getType() == BinaryConvolution)) {
            // Merged with DW_conv. Shape may change
            mergedConv->inDims.push_back(mergedConv->fusedWith[0]->outDims[0]);
        } else {
            mergedConv->inDims.push_back(mergedConv->outDims[0]);
        }

        size_t childIdx = 0lu;
        for (; childIdx < peerNode->getChildEdges().size(); childIdx++) {
            if (peerNode->getChildEdgeAt(childIdx)->getChild() == sum) {
                break;
            }
        }

        int peer_port = peerNode->getChildEdgeAt(childIdx)->getInputNum();
        peerNode->getChildEdgeAt(childIdx)->drop();

        int childPort = 1;
        auto* mergedConvNode = dynamic_cast<MKLDNNConvolutionNode*>(mergedConv.get());
        if (mergedConvNode != nullptr)
            childPort = mergedConvNode->getParentEdges().size();

        MKLDNNEdgePtr edgePtr(new MKLDNNEdge(peerNode, mergedConv, peer_port, childPort));
        graph.GetEdges().push_back(edgePtr);

        mergedConv->addEdge(edgePtr);

        std::vector<MKLDNNEdgeWeakPtr> edges_to_reconnect = lastNode->getChildEdges();
        for (auto &edge_w : edges_to_reconnect) {
            auto edge = edge_w.lock();
            auto child = edge->getChild();
            int idxParent = edge->getInputNum();
            int idxChild = edge->getOutputNum();

            // reconnect after  activation/sum. Port index must be 0
            IE_ASSERT(idxParent == 0);

            edge->drop();

            MKLDNNEdgePtr newEdge(new MKLDNNEdge(mergedConv, child, idxParent, idxChild));
            graph.GetEdges().push_back(newEdge);
            child->addEdge(newEdge);
        }

        if (lastNode != sum) {
            lastNode->remove();
        }
        sum->remove();
    }
}
#endif

void MKLDNNGraphOptimizer::FuseMVNAndSimpleOperation(MKLDNNGraph &graph) {
    auto removeEdge = [](MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == edge) {
                edges.erase(it);
                return;
            }
        }
    };

    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        bool isSutableMVN = (node->getType() == MVN) && (node->inDims[0].ndims() == 4 || node->inDims[0].ndims() == 5);

        if (isSutableMVN) {
            auto *mvnLayer = dynamic_cast<MVNLayer *>(node->getCnnLayer().get());
            if (mvnLayer == nullptr)
                THROW_IE_EXCEPTION << "Cannot get MVN layer " << node->getName();

            return node->getChildEdges().size() == 1 && mvnLayer->across_channels == 0 && mvnLayer->normalize == 1;
        } else {
            return false;
        }
    };

    auto isSutableChildNode = [](MKLDNNNodePtr node) {
        if (!node->getCnnLayer())
            return false;

        if (node->getType() == Quantize) {
            auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(node.get());
            if (quantizeNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get quantize layer " << node->getName();
            return !quantizeNode->isBinarization();
        } else if (node->getType() == Depthwise) {
            auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode*>(node.get());
            if (depthwiseNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get depthwise layer " << node->getName();
            return depthwiseNode->cnnLayer->type == "ScaleShift";
        } else if (node->getType() == Activation) {
            auto* activationNode = dynamic_cast<MKLDNNActivationNode*>(node.get());
            if (activationNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get activation layer " << node->getName();
            return activationNode->getAlgorithm() == eltwise_relu;
        }

        return false;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(childNode)) {
            parent++;
            continue;
        }

        parentNode->fuseWith(childNode);

        if (childNode->getType() == Quantize) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == MVN)
                    continue;

                removeEdge(graph, p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseResampleAndSimpleOperation(MKLDNNGraph &graph) {
    auto removeEdge = [](MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == edge) {
                edges.erase(it);
                return;
            }
        }
    };

    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        bool isSutableResample = (node->getType() == Resample) && (node->inDims[0].ndims() == 4 || node->inDims[0].ndims() == 5);

        if (isSutableResample) {
            auto *resampleLayer = node->getCnnLayer().get();
            if (resampleLayer == nullptr)
                THROW_IE_EXCEPTION << "Cannot get Resample layer " << node->getName();

            return node->getChildEdges().size() == 1 && resampleLayer->GetParamAsString("type") == "caffe.ResampleParameter.NEAREST";
        } else {
            return false;
        }
    };

    auto isSutableChildNode = [](MKLDNNNodePtr node) {
        if (!node->getCnnLayer())
            return false;

        if (node->getType() == Quantize) {
            auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(node.get());
            if (quantizeNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get quantize layer " << node->getName();
            return !quantizeNode->isBinarization();
        } else if (node->getType() == Depthwise) {
            auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode*>(node.get());
            if (depthwiseNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get depthwise layer " << node->getName();
            return depthwiseNode->cnnLayer->type == "ScaleShift";
        } else if (node->getType() == Activation) {
            auto* activationNode = dynamic_cast<MKLDNNActivationNode*>(node.get());
            if (activationNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get activation layer " << node->getName();
            return activationNode->getAlgorithm() == eltwise_relu;
        }

        return false;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(childNode)) {
            parent++;
            continue;
        }

        parentNode->fuseWith(childNode);

        if (childNode->getType() == Quantize) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Resample)
                    continue;

                removeEdge(graph, p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseNormalizeAndSimpleOperation(MKLDNNGraph &graph) {
    auto isOneOf = [&](mkldnn::algorithm alg, std::vector<mkldnn::algorithm> algs) {
        for (auto a : algs) {
            if (alg == a) {
                return true;
            }
        }
        return false;
    };

    auto removeEdge = [](MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == edge) {
                edges.erase(it);
                return;
            }
        }
    };

    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        bool isSutableNormalize = node->getType() == Normalize;

        if (isSutableNormalize) {
            return node->getChildEdges().size() == 1;
        } else {
            return false;
        }
    };

    auto isSutableChildNode = [&](MKLDNNNodePtr node) {
        if (!node->getCnnLayer())
            return false;

        if (node->getType() == Quantize) {
            auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(node.get());
            if (quantizeNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get quantize layer " << node->getName();
            return !quantizeNode->isBinarization();
        } else if (node->getType() == Depthwise) {
            auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode*>(node.get());
            if (depthwiseNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get depthwise layer " << node->getName();
            return ((depthwiseNode->getAlgorithm() == mkldnn::algorithm::depthwise_scale_shift && depthwiseNode->isWithBiases()) ||
                    (depthwiseNode->getAlgorithm() == mkldnn::algorithm::depthwise_prelu));
        } else if (node->getType() == Activation) {
            auto* activationNode = dynamic_cast<MKLDNNActivationNode*>(node.get());
            if (activationNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get activation layer " << node->getName();
            return isOneOf(activationNode->getAlgorithm(), {eltwise_relu, eltwise_gelu, eltwise_elu, eltwise_logistic,
                eltwise_bounded_relu, eltwise_clamp, eltwise_tanh, eltwise_swish, eltwise_mish, eltwise_linear, eltwise_abs,
                eltwise_square, eltwise_sqrt});
        }
        return false;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(childNode)) {
            parent++;
            continue;
        }

        parentNode->fuseWith(childNode);

        if (childNode->getType() == Quantize) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Normalize)
                    continue;

                removeEdge(graph, p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseEltwiseAndSimple(MKLDNNGraph &graph) {
    auto isOneOf = [&](mkldnn::algorithm alg, std::vector<mkldnn::algorithm> algs) {
        for (auto a : algs) {
            if (alg == a) {
                return true;
            }
        }
        return false;
    };

    auto removeEdge = [](MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == edge) {
                edges.erase(it);
                return;
            }
        }
    };

    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        bool isSutableEltwise = node->getType() == Eltwise;

        if (isSutableEltwise) {
            auto *eltwiseLayer = dynamic_cast<EltwiseLayer *>(node->getCnnLayer().get());
            if (eltwiseLayer == nullptr)
                THROW_IE_EXCEPTION << "Cannot get Eltwise layer " << node->getName();

            ptrdiff_t maxChannels = 1;
            for (size_t i = 0; i < node->getParentEdges().size(); i++) {
                if (node->getParentEdgeAt(0)->getDims().ndims() != node->getParentEdgeAt(i)->getDims().ndims())
                    return false;
                if (node->getParentEdgeAt(i)->getDims().ndims() != 2 &&
                    node->getParentEdgeAt(i)->getDims().ndims() != 4 &&
                    node->getParentEdgeAt(i)->getDims().ndims() != 5)
                    return false;
                if (maxChannels < node->getParentEdgeAt(i)->getDims()[1])
                    maxChannels = node->getParentEdgeAt(i)->getDims()[1];
            }

            int simdWidth = mkldnn::impl::cpu::mayiuse(impl::cpu::cpu_isa_t::avx512_common) ? 16 :
                            mkldnn::impl::cpu::mayiuse(impl::cpu::cpu_isa_t::avx2) ? 8 : 4;
            if (maxChannels < simdWidth)
                return false;

            return node->getChildEdges().size() == 1 &&
                   (eltwiseLayer->_operation == EltwiseLayer::Sum || eltwiseLayer->_operation == EltwiseLayer::Prod) &&
                   !node->isFusedWith(Quantize);
        } else {
            return false;
        }
    };

    auto isSutableChildNode = [&](MKLDNNNodePtr node) {
        if (!node->getCnnLayer())
            return false;

        if (node->getType() == Quantize) {
            auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(node.get());
            if (quantizeNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get quantize layer " << node->getName();
            return !quantizeNode->isBinarization();
        } else if (node->getType() == Activation) {
            // Applicability was narrowed down in order not to affect FP32 topologies
            if (node->getChildEdges().size() != 1)
                return false;
            if (node->getChildEdgeAt(0)->getChild()->getType() != Quantize)
                return false;

            auto *activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
            if (activationNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get activation layer " << node->getName();
            return isOneOf(activationNode->getAlgorithm(), {eltwise_relu, eltwise_elu, eltwise_logistic, eltwise_bounded_relu,
                                                            eltwise_clamp, eltwise_swish, eltwise_mish});
        }

        return false;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(childNode)) {
            parent++;
            continue;
        }

        parentNode->fuseWith(childNode);

        if (childNode->getType() == Quantize) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Eltwise)
                    continue;

                removeEdge(graph, p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::RemoveIdentityOperator(MKLDNNGraph &graph) {
    for (MKLDNNNodePtr& node : graph.GetNodes()) {
        bool toDrop = false;

        if (node->getType() == Power) {
            PowerLayer* l = dynamic_cast<PowerLayer*>(node->getCnnLayer().get());
            if (l == nullptr)
                THROW_IE_EXCEPTION << "Cannot get power layer " << node->getName();

            if (l->power == 1.0f && l->scale == 1.0f && l->offset == 0.0f) toDrop = true;
        }

        if (node->getType() == Depthwise && node->getCnnLayer()->type == "ScaleShift") {
            ScaleShiftLayer* l = dynamic_cast<ScaleShiftLayer*>(node->getCnnLayer().get());
            if (l == nullptr)
                THROW_IE_EXCEPTION << "Cannot get scale shift layer " << node->getName();

            if (l->_weights == nullptr && l->_biases == nullptr) toDrop = true;
        }

        if (node->getType() == Copy) toDrop = true;

        if (toDrop) graph.DropNode(node);
    }
}

#if defined (COMPILED_CPU_MKLDNN_REORDER_NODE)
void MKLDNNGraphOptimizer::DropDoubleReorders(MKLDNNGraph &graph) {
    std::set<MKLDNNNodePtr> processed;
    std::vector<MKLDNNNodePtr> newNodes;
    for (MKLDNNNodePtr& node : graph.GetNodes()) {
        if (processed.find(node) == processed.end() && node->getType() == Reorder
            && node->getChildEdges().size() == 1
            && node->getChildEdgeAt(0)->getChild()->getType() == Reorder ) {
            auto nextNode = node->getChildEdgeAt(0)->getChild();
            MKLDNNReorderNode* n = dynamic_cast<MKLDNNReorderNode*>(node.get());
            if (n == nullptr)
                THROW_IE_EXCEPTION << "Cannot get reorder layer " << node->getName();
            MKLDNNReorderNode* nn = dynamic_cast<MKLDNNReorderNode*>(nextNode.get());
            if (nn == nullptr)
                THROW_IE_EXCEPTION << "Cannot get reorder layer " << nextNode->getName();

            auto scales = n->_scales;

            if (n->_scales != nullptr && nn->_scales != nullptr) {
                THROW_IE_EXCEPTION << "Merging scales of two subsequent reorders is unsupported yet";
            } else {
                if (scales == nullptr) {
                    scales = nn->_scales;
                }
            }

            MKLDNNNodePtr p = n->getParentEdgeAt(0)->getParent();
            MKLDNNNodePtr c = nn->getChildEdgeAt(0)->getChild();

            auto oldEdgeNum = n->getParentEdgeAt(0)->getInputNum();

            graph.DropNode(node);
            graph.DropNode(nextNode);

            processed.insert(node);
            processed.insert(nextNode);

            MKLDNNEdgePtr edge;
            for (auto cur : p->getChildEdgesAtPort(oldEdgeNum)) {
                if (cur->getChild() == c)
                    edge = cur;
            }
            if (!edge) THROW_IE_EXCEPTION << "Inappropriate graph processing";


            std::string layerName = edge->getParent()->getName() + "_ScaleReorder_" + edge->getChild()->getName();
            CNNLayerPtr layer(new CNNLayer({layerName,
                                            "Reorder",
                                            n->getInput().getPrecision()}));
            MKLDNNNodePtr newReorder(new MKLDNNReorderNode(layer, graph.getEngine(), graph.weightsCache));
            auto *reorderPtr = dynamic_cast<MKLDNNReorderNode *>(newReorder.get());
            if (reorderPtr) {
                reorderPtr->setDescs(n->getInput(), nn->getOutput());
                reorderPtr->_scales = scales;
            }

            // new !!!
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

            graph.GetEdges().push_back(beforeNode);
            graph.GetEdges().push_back(afterNode);

            // Just to check accordance
            afterNode->getDesc();
            beforeNode->getDesc();

            newNodes.push_back(newReorder);
            graph.GetEdges().erase(std::remove(graph.GetEdges().begin(), graph.GetEdges().end(), edge), graph.GetEdges().end());
        }
    }
    for (MKLDNNNodePtr& node : newNodes) {
        graph.GetNodes().push_back(node);
    }
}

void MKLDNNGraphOptimizer::DropConvertReorder(MKLDNNGraph& graph) {
    for (auto input : graph.GetNodes()) {
        if (input->getType() != Input) {
            continue;
        }

        auto inTD = input->getCnnLayer().get()->outData[0]->getTensorDesc();
        for (size_t i = 0; i < input->getChildEdges().size(); i++) {
            auto inputEdge = input->getChildEdgeAt(i);
            auto convert = inputEdge->getChild();
            if (convert->getType() == Convert) {
                for (int j = 0; j < convert->getChildEdges().size(); j++) {
                    auto convertEdge = convert->getChildEdgeAt(j);
                    auto reorder = convertEdge->getChild();
                    if (reorder->getType() == Reorder) {
                        MKLDNNReorderNode* rn = dynamic_cast<MKLDNNReorderNode*>(reorder.get());
                        auto rnOutput = rn->getOutput();
                        if (inTD.getPrecision() == rnOutput.getPrecision() &&
                            inTD.getLayout() == rnOutput.getLayout() &&
                            inTD.getDims() == rnOutput.getDims()) {
                            auto avterReorder = reorder->getChildEdgeAt(0)->getChild();
                            auto oldEdgeNum = reorder->getChildEdgeAt(0)->getOutputNum();
                            reorder->getChildEdgeAt(0)->drop();
                            convertEdge->drop();

                            MKLDNNEdgePtr newEdge(new MKLDNNEdge(input, avterReorder, i, oldEdgeNum));
                            graph.GetEdges().push_back(newEdge);
                            input->addEdge(newEdge);
                            j--;
                        }
                    }
                }
            }
        }
    }
}
#endif

void MKLDNNGraphOptimizer::RemoveIOScaleShifts(MKLDNNGraph &graph) {
    for (MKLDNNNodePtr& node : graph.GetNodes()) {
        if (node->getType() == Depthwise && node->getCnnLayer()->type == "ScaleShift") {
            ScaleShiftLayer* l = dynamic_cast<ScaleShiftLayer*>(node->getCnnLayer().get());
            if (l == nullptr)
                THROW_IE_EXCEPTION << "Cannot get scale shift layer " << node->getName();

            auto cur = l->insData[0].lock();
            if (cur == nullptr) {
                THROW_IE_EXCEPTION << "[MKLDNN] error - invalid input data";
            }
            if (cur->getTensorDesc().getPrecision() != l->outData[0]->getTensorDesc().getPrecision()) {
                if (node->name.find("_iScaleShift_") != std::string::npos) {
                    auto child = node->childEdges[0].lock()->getChild();
#if defined (COMPILED_CPU_MKLDNN_REORDER_NODE)
                    if (child->type == Reorder) {
                        MKLDNNReorderNode* rn = dynamic_cast<MKLDNNReorderNode*>(child.get());
                        if (rn != nullptr) {
                            rn->_scales = l->_weights;
                            graph.DropNode(node);
                        }
                    } else {
#else
                        THROW_IE_EXCEPTION << "Strange case. No Reorder after iScaleShift";
#endif
#if defined (COMPILED_CPU_MKLDNN_REORDER_NODE)
                    }
#endif
                } else if (node->name.find("_oScaleShift_") != std::string::npos) {
                    auto parent = node->parentEdges[0].lock()->getParent();

#if defined (COMPILED_CPU_MKLDNN_REORDER_NODE)
                    if (parent->type == Reorder) {
                        MKLDNNReorderNode* rn = dynamic_cast<MKLDNNReorderNode*>(parent.get());
                        if (rn != nullptr) {
                            rn->_scales = l->_weights;
                            graph.DropNode(node);
                        }
                    } else {
#else
                        THROW_IE_EXCEPTION << "Strange case. No Reorder before oScaleShift";
#endif
#if defined (COMPILED_CPU_MKLDNN_REORDER_NODE)
                    }
#endif
                }
            }
        }
    }
}

bool MKLDNNGraphOptimizer::IsOneOf(Type type, std::vector<Type> types) {
    for (auto tp : types) {
        if (type == tp) {
            return true;
        }
    }
    return false;
}

void MKLDNNGraphOptimizer::FuseBroadcastAndEltwise(MKLDNNGraph &graph) {
    std::vector<MKLDNNNodePtr>& graphNodes = graph.GetNodes();

    for (auto &graphNode : graphNodes) {
        if (graphNode->getType() != Generic
                || graphNode->getTypeStr() != "Broadcast"
                || graphNode->getChildEdges().size() != 1lu
                || graphNode->getChildEdgeAt(0)->getChild()->getType() != Eltwise)
            continue;

        MKLDNNNodePtr& broadcastNode = graphNode;
        MKLDNNNodePtr eltwiseNode = broadcastNode->getChildEdgeAt(0)->getChild();
        eltwiseNode->inDims[broadcastNode->getChildEdgeAt(0)->getOutputNum()]
                = broadcastNode->getParentEdgeAt(0)->getDims();

        auto& edges = graph.GetEdges();
        for (size_t i = 1lu; i < broadcastNode->getParentEdges().size(); i++) {
            auto constParent = broadcastNode->getParentEdgeAt(i)->getParent();
            for (auto it = edges.begin(); it != edges.end(); it++) {
                if ((*it) == constParent->getChildEdgeAt(0)) {
                    edges.erase(it);
                    constParent->remove();
                    break;
                }
            }
        }
        graph.DropNode(broadcastNode);
    }
}

void MKLDNNGraphOptimizer::FuseClampAndQuantize(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableClampNode = [](MKLDNNNodePtr node) {
        if (node->getType() != Activation)
            return false;

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
        if (activationNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << node->getName() << " to Activation node";

        if (activationNode->getChildEdges().size() != 1)
            return false;

        if (activationNode->getAlgorithm() != eltwise_clamp)
            return false;

        return true;
    };

    auto isSutableQuantizeNode = [](MKLDNNNodePtr node) {
        if (node->getType() != Quantize)
            return false;

        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(node.get());
        if (quantizeNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << node->getName() << " to Quantize node";

        return !quantizeNode->isBinarization();
    };

    auto fuseClampAndQuantizeNodes = [](MKLDNNNodePtr parent, MKLDNNNodePtr child) {
        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(parent.get());
        if (activationNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << parent->getName() << " to Activation node";

        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(child.get());
        if (quantizeNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << child->getName() << " to Quantize node";

        const std::vector<float>& cropLowData = quantizeNode->getCropLow();
        const std::vector<float>& cropHighData = quantizeNode->getCropHigh();

        std::vector<float> newCropLow(cropLowData.size());
        std::vector<float> newCropHigh(cropHighData.size());
        for (int i = 0; i < cropLowData.size(); i++)
            newCropLow[i] = std::max(cropLowData[i], activationNode->getBeta());
        for (int i = 0; i < cropHighData.size(); i++)
            newCropHigh[i] = std::min(cropHighData[i], activationNode->getAlpha());

        quantizeNode->setCropLow(newCropLow);
        quantizeNode->setCropHigh(newCropHigh);

        return true;
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSutableClampNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSutableQuantizeNode(child)) continue;

        if (fuseClampAndQuantizeNodes(parent, child)) {
            graph.DropNode(parent);
        }
    }
}

void MKLDNNGraphOptimizer::FuseScaleShiftAndQuantize(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableScaleShiftNode = [](MKLDNNNodePtr node) {
        if (node->getType() != Depthwise)
            return false;

        auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode *>(node.get());
        if (depthwiseNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << node->getName() << " to Depthwise node";

        if (depthwiseNode->getChildEdges().size() != 1)
            return false;

        if (depthwiseNode->getAlgorithm() != depthwise_scale_shift || depthwiseNode->isBroadcast())
            return false;

        return true;
    };

    auto isSutableQuantizeNode = [](MKLDNNNodePtr node) {
        if (node->getType() != Quantize)
            return false;

        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(node.get());
        if (quantizeNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << node->getName() << " to Quantize node";

        return !quantizeNode->isBinarization();
    };

    auto fuseScaleShiftAndQuantizeNodes = [](MKLDNNNodePtr parent, MKLDNNNodePtr child) {
        auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode *>(parent.get());
        if (depthwiseNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << parent->getName() << " to Depthwise node";

        auto depthwiseLayer = depthwiseNode->getCnnLayer();
        if (depthwiseLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot get scale shift layer " << depthwiseNode->getName();

        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(child.get());
        if (quantizeNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << child->getName() << " to Quantize node";

        Blob::Ptr scalesBlob = depthwiseLayer->blobs["weights"];
        if (scalesBlob == nullptr)
            return false;

        Blob::Ptr shiftsBlob = depthwiseLayer->blobs["biases"];
        if (shiftsBlob == nullptr)
            return false;

        const float* scalesBufferPtr = scalesBlob->buffer().as<float*>();
        const float* shiftsBufferPtr = shiftsBlob->buffer().as<float*>();

        if (scalesBlob->size() != shiftsBlob->size())
            return false;

        for (int i = 0; i < scalesBlob->size(); i++)
            if (scalesBufferPtr[i] <= 0.f)
                return false;

        const std::vector<float>& cropLowData = quantizeNode->getCropLow();
        const std::vector<float>& cropHighData = quantizeNode->getCropHigh();
        const std::vector<float>& inputScaleData = quantizeNode->getInputScale();
        const std::vector<float>& inputShiftData = quantizeNode->getInputShift();

        std::vector<float> newCropLow(scalesBlob->size());
        std::vector<float> newCropHigh(scalesBlob->size());
        std::vector<float> newInputScale(scalesBlob->size());
        std::vector<float> newInputShift(scalesBlob->size());

        for (int i = 0; i < newCropLow.size(); i++) {
            float cl = cropLowData.size() == 1 ? cropLowData[0] : cropLowData[i];

            newCropLow[i] = (cl - shiftsBufferPtr[i]) / scalesBufferPtr[i];
        }

        for (int i = 0; i < newCropHigh.size(); i++) {
            float ch = cropHighData.size() == 1 ? cropHighData[0] : cropHighData[i];

            newCropHigh[i] = (ch - shiftsBufferPtr[i]) / scalesBufferPtr[i];
        }

        for (int i = 0; i < newInputScale.size(); i++) {
            float isc = inputScaleData.size() == 1 ? inputScaleData[0] : inputScaleData[i];

            newInputScale[i] = isc * scalesBufferPtr[i];
        }

        for (int i = 0; i < newInputShift.size(); i++) {
            float isc = inputScaleData.size() == 1 ? inputScaleData[0] : inputScaleData[i];
            float ish = inputShiftData.size() == 1 ? inputShiftData[0] : inputShiftData[i];

            newInputShift[i] = ish + shiftsBufferPtr[i] * isc;
        }

        quantizeNode->setCropLow(newCropLow);
        quantizeNode->setCropHigh(newCropHigh);
        quantizeNode->setInputScale(newInputScale);
        quantizeNode->setInputShift(newInputShift);

        return true;
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSutableScaleShiftNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSutableQuantizeNode(child)) continue;

        if (fuseScaleShiftAndQuantizeNodes(parent, child)) {
            graph.DropNode(parent);
        }
    }
}
