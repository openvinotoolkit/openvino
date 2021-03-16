// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_graph_optimizer.h"

#include "mkldnn_extension_utils.h"
#include "nodes/mkldnn_reshape_node.h"
#include "nodes/mkldnn_pooling_node.h"
#include "nodes/mkldnn_eltwise_node.h"
#include "nodes/mkldnn_concat_node.h"
#include "nodes/mkldnn_reorder_node.h"
#include "nodes/mkldnn_conv_node.h"
#include "nodes/mkldnn_bin_conv_node.h"
#include "nodes/mkldnn_quantize_node.h"
#include "nodes/mkldnn_mvn_node.h"
#include <nodes/mkldnn_permute_node.h>
#include "nodes/mkldnn_interpolate_node.h"
#include "nodes/mkldnn_input_node.h"

#include "mkldnn/ie_mkldnn.h"

#include <blob_factory.hpp>
#include <legacy/ie_layers_internal.hpp>
#include "utils/general_utils.h"

// WA for xbyak.h
#ifdef _WIN32
# ifndef _WINSOCKAPI_
#  define _WINSOCKAPI_
# endif
# ifndef _WINSOCK2API_
#  define _WINSOCK2API_
#endif
#endif
#include <cpu/x64/cpu_isa_traits.hpp>

#include <string>
#include <list>
#include <memory>
#include <set>
#include <algorithm>

#include "mkldnn_itt.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNGraphOptimizer::MKLDNNGraphOptimizer() {}

void MKLDNNGraphOptimizer::ApplyCommonGraphOptimizations(MKLDNNGraph &graph) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNN_LT, "MKLDNNGraphOptimizer::ApplyCommonGraphOptimizations");

    MergeTwoEqualScaleShifts(graph);
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

    FuseConvolutionAndDepthwise(graph);
    graph.RemoveDroppedNodes();

    FuseConvolutionAndActivation(graph);
    graph.RemoveDroppedNodes();

    FuseConvolutionAndDepthwise(graph);
    graph.RemoveDroppedNodes();

    FuseConvolutionAndQuantize(graph);
    graph.RemoveDroppedNodes();

    graph.SortTopologically();
    graph.RemoveDroppedEdges();

    FuseConvolutionAndDepthwise(graph);
    graph.RemoveDroppedNodes();

    FusePoolingAndQuantize(graph);
    graph.RemoveDroppedNodes();

    graph.SortTopologically();
    graph.RemoveDroppedEdges();

    FuseConvolutionAndDWConvolution(graph);
    graph.RemoveDroppedNodes();

    FuseBinaryConvolutionAndQuantize(graph);
    graph.RemoveDroppedNodes();

    FuseBatchNormWithScale(graph);
    graph.RemoveDroppedNodes();

    RemoveIdentityOperator(graph);
    graph.RemoveDroppedNodes();

    FuseConvolutionSumAndConvolutionSumActivation(graph);
    graph.RemoveDroppedNodes();

    FuseConvolutionAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseFullyConnectedAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseMVNAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseInterpolateAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseNormalizeAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    FuseEltwiseAndSimple(graph);
    graph.RemoveDroppedNodes();

    graph.RemoveDroppedEdges();
}

void MKLDNNGraphOptimizer::ApplyImplSpecificGraphOptimizations(MKLDNNGraph &graph) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNN_LT, "MKLDNNGraphOptimizer::ApplyImplSpecificGraphOptimizations");

    RemoveIOScaleShifts(graph);
    graph.RemoveDroppedNodes();

    DropDoubleReorders(graph);
    graph.RemoveDroppedNodes();

    DropConvertReorder(graph);
    graph.RemoveDroppedNodes();

    MergePermuteAndReorder(graph);
    graph.RemoveDroppedNodes();

    graph.RemoveDroppedEdges();
}

void MKLDNNGraphOptimizer::FuseConvolutionAndZeroPoints(MKLDNNGraph &graph) {
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

    auto initializeInputZeroPoints = [](MKLDNNNodePtr node, MKLDNNNodePtr parent0, MKLDNNNodePtr parent1) {
        auto* convNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
        if (convNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get convolution node " << node->getName();

        int IC = node->getParentEdgesAtPort(0)[0]->getDims()[1];
        int OC = node->getChildEdgesAtPort(0)[0]->getDims()[1];

        if (parent0->getType() == Eltwise) {
            // The plug-in doesn't support FP32 convolution with input/weights zero points.
            // In case weights are in FP32 (or we have zero points on weights which are not supported by INT8 convolution) we cannot use
            // INT8 implementation so we have to disable input zero points fusing as well.
            auto weightsLayer = parent1->getCnnLayer();
            if (!weightsLayer || weightsLayer->type != "Const" || weightsLayer->outData[0]->getPrecision() != Precision::I8) {
                return false;
            }

            auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(parent0.get());
            if (eltwiseNode->getOpType() != Subtract)
                return false;

            if (parent0->getParentEdges().size() != 2)
                return false;

            if (parent0->getParentEdgesAtPort(1)[0]->getParent()->getCnnLayer()->type == "Const") {
                auto arg0 = parent0->getParentEdgesAtPort(1)[0]->getParent();
                if (arg0->getCnnLayer()->outData[0]->getPrecision() != Precision::U8)
                    return false;

                if (parent0->getParentEdgesAtPort(1)[0]->getDims().size() < 2) {
                    return false;
                }

                if (parent0->getParentEdgesAtPort(1)[0]->getDims()[1] != 1 &&
                    parent0->getParentEdgesAtPort(1)[0]->getDims()[1] != IC)
                    return false;

                auto arg1 = parent0->getParentEdgesAtPort(0)[0]->getParent();
                if (arg1->getCnnLayer()->outData[0]->getPrecision() != Precision::U8)
                    return false;

                auto zeroPointsBlob = dynamic_cast<TBlob<uint8_t>*>(arg0->getCnnLayer()->blobs["custom"].get());
                if (zeroPointsBlob == nullptr)
                    THROW_IE_EXCEPTION << "Cannot cast to TBlob internal zero points blob";

                auto zeroPointsData = zeroPointsBlob->buffer().as<uint8_t*>();
                if (zeroPointsData == nullptr)
                    THROW_IE_EXCEPTION << "zeroPointsBlob has not allocated buffer";

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

//    auto initializeWeightsZeroPoints = [](MKLDNNNodePtr node, MKLDNNNodePtr parent0) {
//        auto* convNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
//        if (convNode == nullptr)
//            THROW_IE_EXCEPTION << "Cannot get convolution node " << node->getName();
//
//        int OC = node->getChildEdgesAtPort(0)[0]->getDims()[1];
//
//        if (parent0->getType() == Eltwise) {
//            auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(parent0.get());
//            if (eltwiseNode->getOpType() != Subtract)
//                return false;
//
//            if (parent0->getParentEdges().size() != 2)
//                return false;
//
//            if (parent0->getParentEdgesAtPort(1)[0]->getParent()->getCnnLayer()->type == "Const") {
//                auto arg0 = parent0->getParentEdgesAtPort(1)[0]->getParent();
//                if (arg0->getCnnLayer()->outData[0]->getPrecision() != Precision::I8)
//                    return false;
//
//                if (parent0->getParentEdgesAtPort(1)[0]->getDims()[0] != 1 &&
//                    parent0->getParentEdgesAtPort(1)[0]->getDims()[0] != OC)
//                    return false;
//
//                auto arg1 = parent0->getParentEdgesAtPort(0)[0]->getParent();
//                if (arg1->getCnnLayer()->outData[0]->getPrecision() != Precision::I8)
//                    return false;
//
//                auto zeroPointsBlob = dynamic_cast<TBlob<int8_t>*>(arg0->getCnnLayer()->blobs["custom"].get());
//                if (zeroPointsBlob == nullptr)
//                    THROW_IE_EXCEPTION << "Cannot cast to TBlob internal zero points blob";
//
//                auto zeroPointsData = zeroPointsBlob->buffer().as<int8_t*>();
//                if (zeroPointsData == nullptr)
//                    THROW_IE_EXCEPTION << "zeroPointsBlob has not allocated buffer";
//
//                for (int j = 0; j < parent0->getParentEdgesAtPort(1)[0]->getDims()[0]; j++) {
//                    convNode->weightsZeroPoints.push_back(static_cast<float>(zeroPointsData[j]));
//                }
//            } else {
//                return false;
//            }
//        } else {
//            return false;
//        }
//
//        return true;
//    };

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
        if (weightsBlob == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast to TBlob internal weights blob";

        auto weightsPtr = weightsBlob->buffer().as<int8_t*>();
        if (weightsPtr == nullptr)
            THROW_IE_EXCEPTION << "weightsBlob has not allocated buffer";

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
        auto weightsEltwise = conv->getParentEdgesAtPort(1)[0]->getParent();
        if (initializeInputZeroPoints(conv, dataEltwise, weightsEltwise)) {
            auto p_edge = dataEltwise->getParentEdgesAtPort(1)[0];
            removeEdge(graph, p_edge);

            graph.DropNode(dataEltwise);
        }

// [TODO] Weights zero point is not supported on oneDNN side for the moment
//        auto weightsEltwise = conv->getParentEdgesAtPort(1)[0]->getParent();
//        if (initializeWeightsZeroPoints(conv, weightsEltwise)) {
//            auto p_edge = weightsEltwise->getParentEdgesAtPort(1)[0];
//            removeEdge(graph, p_edge);
//
//            graph.DropNode(weightsEltwise);
//        }

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
        if (node->getType() != Eltwise)
            return false;

        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << node->getName() << " to Eltwise node";

        if (eltwiseNode->getChildEdges().size() != 1)
            return false;

        if (eltwiseNode->getOpType() != MulAdd)
            return false;

        return true;
    };

    auto isEqualScaleShiftNodes = [](MKLDNNNodePtr node1, MKLDNNNodePtr node2) {
        if (node1->getParentEdgeAt(0) != node2->getParentEdgeAt(0))
            return false;

        auto *eltwiseNode1 = dynamic_cast<MKLDNNEltwiseNode *>(node1.get());
        auto *eltwiseNode2 = dynamic_cast<MKLDNNEltwiseNode *>(node2.get());

        auto eltwiseLayer1 = eltwiseNode1->getCnnLayer();
        auto eltwiseLayer2 = eltwiseNode2->getCnnLayer();

        Blob::Ptr scalesBlob1 = eltwiseLayer1->blobs["weights"];
        Blob::Ptr shiftsBlob1 = eltwiseLayer1->blobs["biases"];
        Blob::Ptr scalesBlob2 = eltwiseLayer2->blobs["weights"];
        Blob::Ptr shiftsBlob2 = eltwiseLayer2->blobs["biases"];
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

        auto parentEdges = childNode2->parentEdges;
        for (auto &parentEdge : parentEdges) {
            auto p_edge = parentEdge.lock();
            if (p_edge->getParent() == parentNode)
                continue;

            removeEdge(graph, p_edge);
        }

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
                    if (child->type == Eltwise && child->getCnnLayer()->type == "ScaleShift") {
                        bn->fuseWith(child);

                        auto parentEdges = child->parentEdges;
                        for (auto &parentEdge : parentEdges) {
                            auto p_edge = parentEdge.lock();
                            if (p_edge->getParent()->getType() == BatchNormalization)
                                continue;

                            removeEdge(graph, p_edge);
                        }

                        graph.DropNode(child);
                    }
                }
            }
        }
    }
}

void MKLDNNGraphOptimizer::FuseConvolutionAndActivation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isFusingSupported = [&](MKLDNNNodePtr conv, MKLDNNNodePtr activation) {
        auto* binConv = dynamic_cast<MKLDNNBinaryConvolutionNode *>(conv.get());
        if (binConv) {
            if (!binConv->canFuse(activation))
                return false;
        }

        if (!activation->getCnnLayer())
            return false;

        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(activation.get());

        return eltwiseNode &&
            (eltwiseNode->getOpType() == Relu ||
            (conv->getCnnLayer()->precision == Precision::FP32 &&
            IsOneOf(eltwiseNode->getOpType(), {Elu, Logistic, BoundedRelu, Clamp, Swish, Hswish, Mish, Hsigmoid,
                                               Round})));
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
        } else if (childNode->getType() == Eltwise) {
            auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode*>(childNode.get());
            if (eltwiseNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get Eltwise node " << childNode->getName();

            if (IsOneOf(eltwiseNode->getOpType(), {Relu, Gelu, Elu, Logistic, BoundedRelu, Clamp, Swish, Hswish, Mish,
                                                   Hsigmoid, Round})) {
                return true;
            } else if (IsOneOf(eltwiseNode->getOpType(), {MulAdd, Prelu})) {
                if (eltwiseNode->getOpType() == MulAdd && eltwiseNode->getCnnLayer()->blobs.size() != 2)
                    return false;

                if (parentNode->getParentEdgesAtPort(0)[0]->getDims().ndims() != 3) {
                    return true;
                } else {
                    const auto &eltwiseLayer = eltwiseNode->getCnnLayer();
                    if (eltwiseLayer == nullptr)
                        THROW_IE_EXCEPTION << "Cannot get scale shift layer " << eltwiseNode->getName();

                    if (eltwiseNode->getOpType() != MulAdd)
                        return false;

                    Blob::Ptr scalesBlob = eltwiseLayer->blobs["weights"];
                    if (scalesBlob == nullptr)
                        return false;

                    Blob::Ptr shiftsBlob = eltwiseLayer->blobs["biases"];
                    if (shiftsBlob == nullptr)
                        return false;

                    const float *scalesBufferPtr = scalesBlob->buffer().as<float *>();
                    const float *shiftsBufferPtr = shiftsBlob->buffer().as<float *>();

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
            }
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

        if (childNode->getType() == Quantize || childNode->getType() == Eltwise) {
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

void MKLDNNGraphOptimizer::FuseConvolutionAndDepthwise(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        bool isSutableConv = (node->getType() == Convolution) &&
                             node->getCnnLayer()->precision == Precision::FP32;
        bool isSutableBinConv = node->getType() == BinaryConvolution;
        return (isSutableConv || isSutableBinConv) && node->getChildEdges().size() == 1;
    };

    auto isSutableChildNode = [](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        if (childNode->getType() != Eltwise)
            return false;

        if (!childNode->getCnnLayer())
            return false;

        auto* binConv = dynamic_cast<MKLDNNBinaryConvolutionNode *>(parentNode.get());
        if (binConv) {
            if (!binConv->canFuse(childNode))
                return false;
        }

        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(childNode.get());
        if (eltwiseNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get eltwise node " << childNode->getName();
        return ((eltwiseNode->getOpType() == MulAdd && childNode->getCnnLayer()->blobs.size() == 2) ||
                (eltwiseNode->getOpType() == Prelu));
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto conv = graphNodes[i];
        if (!isSutableParentNode(conv)) continue;

        auto depthwise0 = conv->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(conv, depthwise0)) continue;

        conv->fuseWith(depthwise0);

        if (depthwise0->getChildEdges().size() == 1) {
            auto depthwise1 = depthwise0->getChildEdgeAt(0)->getChild();

            if (isSutableChildNode(conv, depthwise1)) {
                conv->fuseWith(depthwise1);

                auto parents = depthwise1->parentEdges;
                for (size_t j = 0; j < parents.size(); j++) {
                    auto p_edge = parents[j].lock();
                    if (p_edge->getParent()->getType() == Eltwise)
                        continue;

                    removeEdge(graph, p_edge);
                }

                graph.DropNode(depthwise1);
            }
        }

        auto parents = depthwise0->parentEdges;
        for (size_t j = 0; j < parents.size(); j++) {
            auto p_edge = parents[j].lock();
            if (p_edge->getParent()->getType() == Convolution || p_edge->getParent()->getType() == BinaryConvolution)
                continue;

            removeEdge(graph, p_edge);
        }

        graph.DropNode(depthwise0);
    }
}

void MKLDNNGraphOptimizer::FuseConvolutionAndDWConvolution(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isConvolutionNode = [](MKLDNNNodePtr node) {
        return node->getType() == Convolution;
    };

    auto is1x1Convolution = [](ConvolutionLayer* layer) {
        return layer->_kernel[X_AXIS] == 1 && layer->_kernel[Y_AXIS] == 1;
    };

    auto isSutableParentConvolution = [&](MKLDNNNodePtr node) {
        auto *layer = dynamic_cast<ConvolutionLayer *>(node->getCnnLayer().get());
        if (layer == nullptr)
            THROW_IE_EXCEPTION << "Cannot get convolution layer " << node->getName();

        auto* parentConvolutionNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
        if (parentConvolutionNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get convolution node " << node->getName();

        if (!parentConvolutionNode->weightsZeroPoints.empty())
            return false;

        // TODO [oneDNN]: is it still valide constrain on conv to fuse in?
        bool isSupportedParams = layer->_group == 1 &&
                is1x1Convolution(layer) &&  // TODO [oneDNN] : fusing is permitted only with 1x1 convolutions
                everyone_is(1, layer->_stride[X_AXIS], layer->_stride[Y_AXIS]) &&
                one_of(layer->outData[0].get()->getPrecision(), Precision::FP32, Precision::U8) &&
                node->getChildEdgeAt(0)->getDims().ndims() == 4;
        if (!isSupportedParams) return false;

        return node->getChildEdges().size() == 1 && isConvolutionNode(node->getChildEdgeAt(0)->getChild());
    };

    auto isSutableChildConvolution = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        auto* childLayer = dynamic_cast<ConvolutionLayer*>(childNode->getCnnLayer().get());
        if (childLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot get convolution layer " << childNode->getName();

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

        auto* childConvolutionNode = dynamic_cast<MKLDNNConvolutionNode*>(childNode.get());
        if (childConvolutionNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot get convolution node " << childNode->getName();

        if (!childConvolutionNode->inputZeroPoints.empty() || !childConvolutionNode->weightsZeroPoints.empty())
            return false;

        auto allPads = getPaddings(*childLayer);

        bool isSupportedParams = childLayer->_out_depth == childLayer->_group &&
                                 childLayer->_out_depth != 1 &&
                                 everyone_is(3, childLayer->_kernel[X_AXIS], childLayer->_kernel[Y_AXIS]) &&
                                 everyone_is(1, allPads.begin[X_AXIS], allPads.begin[Y_AXIS]) &&
                                 everyone_is(1, allPads.end[X_AXIS], allPads.end[Y_AXIS]) &&
                                 everyone_is(1, childLayer->_dilation[X_AXIS], childLayer->_dilation[Y_AXIS]) &&
                                 childLayer->_stride[X_AXIS] == childLayer->_stride[Y_AXIS] &&
                                 false &&  // TODO [oneDNN]: disabled while not ported
                                 one_of(childLayer->_stride[X_AXIS], 1 /*, 2*/) &&  // TODO [oneDNN]: stride 2 should also be supported
                                 childNode->getChildEdgeAt(0)->getDims().ndims() == 4;

        return isSupportedParams;
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        if (!isConvolutionNode(graphNodes[i])) continue;

        auto parentConvNode = graphNodes[i];
        if (!isSutableParentConvolution(parentConvNode)) continue;

        auto childConvNode = parentConvNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildConvolution(parentConvNode, childConvNode)) continue;

        parentConvNode->fuseWith(childConvNode);

        for (auto node : childConvNode->getFusedWith())
            parentConvNode->fuseWith(node);
        childConvNode->clearFusedWith();

        graph.DropDWConvNode(childConvNode);
    }
}

void MKLDNNGraphOptimizer::FuseConvolutionAndQuantize(MKLDNNGraph &graph) {
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
        } else if (node->getType() == Eltwise) {
            auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
            if (eltwiseNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get eltwise node " << node->getName();

            return ((eltwiseNode->getOpType() == MulAdd && node->getCnnLayer()->blobs.size() == 2) ||
                    (eltwiseNode->getOpType() == Prelu) ||
                    IsOneOf(eltwiseNode->getOpType(), {Relu, Elu, Logistic, BoundedRelu, Clamp, Swish, Hswish, Mish,
                                                       Hsigmoid, Round}));
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

        if (childNode->getType() == Quantize || childNode->getType() == Eltwise) {
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
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        bool isSutableBinConv = node->getType() == BinaryConvolution;
        return isSutableBinConv && node->getChildEdges().size() == 1;
    };

    auto isSutableChildNode = [](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        if (childNode->getType() != Quantize)
            return false;

        auto* binConv = dynamic_cast<MKLDNNBinaryConvolutionNode *>(parentNode.get());
        if (!binConv) {
            return false;
        }

        return binConv->canFuse(childNode);
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSutableParentNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(parent, child)) continue;

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
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        bool isSutablePooling = node->getType() == Pooling;

        if (isSutablePooling) {
            auto *poolingLayer = dynamic_cast<PoolingLayer *>(node->getCnnLayer().get());
            if (poolingLayer == nullptr)
                THROW_IE_EXCEPTION << "Cannot get Pooling layer " << node->getName();

            // Optimized FP32 Pooling doesn't support fusing with FQ
            auto inputPrecision = poolingLayer->insData[0].lock()->getPrecision();
            if (inputPrecision != Precision::U8 && inputPrecision != Precision::I8)
                return false;

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

void MKLDNNGraphOptimizer::FuseConvolutionSumAndConvolutionSumActivation(MKLDNNGraph &graph) {
    std::vector<MKLDNNNodePtr> &graphNodes = graph.GetNodes();

    auto isFusingSupported = [&](MKLDNNNodePtr conv, MKLDNNNodePtr activation) {
        if (!activation->getCnnLayer())
            return false;

        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(activation.get());

        return eltwiseNode &&
            (eltwiseNode->getOpType() == Relu ||
            (conv->getCnnLayer()->precision == Precision::FP32 &&
             IsOneOf(eltwiseNode->getOpType(), {Elu, Logistic, BoundedRelu, Clamp, Swish, Hswish, Mish, Hsigmoid,
                                                Round})));
    };

    for (auto &graphNode : graphNodes) {
        if (graphNode->getType() != Eltwise)
            continue;

        if (!std::dynamic_pointer_cast<MKLDNNEltwiseNode>(graphNode)->isSum()) continue;
        if (std::dynamic_pointer_cast<MKLDNNEltwiseNode>(graphNode)->isWithBroadcast()) continue;

        // TODO: Enlarge to several inputs
        bool isSutableNode = graphNode->getParentEdges().size() == 2;
        if (!isSutableNode)
            continue;

        auto parent1 = graphNode->getParentEdgeAt(0)->getParent();
        auto parent2 = graphNode->getParentEdgeAt(1)->getParent();

        bool isSutableParent1 = parent1->getType() == Convolution || parent1->getType() == BinaryConvolution;
        bool isSutableParent2 = parent2->getType() == Convolution || parent2->getType() == BinaryConvolution;

        auto* binConvNode1 = dynamic_cast<MKLDNNBinaryConvolutionNode *>(parent1.get());
        if (binConvNode1) {
            isSutableParent1 = isSutableParent1 && binConvNode1->canFuse(graphNode);
        }

        auto* binConvNode2 = dynamic_cast<MKLDNNBinaryConvolutionNode *>(parent2.get());
        if (binConvNode2) {
            isSutableParent2 = isSutableParent2 && binConvNode2->canFuse(graphNode);
        }

        auto* convNode1 = dynamic_cast<MKLDNNConvolutionNode *>(parent1.get());
        if (convNode1) {
            if (!convNode1->canBeExecutedInInt8()) {
                isSutableParent1 = isSutableParent1 && convNode1->getFusedWith().empty();
            }
        }

        auto* convNode2 = dynamic_cast<MKLDNNConvolutionNode *>(parent2.get());
        if (convNode2) {
            if (!convNode2->canBeExecutedInInt8()) {
                isSutableParent2 = isSutableParent2 && convNode2->getFusedWith().empty();
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

        auto* mergedBinConvNode = dynamic_cast<MKLDNNBinaryConvolutionNode*>(mergedConv.get());
        if (mergedBinConvNode != nullptr)
            childPort = mergedBinConvNode->getParentEdges().size();

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

void MKLDNNGraphOptimizer::FuseMVNAndSimpleOperation(MKLDNNGraph &graph) {
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
        } else if (node->getType() == Eltwise) {
            auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
            if (eltwiseNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get eltwise node " << node->getName();

            return ((eltwiseNode->getOpType() == MulAdd) ||
                    (eltwiseNode->getOpType() == Prelu) ||
                     eltwiseNode->getOpType() == Relu);
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

        if (childNode->getType() == Quantize || childNode->getType() == Eltwise) {
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

void MKLDNNGraphOptimizer::FuseInterpolateAndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        bool isSuitable = (node->getType() == Interpolate);
        if (isSuitable) {
            return node->getChildEdges().size() == 1;
        } else {
            return false;
        }
    };

    auto isSutableChildNode = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        // Avoid cycle dependencies
        for (auto &childParentEdge : childNode->getParentEdges()) {
            for (auto &parentParentEdge : parentNode->getParentEdges()) {
                if (childParentEdge.lock()->getParent() == parentParentEdge.lock()->getParent())
                    return false;
            }
        }
        if (!childNode->getFusedWith().empty())
            return false;
        auto interpolateNode = dynamic_cast<MKLDNNInterpolateNode*>(parentNode.get());
        return interpolateNode->canFuse(childNode);
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

        parentNode->fuseWith(childNode);

        if (childNode->getType() == Quantize || childNode->getType() == Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Interpolate)
                    continue;

                removeEdge(graph, p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseNormalizeAndSimpleOperation(MKLDNNGraph &graph) {
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
        } else if (node->getType() == Eltwise) {
            auto *eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
            if (eltwiseNode == nullptr)
                THROW_IE_EXCEPTION << "Cannot get Eltwise node " << node->getName();
            return IsOneOf(eltwiseNode->getOpType(), {Relu, Gelu, Elu, Logistic, BoundedRelu, Clamp, Tanh, Swish,
                                                      Hswish, Mish, Hsigmoid, Round, Linear, Abs, Square, Sqrt}) ||
                    ((eltwiseNode->getOpType() == MulAdd && eltwiseNode->getCnnLayer()->blobs.size() == 2) ||
                     (eltwiseNode->getOpType() == Prelu));
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

        if (childNode->getType() == Quantize || childNode->getType() == Eltwise) {
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
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Eltwise && node->getChildEdges().size() == 1;
    };

    auto isSutableChildNode = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        for (auto &childParentEdge : childNode->getParentEdges()) {
            // WA to prevent unsupported reorder exception issue in some cases
            if (childParentEdge.lock()->getParent()->getType() == Split) {
                return false;
            }

            // Avoid cycle dependencies
            for (auto &parentParentEdge : parentNode->getParentEdges()) {
                if (childParentEdge.lock()->getParent() == parentParentEdge.lock()->getParent())
                    return false;
            }
        }

        if (!childNode->getFusedWith().empty())
            return false;

        auto eltwiseNode = dynamic_cast<MKLDNNEltwiseNode*>(parentNode.get());
        return eltwiseNode->canFuse(childNode);
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
                if (p_edge->getParent()->getType() == Eltwise)
                    continue;

                removeEdge(graph, p_edge);
            }

            graph.DropNode(childNode);
        } else if (childNode->getType() == Eltwise) {
            auto childs = childNode->childEdges;
            auto parents = childNode->parentEdges;

            for (size_t i = 0; i < parents.size(); i++) {
                auto p_edge = parents[i].lock();
                if (!p_edge) continue;
                auto parent = p_edge->getParent();
                if (!parent) continue;

                if (parent == parentNode) {
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
                            removeEdge(graph, remEdge);
                        }
                        remEdge = childs[j].lock();
                        int outNum = 0;
                        if (remEdge) {
                            outNum = remEdge->getOutputNum();
                            remEdge->drop();
                            removeEdge(graph, remEdge);
                        }
                        MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, child, inNum, outNum));
                        auto &graphEdges = graph.GetEdges();
                        graphEdges.push_back(newEdge);
                        parent->addEdge(newEdge);

                        parent->outDims[inNum] = child->inDims[outNum];
                    }
                } else {
                    MKLDNNEdgePtr &remEdge = p_edge;
                    int inNum = 0;
                    if (remEdge) {
                        inNum = remEdge->getInputNum();
                        remEdge->drop();
                        removeEdge(graph, remEdge);
                    }

                    auto parentEltwise = parentNode;
                    MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, parentEltwise, inNum, parentEltwise->getParentEdges().size()));
                    auto &graphEdges = graph.GetEdges();
                    graphEdges.push_back(newEdge);
                    parent->addEdge(newEdge);

                    parentEltwise->inDims.push_back(parent->outDims[0]);
                }
            }

            graph.DropNode(childNode);
        } else {
            graph.DropNode(childNode);
        }
    }
}

void MKLDNNGraphOptimizer::RemoveIdentityOperator(MKLDNNGraph &graph) {
    for (MKLDNNNodePtr& node : graph.GetNodes()) {
        bool toDrop = false;

        if (node->getType() == Eltwise) {
            auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode*>(node.get());
            if (eltwiseNode->getOpType() == PowerStatic) {
                PowerLayer *l = dynamic_cast<PowerLayer *>(node->getCnnLayer().get());
                if (l == nullptr)
                    THROW_IE_EXCEPTION << "Cannot get power layer " << node->getName();

                if (l->power == 1.0f && l->scale == 1.0f && l->offset == 0.0f) toDrop = true;
            }
        }

        if (node->getType() == Eltwise && node->getCnnLayer()->type == "ScaleShift") {
            ScaleShiftLayer* l = dynamic_cast<ScaleShiftLayer*>(node->getCnnLayer().get());
            if (l == nullptr)
                THROW_IE_EXCEPTION << "Cannot get scale shift layer " << node->getName();

            if (l->_weights == nullptr && l->_biases == nullptr) toDrop = true;
        }

        if (node->getType() == Copy) toDrop = true;

        if (toDrop) graph.DropNode(node);
    }
}

void MKLDNNGraphOptimizer::DropDoubleReorders(MKLDNNGraph &graph) {
    std::set<MKLDNNNodePtr> processed;
    int graphNodesSize = graph.GetNodes().size();
    for (int i = 0; i < graphNodesSize; i++) {
        MKLDNNNodePtr& node = graph.GetNodes()[i];
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
            graph.InsertReorder(edge, layerName, n->getInput(), nn->getOutput(), false, scales);
            graph.GetEdges().erase(std::remove(graph.GetEdges().begin(), graph.GetEdges().end(), edge), graph.GetEdges().end());
        }
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

void MKLDNNGraphOptimizer::RemoveIOScaleShifts(MKLDNNGraph &graph) {
    for (MKLDNNNodePtr& node : graph.GetNodes()) {
        if (node->getType() == Eltwise && node->getCnnLayer()->type == "ScaleShift") {
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
                    if (child->type == Reorder) {
                        MKLDNNReorderNode* rn = dynamic_cast<MKLDNNReorderNode*>(child.get());
                        if (rn != nullptr) {
                            rn->_scales = l->_weights;
                            graph.DropNode(node);
                        }
                    } else {
                        THROW_IE_EXCEPTION << "Strange case. No Reorder after iScaleShift";
                    }
                } else if (node->name.find("_oScaleShift_") != std::string::npos) {
                    auto parent = node->parentEdges[0].lock()->getParent();

                    if (parent->type == Reorder) {
                        MKLDNNReorderNode* rn = dynamic_cast<MKLDNNReorderNode*>(parent.get());
                        if (rn != nullptr) {
                            rn->_scales = l->_weights;
                            graph.DropNode(node);
                        }
                    } else {
                        THROW_IE_EXCEPTION << "Strange case. No Reorder before oScaleShift";
                    }
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

bool MKLDNNGraphOptimizer::IsOneOf(EltwiseOpType alg, std::vector<EltwiseOpType> algs) {
    for (auto a : algs) {
        if (alg == a) {
            return true;
        }
    }
    return false;
}

void MKLDNNGraphOptimizer::removeEdge(MKLDNNGraph &graph, MKLDNNEdgePtr& edge) {
    auto& edges = graph.GetEdges();
    for (auto it = edges.begin(); it != edges.end(); it++) {
        if ((*it) == edge) {
            edges.erase(it);
            return;
        }
    }
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
        if (node->getType() != Eltwise)
            return false;

        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << node->getName() << " to Eltwise node";

        if (eltwiseNode->getChildEdges().size() != 1)
            return false;

        if (eltwiseNode->getOpType() != Clamp)
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
        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(parent.get());
        if (eltwiseNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << parent->getName() << " to Eltwise node";

        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(child.get());
        if (quantizeNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << child->getName() << " to Quantize node";

        const std::vector<float>& cropLowData = quantizeNode->getCropLow();
        const std::vector<float>& cropHighData = quantizeNode->getCropHigh();

        std::vector<float> newCropLow(cropLowData.size());
        std::vector<float> newCropHigh(cropHighData.size());
        for (int i = 0; i < cropLowData.size(); i++)
            newCropLow[i] = std::max(cropLowData[i], eltwiseNode->getAlpha());
        for (int i = 0; i < cropHighData.size(); i++)
            newCropHigh[i] = std::min(cropHighData[i], eltwiseNode->getBeta());

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
        if (node->getType() != Eltwise)
            return false;

        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << node->getName() << " to eltwise node";

        if (eltwiseNode->getChildEdges().size() != 1)
            return false;

        if (eltwiseNode->getOpType() != MulAdd)
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
        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(parent.get());
        if (eltwiseNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << parent->getName() << " to eltwise node";

        auto eltwiseLayer = eltwiseNode->getCnnLayer();
        if (eltwiseLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot get scale shift layer " << eltwiseNode->getName();

        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode*>(child.get());
        if (quantizeNode == nullptr)
            THROW_IE_EXCEPTION << "Cannot cast " << child->getName() << " to Quantize node";

        Blob::Ptr scalesBlob = eltwiseLayer->blobs["weights"];
        if (scalesBlob == nullptr)
            return false;

        Blob::Ptr shiftsBlob = eltwiseLayer->blobs["biases"];
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
            auto parentEdges = parent->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getCnnLayer()->type != "Const")
                    continue;

                removeEdge(graph, p_edge);
            }

            graph.DropNode(parent);
        }
    }
}

void MKLDNNGraphOptimizer::MergePermuteAndReorder(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Permute && node->getChildEdges().size() == 1;
    };

    auto isSutableChildNode = [](MKLDNNNodePtr node) {
        return node->getType() == Reorder && node->getChildEdges().size() == 1;
    };

    // Method checkAscendingSummaryOrder() checks that after the sequential execution of Permute and Reorder nodes,
    // the order of the elements in the memory will not change. In other words, that Permute+Reorder is identical permutation.
    auto checkAscendingSummaryOrder = [](std::shared_ptr<MKLDNNNode> &parentNode, std::shared_ptr<MKLDNNNode> &childNode) -> bool {
        auto* permuteNode = dynamic_cast<MKLDNNPermuteNode*>(parentNode.get());
        auto* reorderNode = dynamic_cast<MKLDNNReorderNode*>(childNode.get());
        if (!permuteNode || !reorderNode) {
            return false;
        }

        auto& permuteOrder = permuteNode->getOrder();
        auto& layoutOrder = permuteNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc.getBlockingDesc().getOrder();
        auto& inOrder = reorderNode->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getBlockingDesc().getOrder();
        auto& outOrder = reorderNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc.getBlockingDesc().getOrder();

        if (permuteOrder.size() != layoutOrder.size() || layoutOrder.size() != inOrder.size() || inOrder.size() != outOrder.size()) {
            return false;
        }

        // revLayoutOrder - reverse permutation for layoutOrder
        auto revLayoutOrder = SizeVector(layoutOrder.size());
        for (int i = 0; i < revLayoutOrder.size(); i++) {
            revLayoutOrder[layoutOrder[i]] = i;
        }

        // newPermuteOrder - Permute layout-aware permutation
        auto newPermuteOrder = SizeVector(permuteOrder.size());
        for (int i = 0; i < newPermuteOrder.size(); i++) {
            newPermuteOrder[i] = layoutOrder[permuteOrder[revLayoutOrder[i]]];
        }

        // reorderOrder - Reorder layout-aware permutation
        auto reorderOrder = SizeVector(outOrder.size());
        for (int i = 0; i < reorderOrder.size(); i++) {
            for (int j = 0; j < reorderOrder.size(); j++) {
                if (outOrder[i] == inOrder[j]) {
                    reorderOrder[i] = j;
                    continue;
                }
            }
        }

        // summaryOrder - resulting Permute+Reorder permutation
        auto summaryOrder = SizeVector(permuteOrder.size());
        for (int i = 0; i < summaryOrder.size(); i++) {
            summaryOrder[i] = reorderOrder[newPermuteOrder[i]];
        }

        // check that Permute+Reorder is the identical permutation
        for (int i = 0; i < summaryOrder.size(); i++) {
            if (summaryOrder[i] != i) {
                return false;
            }
        }

        return true;
    };

    // Permute and Reorder do opposite permutation to each other.
    // Example:
    //      chain [physical layout: NCHW, logical layout: NCHW] -> Permute(order=0312) -> [physical layout: NWCH, logical layout: NCHW] ->
    //      Reorder(nchw->nhwc) -> [physical layout: NCHW, logical layout: NHWC] can be replaced with Reorder(nchw->nhwc; isOptimized=true)
    //      which will just reinterprets layout without physical change of the memory.
    // Two cases are possible:
    //      1) inPrec = outPrec
    //          In this case, we replace Permute+Reorder pattern with a new Reorder that does nothing.
    //      2) inPrec != outPrec
    //          As in the first case, we also replace Permute+Reorder pattern with a new Reorder.
    //          Additionally, we insert another Reorder that performs the conversion from the input precision (inPrec)
    //          to the output precision (outPrec)
    auto mergePermuteAndReorder = [&](std::shared_ptr<MKLDNNNode>& parentNode, std::shared_ptr<MKLDNNNode>& childNode) {
        auto parentParentNode = parentNode->getParentEdgeAt(0)->getParent();
        auto childChildNode = childNode->getChildEdgeAt(0)->getChild();

        graph.DropNode(parentNode);
        graph.DropNode(childNode);

        auto inDesc = parentNode->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc;
        auto outDesc = childNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc;

        auto inPrec = inDesc.getPrecision();
        auto outPrec = outDesc.getPrecision();

        auto reorderInDesc = TensorDesc(inDesc);
        auto reorderOutDesc = TensorDesc(outDesc);
        reorderOutDesc.setPrecision(inPrec);

        std::string reorderlayerName = parentParentNode->getName() + "_" +
                MKLDNNExtensionUtils::getReorderArgs(reorderInDesc, reorderOutDesc) + "_" + "fake";

        MKLDNNEdgePtr edge;
        for (auto &childEdge : parentParentNode->getChildEdges()) {
            if (childEdge.lock()->getChild() == childChildNode) {
                edge = childEdge.lock();
                break;
            }
        }

        auto reorderNode = graph.InsertReorder(edge, reorderlayerName, reorderInDesc, reorderOutDesc, true);

        // case 2
        if (inPrec != outPrec) {
            auto reorderInDesc2 = TensorDesc(reorderOutDesc);
            auto reorderOutDesc2 = TensorDesc(outDesc);

            std::string reorderLayerName2 = reorderNode->getName() + "_" +
                                    MKLDNNExtensionUtils::getReorderArgs(reorderInDesc2, reorderOutDesc2) + "_" + childChildNode->getName();

            graph.InsertReorder(reorderNode->getChildEdgeAt(0), reorderLayerName2, reorderInDesc2, reorderOutDesc2, false);
        }
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parentNode = graphNodes[i];
        if (!isSutableParentNode(parentNode)) {
            continue;
        }
        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(childNode)) {
            continue;
        }

        if (checkAscendingSummaryOrder(parentNode, childNode)) {
            mergePermuteAndReorder(parentNode, childNode);
        }
    }
}