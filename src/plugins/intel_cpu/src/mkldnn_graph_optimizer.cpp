// Copyright (C) 2018-2022 Intel Corporation
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
#include "nodes/mkldnn_deconv_node.h"
#include "nodes/mkldnn_bin_conv_node.h"
#include "nodes/mkldnn_fake_quantize_node.h"
#include "nodes/mkldnn_mvn_node.h"
#include <nodes/mkldnn_transpose_node.h>
#include "nodes/mkldnn_interpolate_node.h"
#include "nodes/mkldnn_reduce_node.h"
#include "nodes/mkldnn_input_node.h"
#include "nodes/mkldnn_rnn.h"
#include "nodes/common/cpu_convert.h"

#include "mkldnn/ie_mkldnn.h"

#include <blob_factory.hpp>
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ie_ngraph_utils.hpp>

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
#include "memory_desc/cpu_memory_desc_utils.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNGraphOptimizer::MKLDNNGraphOptimizer() {}

void MKLDNNGraphOptimizer::ApplyCommonGraphOptimizations(MKLDNNGraph &graph) {
    OV_ITT_SCOPE_CHAIN(FIRST_INFERENCE, taskChain, itt::domains::MKLDNN_LT, "ApplyCommonGraphOptimizations", "FuseConvolutionAndBias");
    FuseConvolutionMatMulAndBias(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseMultiplyAndAdd");
    FuseMultiplyAndAdd(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseDeconvolutionAndSimpleOperation");
    FuseDeconvolutionAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseBroadcastAndEltwise");
    FuseBroadcastAndEltwise(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseClampAndFakeQuantize");
    FuseClampAndFakeQuantize(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FusePerformedAsScaleShiftAndFakeQuantize");
    FusePerformedAsScaleShiftAndFakeQuantize(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseConvolutionAndZeroPoints");
    FuseConvolutionAndZeroPoints(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseConvolutionAndSimpleOperationThroughMaxPool");
    FuseConvolutionAndSimpleOperationThroughMaxPool(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseConvolutionAndSimpleOperation");
    FuseConvolutionAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "RemoveDroppedEdges");
    graph.SortTopologically();
    graph.RemoveDroppedEdges();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FusePoolingAndFakeQuantize");
    FusePoolingAndFakeQuantize(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "RemoveDroppedEdges");
    graph.SortTopologically();
    graph.RemoveDroppedEdges();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseConvolutionAndDWConvolution");
    FuseConvolutionAndDWConvolution(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseConvolutionSumAndConvolutionSumActivation");
    FuseConvolutionSumAndConvolutionSumActivation(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseConvolutionAndSimpleOperation");
    FuseConvolutionAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseFullyConnectedAndSimpleOperation");
    FuseFullyConnectedAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseMatMulAndSimpleOperation");
    FuseMatMulAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseMVNAndSimpleOperation");
    FuseMVNAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseInterpolateAndSimpleOperation");
    FuseInterpolateAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseNormalizeL2AndSimpleOperation");
    FuseNormalizeL2AndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseReduceAndSimpleOperation");
    FuseReduceAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseEltwiseAndSimple");
    FuseEltwiseAndSimple(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "reshapeRnnSeq");
    reshapeRnnSeq(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "RemoveDroppedEdges");
    graph.RemoveDroppedEdges();
}

void MKLDNNGraphOptimizer::ApplyImplSpecificGraphOptimizations(MKLDNNGraph &graph) {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::MKLDNN_LT, "MKLDNNGraphOptimizer::ApplyImplSpecificGraphOptimizations");

    DropDoubleReorders(graph);
    graph.RemoveDroppedNodes();

    MergeTransposeAndReorder(graph);
    graph.RemoveDroppedNodes();

    graph.RemoveDroppedEdges();
}

void MKLDNNGraphOptimizer::FuseConvolutionMatMulAndBias(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const MKLDNNNodePtr& node) {
        return (node->getType() == Convolution || node->getType() == MatMul) &&
               node->getChildEdges().size() == 1 &&
               node->getParentEdges().size() == 2 &&
               node->getFusedWith().empty();
    };

    auto isSuitableChildNode = [&](const MKLDNNNodePtr& parentNode, const MKLDNNNodePtr& childNode) {
        if (childNode->getAlgorithm() != EltwiseAdd || !childNode->getFusedWith().empty() || childNode->getParentEdges().size() != 2)
            return false;

        const auto biasNode = childNode->getParentEdgesAtPort(1)[0]->getParent();
        if (biasNode->getType() != Input || !biasNode->isConstant() || biasNode->getChildEdges().size() != 1)
            return false;

        const auto parentOutDims = parentNode->getOutputShapeAtPort(0).getDims();
        const auto biasDims = getNormalizedDimsBySize(biasNode->getOutputShapeAtPort(0).getDims(),
                                                parentOutDims.size());
        // TODO [NM]: Legacy ConvBias fusion transformation supports both per-tensor (via explicit broadcasing) and per-channel cases.
        // Most of the real models contain per-channel bias, so we need to reavaluate the need to support per-tensor variant.
        if (parentOutDims.size() != biasDims.size() || biasDims.size() < 2)
            return false;

        const auto channelAxis = parentNode->getFusingAxis();
        if (!dimsEqualStrong(biasDims[channelAxis], parentOutDims[channelAxis]))
            return false;

        for (int i = 0; i < biasDims.size(); i++) {
            if (biasDims[i] != 1 && i != channelAxis)
                return false;
        }

        return true;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

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
                        graph.RemoveEdge(remEdge);
                    }
                    remEdge = childs[j].lock();
                    int outNum = 0;
                    if (remEdge) {
                        outNum = remEdge->getOutputNum();
                        remEdge->drop();
                        graph.RemoveEdge(remEdge);
                    }
                    MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, child, inNum, outNum));
                    auto &graphEdges = graph.GetEdges();
                    graphEdges.push_back(newEdge);
                    parent->addEdge(newEdge);
                }
            } else {
                MKLDNNEdgePtr &remEdge = p_edge;
                int inNum = 0;
                if (remEdge) {
                    inNum = remEdge->getInputNum();
                    remEdge->drop();
                    graph.RemoveEdge(remEdge);
                }

                const auto& parentEltwise = parentNode;
                MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, parentEltwise, inNum, parentEltwise->getParentEdges().size()));
                auto& graphEdges = graph.GetEdges();
                graphEdges.push_back(newEdge);
                parent->addEdge(newEdge);

                const auto fusingAxis = parentEltwise->getFusingAxis();
                const auto& outShape = parentEltwise->getOutputShapeAtPort(0);

                parent->outputShapes[inNum] = Shape({outShape.getMinDims()[fusingAxis]}, {outShape.getMaxDims()[fusingAxis]});
                parentEltwise->inputShapes.push_back(parent->getOutputShapeAtPort(0));
            }
        }

        graph.DropNode(childNode);
        parentNode->addOriginalLayer(childNode->getOriginalLayers());
        parentNode->addOriginalInputPrecision(childNode->getOriginalInputPrecisionAtPort(1));
    }
}

void MKLDNNGraphOptimizer::FuseDeconvolutionAndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        if (node->getType() != Deconvolution || node->getChildEdges().size() != 1)
            return false;
        const auto deconv = std::dynamic_pointer_cast<MKLDNNDeconvolutionNode>(node);
        if (deconv == nullptr)
            IE_THROW() << "Cannot cast to deconvolution node " << node->getName();

        if (deconv->getAlgorithm() != DeconvolutionCommon) {
            return true;
        }

        const auto& strides = deconv->getStride();
        const auto& kernel = deconv->getWeightDims();
        // WA oneDNN doesn't support fusing post ops after deconvolution with strides over kernel size
        bool isSupportedParams = strides[strides.size() - 1] <= kernel[kernel.size() - 1];
        if (strides.size() > 1)
            isSupportedParams &= strides[strides.size() - 2] <= kernel[kernel.size() - 2];
        if (strides.size() > 2)
            isSupportedParams &= strides[strides.size() - 3] <= kernel[kernel.size() - 3];
        return isSupportedParams;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        auto parentEdges = childNode->parentEdges;
        for (auto &parentEdge : parentEdges) {
            auto p_edge = parentEdge.lock();
            if (p_edge->getParent()->getType() == Deconvolution)
                continue;

            graph.RemoveEdge(p_edge);
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseMultiplyAndAdd(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableSecondInput = [](MKLDNNNodePtr node, VectorDims dataDims) {
        if (node->getType() != Input || !node->isConstant())
            return false;
        auto secondInputDims = node->getOutputShapeAtPort(0).getStaticDims();
        if (secondInputDims.size() != dataDims.size() || secondInputDims.size() < 2)
            return false;

        if (secondInputDims[0] != 1 || !dimsEqualWeak(secondInputDims[1], dataDims[1]))
            return false;

        for (size_t i = 2; i < secondInputDims.size(); i++) {
            if (secondInputDims[i] != 1)
                return false;
        }

        return true;
    };

    auto isSuitableParentNode = [&](MKLDNNNodePtr node) {
        if (node->getAlgorithm() != EltwiseMultiply || !node->getFusedWith().empty() ||
            node->getParentEdges().size() != 2 || node->getChildEdges().size() != 1)
            return false;

        return isSuitableSecondInput(node->getParentEdgesAtPort(1)[0]->getParent(), node->getInputShapeAtPort(0).getDims());
    };

    auto isSuitableChildNode = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        if (childNode->getAlgorithm() != EltwiseAdd || !childNode->getFusedWith().empty() || childNode->getParentEdges().size() != 2)
            return false;

        return isSuitableSecondInput(childNode->getParentEdgesAtPort(1)[0]->getParent(), childNode->getInputShapeAtPort(0).getDims()) &&
                                     parentNode->canFuse(childNode);
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

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
                        graph.RemoveEdge(remEdge);
                    }
                    remEdge = childs[j].lock();
                    int outNum = 0;
                    if (remEdge) {
                        outNum = remEdge->getOutputNum();
                        remEdge->drop();
                        graph.RemoveEdge(remEdge);
                    }
                    MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, child, inNum, outNum));
                    auto &graphEdges = graph.GetEdges();
                    graphEdges.push_back(newEdge);
                    parent->addEdge(newEdge);
                }
            } else {
                MKLDNNEdgePtr &remEdge = p_edge;
                int inNum = 0;
                if (remEdge) {
                    inNum = remEdge->getInputNum();
                    remEdge->drop();
                    graph.RemoveEdge(remEdge);
                }

                auto parentEltwise = parentNode;
                MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, parentEltwise, inNum, parentEltwise->getParentEdges().size()));
                auto &graphEdges = graph.GetEdges();
                graphEdges.push_back(newEdge);
                parent->addEdge(newEdge);

                parentEltwise->inputShapes.push_back(parent->getOutputShapeAtPort(0));
            }
        }

        parentNode->addOriginalInputPrecision(childNode->getOriginalInputPrecisionAtPort(1));
        parentNode->setAlgorithm(EltwiseMulAdd);
        parentNode->setTypeStr("MulAdd");
        parentNode->addOriginalLayer(childNode->getOriginalLayers());
        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseConvolutionAndZeroPoints(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableConvNode = [](MKLDNNNodePtr node) {
        bool retVal = false;
        if (node->getType() == Convolution) {
            if (auto convNode = std::dynamic_pointer_cast<MKLDNNConvolutionNode>(node)) {
                auto rank = convNode->getInputShapeAtPort(0).getRank();
                // int8 depthwise convolution does not support fusing zero points in 3D case
                if (implication(convNode->isDepthWise(), rank < 5)) {
                    retVal = true;
                }
            }
        }
        return retVal;
    };

    auto initializeInputZeroPoints = [](MKLDNNNodePtr node, MKLDNNNodePtr parent0, MKLDNNNodePtr parent1) {
        auto* convNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
        if (convNode == nullptr)
            IE_THROW() << "Cannot get convolution node " << node->getName();

        int IC = node->getInputShapeAtPort(0).getDims()[1];
        int OC = node->getOutputShapeAtPort(0).getDims()[1];

        if (Shape::UNDEFINED_DIM == IC || Shape::UNDEFINED_DIM == OC) {
            return false;
        }

        if (parent0->getType() == Eltwise) {
            if (!parent0->getFusedWith().empty() || !parent1->getFusedWith().empty())
                return false;

            // The plug-in doesn't support FP32 convolution with input/weights zero points.
            // In case weights are in FP32 (or we have zero points on weights which are not supported by INT8 convolution) we cannot use
            // INT8 implementation so we have to disable input zero points fusing as well.
            if (parent1->getType() != Input || !parent1->isConstant() || parent1->getOriginalOutputPrecisionAtPort(0) != Precision::I8) {
                return false;
            }

            if (parent0->getAlgorithm() != Algorithm::EltwiseSubtract)
                return false;

            if (parent0->getParentEdges().size() != 2)
                return false;

            auto arg0 = parent0->getParentEdgesAtPort(1)[0]->getParent();
            if (arg0->getType() == Input && arg0->isConstant()) {
                if (arg0->getOriginalOutputPrecisionAtPort(0) != Precision::U8)
                    return false;

                if (parent0->getInputShapeAtPort(1).getRank() < 2) {
                    return false;
                }

                auto zpDims = parent0->getInputShapeAtPort(1).getDims();
                if (zpDims[0] != 1 || !dimsEqualStrong(zpDims[1], IC))
                    return false;

                for (int i = 2; i < zpDims.size(); i++) {
                    if (zpDims[i] != 1)
                        return false;
                }

                auto arg1 = parent0->getParentEdgesAtPort(0)[0]->getParent();
                if (arg1->getOriginalOutputPrecisionAtPort(0) != Precision::U8)
                    return false;

                auto zeroPointsConstant = dynamic_cast<MKLDNNInputNode*>(arg0.get());
                if (zeroPointsConstant == nullptr)
                    IE_THROW() << "Cannot cast to Input node";

                auto zeroPointsBlob = zeroPointsConstant->getMemoryPtr();
                if (zeroPointsBlob == nullptr)
                    IE_THROW() << "Cannot cast to TBlob internal zero points blob";

                auto zeroPointsData = static_cast<const uint8_t*>(zeroPointsBlob->GetPtr());
                if (zeroPointsData == nullptr)
                    IE_THROW() << "zeroPointsBlob has not allocated buffer";

                auto zeroPointDataSize =  parent0->getInputShapeAtPort(1).getDims()[1];
                if (Shape::UNDEFINED_DIM == zeroPointDataSize) {
                    return false;
                }

                for (int j = 0; j < zeroPointDataSize; j++) {
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

    auto initializeOutputCompensation = [](MKLDNNNodePtr node) {
        auto* convNode = dynamic_cast<MKLDNNConvolutionNode*>(node.get());
        if (convNode == nullptr)
            IE_THROW() << "Cannot get convolution node " << node->getName();

        if (convNode->inputZeroPoints.empty())
            return;

        auto weightsConstant = dynamic_cast<MKLDNNInputNode*>(convNode->getParentEdgesAtPort(1)[0]->getParent().get());
        if (!weightsConstant || !weightsConstant->isConstant())
            return;

        auto weightsBlob = weightsConstant->getMemoryPtr();
        if (weightsBlob == nullptr)
            IE_THROW() << "Cannot cast to TBlob internal weights blob";

        auto weightsPtr = static_cast<const int8_t*>(weightsBlob->GetPtr());
        if (weightsPtr == nullptr)
            IE_THROW() << "weightsBlob has not allocated buffer";

        ptrdiff_t G = convNode->getGroupNum();
        const int groupOffset = convNode->getAlgorithm() == ConvolutionGrouped ? 1 : 0;
        auto& weightsConstantDims = weightsConstant->outputShapes[0].getStaticDims();

        ptrdiff_t OC = weightsConstantDims[0 + groupOffset];
        ptrdiff_t IC = weightsConstantDims[1 + groupOffset];
        ptrdiff_t KD = weightsConstantDims.size() == (5 + groupOffset) ? weightsConstantDims[weightsConstantDims.size() - 3] : 1;
        ptrdiff_t KH = weightsConstantDims.size() == (3 + groupOffset) ? 1 : weightsConstantDims[weightsConstantDims.size() - 2];
        ptrdiff_t KW = weightsConstantDims[weightsConstantDims.size() - 1];

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
        if (!isSuitableConvNode(conv)) continue;

        auto dataEltwise = conv->getParentEdgesAtPort(0)[0]->getParent();
        auto weightsEltwise = conv->getParentEdgesAtPort(1)[0]->getParent();
        if (initializeInputZeroPoints(conv, dataEltwise, weightsEltwise)) {
            auto p_edge = dataEltwise->getParentEdgesAtPort(1)[0];
            graph.RemoveEdge(p_edge);

            graph.DropNode(dataEltwise);
        }

        initializeOutputCompensation(conv);
    }
}

/**
 * @todo FQ fusing was disabled for BF16 output since oneDNN primitives lack support
 *       for bf16 depthwise postops.
 *       This is not the case anymore, because after migration to oneDNN 2.3 FQ will be fused as
 *       multiple binary post ops.
 *       This check can already be removed for FC fusing, but should be kept for Convolution,
 *       which still uses legacy depthwise postops for performance reasons.
 */
static bool BF16QuantizeNodeFusing(const MKLDNNNodePtr& parentNode, const MKLDNNNodePtr& childNode) {
    return childNode->getType() == FakeQuantize &&
        one_of(Precision::BF16,
            parentNode->getOriginalOutputPrecisionAtPort(0),
            childNode->getOriginalOutputPrecisionAtPort(0));
}

void MKLDNNGraphOptimizer::FuseFullyConnectedAndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == FullyConnected && node->getChildEdges().size() == 1;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        //  BF16 Quantize Layer Fusing Disabling
        if (BF16QuantizeNodeFusing(parentNode, childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize || childNode->getType() == Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == FullyConnected)
                    continue;

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseMatMulAndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](const MKLDNNNodePtr& node) {
        return node->getType() == MatMul && node->getChildEdges().size() == 1;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize || childNode->getType() == Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == MatMul)
                    continue;

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseConvolutionAndDWConvolution(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isConvolutionNode = [](const MKLDNNNodePtr &node) {
        return node->getType() == Convolution;
    };

    auto is1x1Convolution = [](const std::shared_ptr<MKLDNNConvolutionNode> &conv) {
        const auto weightRank = conv->getWeightDims().size();
        return conv->getWeightDims()[weightRank - 1] == 1 && conv->getWeightDims()[weightRank - 2] == 1;
    };

    auto isSuitableParentConvolution = [&](MKLDNNNodePtr node) {
        if (node->isDropped())
            return false;

        if (node->isDynamicNode())
            return false;

        const auto conv = std::dynamic_pointer_cast<MKLDNNConvolutionNode>(node);
        if (conv == nullptr)
            IE_THROW() << "Cannot cast to convolution node " << node->getName();

        if (!conv->weightsZeroPoints.empty())
            return false;

        const auto &strides = conv->getStride();
        const auto &paddings = conv->getPaddingL();
        const auto &inDims = node->getInputShapeAtPort(0).getDims();
        const auto &outDims = node->getOutputShapeAtPort(0).getDims();
        bool isSupportedParams = conv->getGroupNum() == 1 &&
                inDims.size() == 4 &&
                dimsEqualStrong(inDims[inDims.size() - 1], outDims[outDims.size() - 1]) &&
                dimsEqualStrong(inDims[inDims.size() - 2], outDims[outDims.size() - 2]) &&
                is1x1Convolution(conv) &&  // TODO [oneDNN] : fusing is permitted only with 1x1 convolutions
                everyone_is(1, strides[strides.size() - 1], strides[strides.size() - 2]) &&
                everyone_is(0, paddings[paddings.size() - 1], paddings[paddings.size() - 2]) &&
                !conv->canBeExecutedInInt8();
        if (!isSupportedParams) return false;

        return node->getChildEdges().size() == 1 && isConvolutionNode(node->getChildEdgeAt(0)->getChild());
    };

    auto isSuitableChildConvolution = [&](const MKLDNNNodePtr &parentNode, const MKLDNNNodePtr &childNode) {
        if (parentNode->isDropped() || childNode->isDropped())
            return false;

        if (childNode->isDynamicNode())
            return false;

        const auto convChild = std::dynamic_pointer_cast<MKLDNNConvolutionNode>(childNode);
        if (convChild == nullptr)
            IE_THROW() << "Cannot cast to convolution node " << childNode->getName();

        const auto convParent = std::dynamic_pointer_cast<MKLDNNConvolutionNode>(parentNode);
        if (convParent == nullptr)
            IE_THROW() << "Cannot cast to convolution node " << parentNode->getName();

        if (!everyone_is(Precision::FP32, convParent->getOriginalOutputPrecisionAtPort(0), convChild->getOriginalInputPrecisionAtPort(0),
                convChild->getOriginalOutputPrecisionAtPort(0)))
            return false;

        auto parentOutputPrecision = !parentNode->fusedWith.empty()
                ? parentNode->fusedWith[parentNode->fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0)
                : parentNode->getOriginalOutputPrecisionAtPort(0);

        auto childOutputPrecision = !childNode->fusedWith.empty()
                ? childNode->fusedWith[childNode->fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0)
                : childNode->getOriginalOutputPrecisionAtPort(0);

        if (!everyone_is(Precision::FP32, parentOutputPrecision, childOutputPrecision))
            return false;

        if (!convChild->inputZeroPoints.empty() || !convChild->weightsZeroPoints.empty())
            return false;

        bool withBias = convChild->getOriginalInputPrecisions().size() == 3;

        const auto weightRank = convChild->getWeightDims().size();
        const auto stridesSize = convChild->getStride().size();
        bool isSupportedParams = dimsEqualStrong(convChild->outputShapes[0].getDims()[1], convChild->getGroupNum()) &&
                                 convChild->outputShapes[0].getDims()[1] != 1 &&
                                 everyone_is(3, convChild->getWeightDims()[weightRank - 1], convChild->getWeightDims()[weightRank - 2]) &&
                                 everyone_is(1, convChild->getPaddingL()[stridesSize - 1], convChild->getPaddingL()[stridesSize - 2]) &&
                                 everyone_is(1, convChild->getPaddingR()[stridesSize - 1], convChild->getPaddingR()[stridesSize - 2]) &&
                                 everyone_is(1, convChild->getDilation()[stridesSize - 1] + 1, convChild->getDilation()[stridesSize - 2] + 1) &&
                                 convChild->getStride()[stridesSize - 1] == convChild->getStride()[stridesSize - 2] &&
                                 withBias &&
                                 one_of(convChild->getStride()[stridesSize - 1], 1, 2) &&
                                 childNode->getOutputShapeAtPort(0).getRank() == 4;

        return isSupportedParams;
    };

    auto isFusingWorthwhile = [&](const MKLDNNNodePtr &parentNode, const MKLDNNNodePtr &childNode) {
        if (!childNode->inputShapes[0].isStatic() || !childNode->outputShapes[0].isStatic()) {
            return false;
        }

        auto inDims = childNode->inputShapes[0].getStaticDims();
        auto outDims = childNode->outputShapes[0].getStaticDims();
        int elemSize = childNode->getOriginalOutputPrecisionAtPort(0).size();

        int L3_cache_size = utils::get_cache_size(3, false);
        int dw_conv_input_size = inDims[0] * inDims[1] * inDims[2] * inDims[3] * elemSize;
        int dw_conv_output_size = outDims[0] * outDims[1]* outDims[2] * outDims[3] * elemSize;

        auto parentConvolutionNode = std::dynamic_pointer_cast<MKLDNNConvolutionNode>(parentNode);
        if (parentConvolutionNode == nullptr)
            IE_THROW() << "Cannot get convolution node " << parentNode->getName();

        if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx2) || impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_common))
            return false;

        return (dw_conv_input_size + dw_conv_output_size > L3_cache_size / 2);
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        if (!isConvolutionNode(graphNodes[i])) continue;

        auto parentConvNode = graphNodes[i];
        if (!isSuitableParentConvolution(parentConvNode)) continue;

        auto childConvNode = parentConvNode->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildConvolution(parentConvNode, childConvNode)) continue;

        if (!isFusingWorthwhile(parentConvNode, childConvNode)) continue;

        parentConvNode->addFusedNode(childConvNode);

        for (auto node : childConvNode->getFusedWith()) {
            parentConvNode->addFusedNode(node);
        }
        childConvNode->clearFusedWith();

        graph.DropDWConvNode(childConvNode);
    }
}

// TODO [NM]: unite with FuseConvolutionAndSimpleOperation
void MKLDNNGraphOptimizer::FuseConvolutionAndSimpleOperationThroughMaxPool(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        return (node->getType() == Convolution || node->getType() == BinaryConvolution) && node->getChildEdges().size() == 1 &&
               node->getOriginalOutputPrecisionAtPort(0) == Precision::FP32;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (childNode->getAlgorithm() != PoolingMax || childNode->getChildEdges().size() != 1) {
            parent++;
            continue;
        }

        auto fuseCandidate = childNode->getChildEdgeAt(0)->getChild();
        if (parentNode->getType() == BinaryConvolution && !parentNode->canFuse(fuseCandidate)) {
            parent++;
            continue;
        }

        if (!one_of(fuseCandidate->getAlgorithm(), EltwiseRelu, EltwiseGelu, EltwiseElu, EltwiseSigmoid, EltwiseClamp, EltwiseTanh,
                                                   EltwiseSwish, EltwiseHswish, EltwiseMish, EltwiseHsigmoid, EltwiseRoundHalfToEven,
                                                   EltwiseRoundHalfAwayFromZero, EltwiseAbs, EltwiseSqrt, EltwiseSoftRelu)) {
            parent++;
            continue;
        }
        parentNode->addFusedNode(fuseCandidate);
        parentNode->addOriginalLayer(fuseCandidate->getOriginalLayers());
        auto parentEdges = fuseCandidate->parentEdges;
        for (auto &parentEdge : parentEdges) {
            auto p_edge = parentEdge.lock();
            if (p_edge->getParent() == childNode)
                continue;

            graph.RemoveEdge(p_edge);
        }
        graph.DropNode(fuseCandidate);
    }
}

void MKLDNNGraphOptimizer::FuseConvolutionAndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        return (node->getType() == Convolution || node->getType() == BinaryConvolution) && node->getChildEdges().size() == 1;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }
        const auto parentNodeType = parentNode->getType();

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        //  BF16 Quantize Layer Fusing Disabling
        if (BF16QuantizeNodeFusing(parentNode, childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize || childNode->getType() == Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == parentNodeType)
                    continue;

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FusePoolingAndFakeQuantize(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        if (node->getType() == Pooling) {
            if (!one_of(node->getOriginalInputPrecisionAtPort(0), Precision::U8, Precision::I8))
                return false;
            return node->getChildEdges().size() == 1 && node->getAlgorithm() == Algorithm::PoolingAvg;
        }
        return false;
    };

    auto isSuitableChildNode = [](MKLDNNNodePtr node) {
        return node->getType() == FakeQuantize && node->getAlgorithm() != Algorithm::FQBinarization;
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSuitableParentNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildNode(child)) continue;

        child->fuseInto(parent);

        auto parents = child->parentEdges;
        for (size_t i = 0; i < parents.size(); i++) {
            auto p_edge = parents[i].lock();
            if (p_edge->getParent()->getType() == Pooling)
                continue;

            graph.RemoveEdge(p_edge);
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
    auto &graphNodes = graph.GetNodes();

    auto isFusingSupported = [&](MKLDNNNodePtr conv, MKLDNNNodePtr child) {
        return child->getType() == Eltwise &&
                one_of(child->getAlgorithm(), EltwiseRelu, EltwiseElu, EltwiseSigmoid, EltwiseClamp, EltwiseSwish, EltwiseHswish,
                                              EltwiseMish, EltwiseHsigmoid, EltwiseRoundHalfToEven, EltwiseRoundHalfAwayFromZero, EltwiseSoftRelu);
    };

    for (auto &graphNode : graphNodes) {
        // TODO [DS]: at this moment this transformation prohibit for dynamic case
        if (graphNode->getType() != Eltwise || graphNode->getAlgorithm() != EltwiseAdd || graphNode->isDynamicNode() ||
                std::dynamic_pointer_cast<MKLDNNEltwiseNode>(graphNode)->isWithBroadcast())
            continue;

        // TODO: Enlarge to several inputs
        bool isSuitableNode = graphNode->getParentEdges().size() == 2;
        if (!isSuitableNode)
            continue;

        auto parent1 = graphNode->getParentEdgesAtPort(0)[0]->getParent();
        auto parent2 = graphNode->getParentEdgesAtPort(1)[0]->getParent();

        bool isSuitableParent1 = parent1->getType() == Convolution || parent1->getType() == BinaryConvolution;
        bool isSuitableParent2 = parent2->getType() == Convolution || parent2->getType() == BinaryConvolution;

        auto canFuseSum = [](MKLDNNBinaryConvolutionNode *binConv, MKLDNNNodePtr fuseCandidate) {
            if (binConv->getImplType() == impl_desc_type::ref)
                return false;

            if (binConv->isFusedWith(FakeQuantize))
                return false;

            if (fuseCandidate->getAlgorithm() == EltwiseAdd) {
                for (auto& fusedNode : binConv->fusedWith) {
                    const auto eltwise = std::dynamic_pointer_cast<MKLDNNEltwiseNode>(fusedNode);
                    if (eltwise && eltwise->isSpecialConvolutionAddFusing()) {
                        return false;
                    }
                }
                return true;
            }
            return false;
        };

        auto* binConvNode1 = dynamic_cast<MKLDNNBinaryConvolutionNode *>(parent1.get());
        if (binConvNode1) {
            isSuitableParent1 = isSuitableParent1 && canFuseSum(binConvNode1, graphNode);
        }

        auto* binConvNode2 = dynamic_cast<MKLDNNBinaryConvolutionNode *>(parent2.get());
        if (binConvNode2) {
            isSuitableParent2 = isSuitableParent2 && canFuseSum(binConvNode2, graphNode);
        }

        auto* convNode1 = dynamic_cast<MKLDNNConvolutionNode *>(parent1.get());
        if (convNode1) {
            if (!convNode1->canBeExecutedInInt8()) {
                isSuitableParent1 = isSuitableParent1 && convNode1->getFusedWith().empty();
            }
        }

        auto* convNode2 = dynamic_cast<MKLDNNConvolutionNode *>(parent2.get());
        if (convNode2) {
            if (!convNode2->canBeExecutedInInt8()) {
                isSuitableParent2 = isSuitableParent2 && convNode2->getFusedWith().empty();
            }
        }

        if (!isSuitableParent1 && !isSuitableParent2)
            continue;

        std::shared_ptr<MKLDNNNode> mergedConv;
        std::shared_ptr<MKLDNNNode> peerNode;

        if (isSuitableParent1 && isSuitableParent2) {
            // not merged operation (peerNode) has to be in low precision
            const auto isBranchQuantized = [](const MKLDNNNodePtr& branchParent) {
                const auto& fused = branchParent->getFusedWith();
                const auto branchPrecision = fused.empty() ?
                        branchParent->getOriginalOutputPrecisionAtPort(0) :
                        fused[fused.size() - 1]->getOriginalOutputPrecisionAtPort(0);
                return (branchPrecision == Precision::I8) || (branchPrecision == Precision::U8);
            };

            const auto isBranch1Quantized = isBranchQuantized(graphNode->getParentEdgesAtPort(0)[0]->getParent());
            const auto isBranch2Quantized = isBranchQuantized(graphNode->getParentEdgesAtPort(1)[0]->getParent());
            if (isBranch1Quantized || isBranch2Quantized) {
                // INT8
                const auto parent1CanBeMerged = parent1->getChildEdges().size() == 1ul;

                // if both branches are quantized, then parent1 is selected (result is not changed)
                mergedConv = isBranch2Quantized && parent1CanBeMerged ? parent1 : parent2;
                peerNode = isBranch2Quantized && parent1CanBeMerged ? parent2 : parent1;
            } else {
                // original FP32
                mergedConv = isSuitableParent1 ? parent1 : parent2;
                peerNode = isSuitableParent1 ? parent2 : parent1;
            }
        } else {
            mergedConv = isSuitableParent1 ? parent1 : parent2;
            peerNode = isSuitableParent1 ? parent2 : parent1;
        }

        if (isSuitableParent1 && isSuitableParent2) {
            if ((peerNode->getType() == Convolution || peerNode->getType() == BinaryConvolution) &&
                mergedConv->getChildEdges().size() != 1) {
                mergedConv = parent2;
                peerNode = parent1;
            }
        }
        if (peerNode->isConstant())
            continue;
        auto sum = graphNode;

        if (mergedConv->isConstant() && !sum->isConstant())
            continue;

        auto lastNode = sum;

        bool fuse_allowed = mergedConv->getChildEdges().size() == 1;
        for (size_t j = 0; fuse_allowed && j < mergedConv->getParentEdges().size(); j++)
            if (mergedConv->getParentEdgesAtPort(j)[0]->getParent() == peerNode)
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
            if (mergedConv->isConstant() && !lastNode->isConstant())
                continue;
            sum->fuseInto(mergedConv);
        }

        lastNode->fuseInto(mergedConv);

        if (mergedConv->fusedWith.size() > 0 &&
           (mergedConv->fusedWith[0]->getType() == Convolution || mergedConv->fusedWith[0]->getType() == BinaryConvolution)) {
            // Merged with DW_conv. Shape may change
            mergedConv->inputShapes.push_back(mergedConv->fusedWith[0]->outputShapes[0]);
        } else {
            mergedConv->inputShapes.push_back(mergedConv->outputShapes[0]);
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

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        return (node->getType() == MVN) && (node->getChildEdges().size() == 1);
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize || childNode->getType() == Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == MVN)
                    continue;

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseInterpolateAndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Interpolate && node->getChildEdges().size() == 1;
    };

    auto isSuitableChildNode = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
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
        if (!interpolateNode) {
            IE_THROW() << "Cannot cast " << parentNode->getName() << " to MKLDNNInterpolateNode";
        }
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
        if (!isSuitableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize || childNode->getType() == Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Interpolate)
                    continue;

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseNormalizeL2AndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == NormalizeL2 && node->getChildEdges().size() == 1;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize || childNode->getType() == Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == NormalizeL2)
                    continue;

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseReduceAndSimpleOperation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Reduce && node->getChildEdges().size() == 1;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize || childNode->getType() == Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge == nullptr)
                    IE_THROW() << "Cannot get parent edge " << childNode->getName();
                if (p_edge->getParent()->getType() == Reduce)
                    continue;

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void MKLDNNGraphOptimizer::FuseEltwiseAndSimple(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Eltwise && node->getChildEdges().size() == 1;
    };

    auto isSuitableChildNode = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        if (parentNode->isConstant() && !childNode->isConstant())
            return false;
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

        return parentNode->canFuse(childNode);
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();

        if ((parentNode->isDynamicNode() && !childNode->isDynamicNode()) || (!parentNode->isDynamicNode() && childNode->isDynamicNode())) {
            parent++;
            continue;
        }

        if (!isSuitableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == FakeQuantize) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Eltwise)
                    continue;

                graph.RemoveEdge(p_edge);
            }

            graph.DropNode(childNode);
        } else if (childNode->getType() == Eltwise) {
            auto children = childNode->childEdges;
            auto parents = childNode->parentEdges;
            auto initialParentInNum = parentNode->getParentEdges().size();

            for (size_t i = 0; i < parents.size(); i++) {
                auto p_edge = parents[i].lock();
                if (!p_edge) continue;
                auto parent = p_edge->getParent();
                if (!parent) continue;

                if (parent == parentNode) {
                    for (size_t j = 0; j < children.size(); j++) {
                        if (!children[j].lock())
                            continue;
                        auto child = children[j].lock()->getChild();
                        if (!child)
                            continue;

                        MKLDNNEdgePtr &remEdge = p_edge;
                        int inNum = 0;
                        if (remEdge) {
                            inNum = remEdge->getInputNum();
                            remEdge->drop();
                            graph.RemoveEdge(remEdge);
                        }
                        remEdge = children[j].lock();
                        int outNum = 0;
                        if (remEdge) {
                            outNum = remEdge->getOutputNum();
                            remEdge->drop();
                            graph.RemoveEdge(remEdge);
                        }
                        MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, child, inNum, outNum));
                        auto &graphEdges = graph.GetEdges();
                        graphEdges.push_back(newEdge);
                        parent->addEdge(newEdge);

                        parent->outputShapes[inNum] = child->inputShapes[outNum];
                    }
                } else {
                    MKLDNNEdgePtr &remEdge = p_edge;
                    int inNum = 0;
                    int outNum = parentNode->getParentEdges().size();
                    if (remEdge) {
                        inNum = remEdge->getInputNum();
                        // Need to keep order for MulAdd
                        if (childNode->getAlgorithm() == EltwiseMulAdd) {
                            outNum = initialParentInNum + remEdge->getOutputNum() - 1;
                        }
                        remEdge->drop();
                        graph.RemoveEdge(remEdge);
                    }

                    MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, parentNode, inNum, outNum));
                    auto &graphEdges = graph.GetEdges();
                    graphEdges.push_back(newEdge);
                    parent->addEdge(newEdge);

                    parentNode->inputShapes.push_back(parent->outputShapes[0]);
                }
            }

            graph.DropNode(childNode);
        } else {
            graph.DropNode(childNode);
        }
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
                IE_THROW() << "Cannot get reorder layer " << node->getName();
            MKLDNNReorderNode* nn = dynamic_cast<MKLDNNReorderNode*>(nextNode.get());
            if (nn == nullptr)
                IE_THROW() << "Cannot get reorder layer " << nextNode->getName();

            MKLDNNNodePtr p = n->getParentEdgesAtPort(0)[0]->getParent();
            MKLDNNNodePtr c = nn->getChildEdgesAtPort(0)[0]->getChild();

            auto oldEdgeNum = n->getParentEdgesAtPort(0)[0]->getInputNum();

            graph.DropNode(node);
            graph.DropNode(nextNode);

            processed.insert(node);
            processed.insert(nextNode);

            MKLDNNEdgePtr edge;
            for (auto cur : p->getChildEdgesAtPort(oldEdgeNum)) {
                if (cur->getChild() == c)
                    edge = cur;
            }
            if (!edge) IE_THROW() << "Inappropriate graph processing";


            std::string layerName = edge->getParent()->getName() + "_ScaleReorder_" + edge->getChild()->getName();
            graph.InsertReorder(edge, layerName, n->getInput(), nn->getOutput(), false);
            graph.GetEdges().erase(std::remove(graph.GetEdges().begin(), graph.GetEdges().end(), edge), graph.GetEdges().end());
        }
    }
}

void MKLDNNGraphOptimizer::FuseBroadcastAndEltwise(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    for (auto &graphNode : graphNodes) {
        if (graphNode->getType() != Generic
                || graphNode->getTypeStr() != "Broadcast"
                || graphNode->getChildEdges().size() != 1lu
                || graphNode->getChildEdgeAt(0)->getChild()->getType() != Eltwise)
            continue;

        MKLDNNNodePtr& broadcastNode = graphNode;
        MKLDNNNodePtr eltwiseNode = broadcastNode->getChildEdgeAt(0)->getChild();
        eltwiseNode->inputShapes[broadcastNode->getChildEdgeAt(0)->getOutputNum()]
                = broadcastNode->getInputShapeAtPort(0);

        auto& edges = graph.GetEdges();
        for (size_t i = 1lu; i < broadcastNode->getParentEdges().size(); i++) {
            auto constParent = broadcastNode->getParentEdgesAtPort(i)[0]->getParent();
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

void MKLDNNGraphOptimizer::FuseClampAndFakeQuantize(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableClampNode = [](MKLDNNNodePtr node) {
        return node->getType() == Eltwise && node->getChildEdges().size() == 1 && node->getAlgorithm() == EltwiseClamp;
    };

    auto isSuitableFakeQuantizeNode = [](MKLDNNNodePtr node) {
        return node->getType() == FakeQuantize && node->getAlgorithm() != FQBinarization;
    };

    auto fuseClampAndFakeQuantizeNodes = [](MKLDNNNodePtr parent, MKLDNNNodePtr child) {
        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(parent.get());
        if (eltwiseNode == nullptr)
            IE_THROW() << "Cannot cast " << parent->getName() << " to Eltwise node";

        auto* fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode*>(child.get());
        if (fakeQuantizeNode == nullptr)
            IE_THROW() << "Cannot cast " << child->getName() << " to FakeQuantize node";

        const std::vector<float>& cropLowData = fakeQuantizeNode->getCropLow();
        const std::vector<float>& cropHighData = fakeQuantizeNode->getCropHigh();

        std::vector<float> newCropLow(cropLowData.size());
        std::vector<float> newCropHigh(cropHighData.size());
        for (int i = 0; i < cropLowData.size(); i++)
            newCropLow[i] = std::max(cropLowData[i], eltwiseNode->getAlpha());
        for (int i = 0; i < cropHighData.size(); i++)
            newCropHigh[i] = std::min(cropHighData[i], eltwiseNode->getBeta());

        fakeQuantizeNode->setCropLow(newCropLow);
        fakeQuantizeNode->setCropHigh(newCropHigh);

        return true;
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSuitableClampNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSuitableFakeQuantizeNode(child)) continue;

        if (fuseClampAndFakeQuantizeNodes(parent, child)) {
            graph.DropNode(parent);
        }
    }
}

void MKLDNNGraphOptimizer::FusePerformedAsScaleShiftAndFakeQuantize(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto getConstPort = [](const MKLDNNNodePtr node) -> int {
        if (node->getParentEdgesAtPort(0)[0]->getParent()->getType() == Input && node->getParentEdgesAtPort(0)[0]->getParent()->isConstant()) {
            return 0;
        } else if (node->getParentEdgesAtPort(1)[0]->getParent()->getType() == Input && node->getParentEdgesAtPort(1)[0]->getParent()->isConstant()) {
           return 1;
        } else {
            return -1;
        }
    };

    auto isSuitableScaleShiftNode = [getConstPort](MKLDNNNodePtr node) {
        if (one_of(node->getAlgorithm(), EltwiseAdd, EltwiseSubtract, EltwiseMultiply, EltwiseDivide, EltwiseMulAdd)) {
            MKLDNNNode *parent = nullptr;
            if (node->getAlgorithm() != EltwiseMulAdd) {
                const auto constPort = getConstPort(node);
                if (constPort == -1) {
                    return false;
                }
                parent = node->getParentEdgesAtPort(1 - constPort)[0]->getParent().get();
            }
            return node->getType() == Eltwise && node->getChildEdges().size() == 1 && node->canBePerformedAsScaleShift(parent);
        }
        return false;
    };

    auto isSuitableFakeQuantizeNode = [](MKLDNNNodePtr node) {
        return node->getType() == FakeQuantize && node->getAlgorithm() != FQBinarization;
    };

    auto fuseScaleShiftAndFakeQuantizeNodes = [getConstPort](MKLDNNNodePtr parent, MKLDNNNodePtr child) {
        auto fakeQuantizeNode = std::dynamic_pointer_cast<MKLDNNFakeQuantizeNode>(child);
        if (fakeQuantizeNode == nullptr)
            IE_THROW() << "Cannot cast " << child->getName() << " to FakeQuantize node";

        std::vector<float> scalesBuffer;
        std::vector<float> shiftsBuffer;
        auto parentEltwise = std::dynamic_pointer_cast<MKLDNNEltwiseNode>(parent);
        if (!parentEltwise) {
            IE_THROW() << "Cannot cast " << parent->getName() << " to Eltwise node";
        }

        std::tie(scalesBuffer, shiftsBuffer) = parentEltwise->getScalesAndShifts(parent->getParentEdgesAtPort(1 - getConstPort(parent))[0]->getParent().get());

        const auto &outputShape = child->getOutputShapeAtPort(0);
        VectorDims outputDims = outputShape.getDims();
        const size_t channelPos = outputDims.size() > 1 ? 1 : 0;
        if (outputShape.isDynamic()) {
            if (outputDims[channelPos] == Shape::UNDEFINED_DIM) {
                if (scalesBuffer.size() > 1) {
                    outputDims[channelPos] = scalesBuffer.size();
                } else if (shiftsBuffer.size() > 1) {
                    outputDims[channelPos] = shiftsBuffer.size();
                } else {
                    return false;
                }
            }
        }

        scalesBuffer = makeAlignedBuffer(outputDims[channelPos], scalesBuffer, 1);
        shiftsBuffer = makeAlignedBuffer(outputDims[channelPos], shiftsBuffer, 1);

        for (int i = 0; i < scalesBuffer.size(); i++)
            if (scalesBuffer[i] == 0.f)
                return false;

        const std::vector<float>& cropLowData = fakeQuantizeNode->getCropLow();
        const std::vector<float>& cropHighData = fakeQuantizeNode->getCropHigh();
        const std::vector<float>& inputScaleData = fakeQuantizeNode->getInputScale();
        const std::vector<float>& inputShiftData = fakeQuantizeNode->getInputShift();

        std::vector<float> newCropLow(scalesBuffer.size());
        std::vector<float> newCropHigh(scalesBuffer.size());
        std::vector<float> newInputScale(scalesBuffer.size());
        std::vector<float> newInputShift(scalesBuffer.size());

        for (int i = 0; i < newCropLow.size(); i++) {
            float cl = cropLowData.size() == 1 ? cropLowData[0] : cropLowData[i];
            float ch = cropHighData.size() == 1 ? cropHighData[0] : cropHighData[i];

            float newCL = (cl - shiftsBuffer[i]) / scalesBuffer[i];
            float newCH = (ch - shiftsBuffer[i]) / scalesBuffer[i];

            newCropLow[i] = std::min(newCL, newCH);
            newCropHigh[i] = std::max(newCL, newCH);
            if (std::isinf(newCropLow[i])) {
                newCropLow[i] = std::numeric_limits<float>::lowest();
            }
            if (std::isinf(newCropHigh[i])) {
                newCropHigh[i] = std::numeric_limits<float>::max();
            }
        }

        std::vector<float> zeroShift(newInputScale.size(), 0.f);

        const auto isSubnormal = [](const float value) {
            const uint32_t *u32data = reinterpret_cast<const uint32_t*>(&value);
            return (*u32data) && (((*u32data) & (0xFF << 23)) == 0);
        };

        for (int i = 0; i < newInputScale.size(); i++) {
            float isc = inputScaleData.size() == 1 ? inputScaleData[0] : inputScaleData[i];

            newInputScale[i] = isc * scalesBuffer[i];
            if (isSubnormal(newInputScale[i])) {
                newInputScale[i] = 0.f;
                // zero value have to be shifted if it's not in input range
                float cl = cropLowData.size() == 1 ? cropLowData[0] : cropLowData[i];
                float ch = cropHighData.size() == 1 ? cropHighData[0] : cropHighData[i];
                if (0.f < cl) {
                    zeroShift[i] = isc * cl;
                }
                if (ch < 0.f) {
                    zeroShift[i] = isc * ch;
                }
            }
        }

        for (int i = 0; i < newInputShift.size(); i++) {
            float isc = inputScaleData.size() == 1 ? inputScaleData[0] : inputScaleData[i];
            float ish = inputShiftData.size() == 1 ? inputShiftData[0] : inputShiftData[i];

            newInputShift[i] = ish + shiftsBuffer[i] * isc + zeroShift[i];
            if (isSubnormal(newInputShift[i])) {
                newInputShift[i] = 0.f;
            }
        }

        fakeQuantizeNode->setCropLow(newCropLow);
        fakeQuantizeNode->setCropHigh(newCropHigh);
        fakeQuantizeNode->setInputScale(newInputScale);
        fakeQuantizeNode->setInputShift(newInputShift);

        return true;
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSuitableScaleShiftNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSuitableFakeQuantizeNode(child)) continue;

        if (fuseScaleShiftAndFakeQuantizeNodes(parent, child)) {
            auto parentEdges = parent->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (!p_edge->getParent()->isConstant())
                    continue;

                graph.RemoveEdge(p_edge);
            }

            graph.DropNode(parent);
        }
    }
}

void MKLDNNGraphOptimizer::MergeTransposeAndReorder(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        return node->getType() == Transpose
                && node->getChildEdges().size() == 1
                && !node->isDynamicNode();   // TODO [DS]: enable for dynamic shapes when inPlace in the dynamic case is available (CVS-74863)
    };

    auto isSuitableChildNode = [](MKLDNNNodePtr node) {
        return node->getType() == Reorder
                && node->getChildEdges().size() == 1
                && !node->isDynamicNode();   // TODO [DS]: enable for dynamic shapes when inPlace in the dynamic case is available (CVS-74863)
    };

    // Method checkAscendingSummaryOrder() checks that after the sequential execution of Transpose and Reorder nodes,
    // the order of the elements in the memory will not change. In other words, that Transpose+Reorder is identical permutation.
    auto checkAscendingSummaryOrder = [](std::shared_ptr<MKLDNNNode> &parentNode, std::shared_ptr<MKLDNNNode> &childNode) -> bool {
        auto* transposeNode = dynamic_cast<MKLDNNTransposeNode*>(parentNode.get());
        auto* reorderNode = dynamic_cast<MKLDNNReorderNode*>(childNode.get());
        if (!transposeNode || !reorderNode) {
            return false;
        }

        auto& transposeOrder = transposeNode->getOrder();
        auto layoutOrder = transposeNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc->as<BlockedMemoryDesc>()->getOrder();

        auto inBlockedDesc = reorderNode->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc->as<BlockedMemoryDesc>();
        auto outBlockedDesc = reorderNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc->as<BlockedMemoryDesc>();

        auto& inOrder = inBlockedDesc->getOrder();
        auto& outOrder = outBlockedDesc->getOrder();

        if (transposeOrder.size() != layoutOrder.size() || layoutOrder.size() != inOrder.size() || inOrder.size() != outOrder.size()) {
            return false;
        }

        // revLayoutOrder - reverse permutation for layoutOrder
        auto revLayoutOrder = VectorDims(layoutOrder.size());
        for (int i = 0; i < revLayoutOrder.size(); i++) {
            revLayoutOrder[layoutOrder[i]] = i;
        }

        // newTransposeOrder - Transpose layout-aware permutation
        auto newTransposeOrder = VectorDims(transposeOrder.size());
        for (int i = 0; i < newTransposeOrder.size(); i++) {
            newTransposeOrder[i] = layoutOrder[transposeOrder[revLayoutOrder[i]]];
        }

        // reorderOrder - Reorder layout-aware permutation
        auto reorderOrder = VectorDims(outOrder.size());
        for (int i = 0; i < reorderOrder.size(); i++) {
            for (int j = 0; j < reorderOrder.size(); j++) {
                if (outOrder[i] == inOrder[j]) {
                    reorderOrder[i] = j;
                    continue;
                }
            }
        }

        // summaryOrder - resulting Transpose+Reorder permutation
        auto summaryOrder = VectorDims(transposeOrder.size());
        for (int i = 0; i < summaryOrder.size(); i++) {
            summaryOrder[i] = reorderOrder[newTransposeOrder[i]];
        }

        // check that Transpose+Reorder is the identical permutation
        for (int i = 0; i < summaryOrder.size(); i++) {
            if (summaryOrder[i] != i) {
                return false;
            }
        }

        return true;
    };

    // Transpose and Reorder do opposite permutation to each other.
    // Example:
    //      chain [physical layout: NCHW, logical layout: NCHW] -> Transpose(order=0312) -> [physical layout: NWCH, logical layout: NCHW] ->
    //      Reorder(nchw->nhwc) -> [physical layout: NCHW, logical layout: NHWC] can be replaced with Reorder(nchw->nhwc; isOptimized=true)
    //      which will just reinterprets layout without physical change of the memory.
    // Two cases are possible:
    //      1) inPrec = outPrec
    //          In this case, we replace Transpose+Reorder pattern with a new Reorder that does nothing.
    //      2) inPrec != outPrec
    //          As in the first case, we also replace Transpose+Reorder pattern with a new Reorder.
    //          Additionally, we insert another Reorder that performs the conversion from the input precision (inPrec)
    //          to the output precision (outPrec)
    auto mergeTransposeAndReorder = [&](std::shared_ptr<MKLDNNNode>& parentNode, std::shared_ptr<MKLDNNNode>& childNode) {
        auto parentParentNode = parentNode->getParentEdgesAtPort(0)[0]->getParent();
        auto parentParentConstNode = parentNode->getParentEdgesAtPort(1)[0]->getParent();
        auto childChildNode = childNode->getChildEdgeAt(0)->getChild();

        auto &remEdge = parentParentConstNode->getChildEdgeAt(0);
        remEdge->drop();
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == remEdge) {
                edges.erase(it);
                parentParentConstNode->remove();
                break;
            }
        }

        graph.DropNode(parentNode);
        graph.DropNode(childNode);

        auto& inDesc = parentNode->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc;
        auto& outDesc = childNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc;

        auto inPrec = inDesc->getPrecision();
        auto outPrec = outDesc->getPrecision();

        auto reorderInDesc = inDesc;
        auto reorderOutDesc = outDesc->cloneWithNewPrecision(inPrec);

        std::string reorderlayerName = parentParentNode->getName() + "_" +
                MKLDNNReorderNode::getReorderArgs(*reorderInDesc, *reorderOutDesc) + "_" + "fake";

        MKLDNNEdgePtr edge;
        for (auto &childEdge : parentParentNode->getChildEdges()) {
            if (childEdge.lock()->getChild() == childChildNode) {
                edge = childEdge.lock();
                break;
            }
        }
        if (!edge) {
            IE_THROW() << "Transpose node '" << parentNode->getName() << "' has invalid edges.";
        }

        auto reorderNode = graph.InsertReorder(edge, reorderlayerName, *reorderInDesc, *reorderOutDesc, true);

        // case 2
        if (inPrec != outPrec) {
            auto reorderInDesc2 = reorderOutDesc;
            auto reorderOutDesc2 = outDesc;

            std::string reorderLayerName2 = reorderNode->getName() + "_" +
                                    MKLDNNReorderNode::getReorderArgs(*reorderInDesc2, *reorderOutDesc2) + "_" + childChildNode->getName();

            graph.InsertReorder(reorderNode->getChildEdgeAt(0), reorderLayerName2, *reorderInDesc2, *reorderOutDesc2, false);
        }
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parentNode = graphNodes[i];
        if (!isSuitableParentNode(parentNode)) {
            continue;
        }
        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildNode(childNode)) {
            continue;
        }

        if (checkAscendingSummaryOrder(parentNode, childNode)) {
            mergeTransposeAndReorder(parentNode, childNode);
        }
    }
}

void MKLDNNGraphOptimizer::reshapeRnnSeq(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](MKLDNNNodePtr node) {
        if (node->type != RNNSeq)
            return false;
        auto rnnNode = std::dynamic_pointer_cast<MKLDNNRNN>(node);
        return rnnNode && !rnnNode->hasNativeOrder() && node->outputShapes[0].getRank() == 4 && node->outputShapes[0].getDims()[1] == 1;
    };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto parentNode = graphNodes[i];
        if (!isSuitableParentNode(parentNode)) {
            continue;
        }

        auto childrenEdges = parentNode->getChildEdgesAtPort(0);
        auto minDims = parentNode->getOutputShapeAtPort(0).getMinDims();
        auto maxDims = parentNode->getOutputShapeAtPort(0).getMaxDims();
        minDims.erase(minDims.begin() + 1);
        maxDims.erase(maxDims.begin() + 1);
        parentNode->outputShapes[0] = {minDims, maxDims};

        for (size_t j = 0; j < childrenEdges.size(); j++) {
            auto edge = childrenEdges[j];
            auto childNode = edge->getChild();

            const auto secondInput = std::make_shared<ngraph::opset1::Constant>(ov::element::i32, ngraph::Shape{1}, std::vector<int>{1});
            const auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(
                std::make_shared<ngraph::opset1::Parameter>(details::convertPrecision(parentNode->getOriginalOutputPrecisionAtPort(0)),
                                                            parentNode->getOutputShapeAtPort(0).toPartialShape()), secondInput);
            unsqueeze->set_friendly_name(parentNode->getName() + "_abc_a1bc_" + std::to_string(j));

            const auto cpuUnsqueeze = std::make_shared<MKLDNNReshapeNode>(unsqueeze, graph.getEngine(), graph.weightsCache);
            graph.InsertNode(parentNode, childNode, cpuUnsqueeze, edge->getInputNum(), edge->getOutputNum(), false);

            const auto cpuConstant = std::make_shared<MKLDNNInputNode>(secondInput, graph.getEngine(), graph.weightsCache);
            MKLDNNEdgePtr newEdge(new MKLDNNEdge(cpuConstant, cpuUnsqueeze, 0, 1));
            cpuUnsqueeze->addEdge(newEdge);
            auto &graphEdges = graph.GetEdges();
            graphEdges.push_back(newEdge);
            graphNodes.push_back(cpuConstant);

            edge->drop();
            graph.RemoveEdge(edge);
        }
    }
}
