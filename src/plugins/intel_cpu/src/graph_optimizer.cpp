// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_optimizer.h"

#include "dnnl_extension_utils.h"
#include "nodes/reshape.h"
#include "nodes/pooling.h"
#include "nodes/eltwise.h"
#include "nodes/concat.h"
#include "nodes/reorder.h"
#include "nodes/conv.h"
#include "nodes/deconv.h"
#include "nodes/bin_conv.h"
#include "nodes/fake_quantize.h"
#include "nodes/mvn.h"
#include "nodes/transpose.h"
#include "nodes/interpolate.h"
#include "nodes/reduce.h"
#include "nodes/input.h"
#include "nodes/rnn.h"
#include "nodes/common/cpu_convert.h"

#include "onednn/dnnl.h"

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

#include "itt.h"
#include "memory_desc/cpu_memory_desc_utils.h"

using namespace dnnl;
using namespace InferenceEngine;
using namespace ov::intel_cpu::node;

namespace ov {
namespace intel_cpu {

GraphOptimizer::GraphOptimizer() {}

void GraphOptimizer::ApplyCommonGraphOptimizations(Graph &graph) {
    OV_ITT_SCOPE_CHAIN(FIRST_INFERENCE, taskChain, itt::domains::intel_cpu_LT, "ApplyCommonGraphOptimizations", "FuseConvolutionAndBias");
    FuseConvolutionMatMulDeconvAndBias(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseMultiplyAndAdd");
    FuseMultiplyAndAdd(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "MergeConvertAndScaleShift");
    MergeConvertAndScaleShift(graph);
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

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "RemoveSameConvert");
    RemoveSameConvert(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "RemoveDroppedEdges");
    graph.RemoveDroppedEdges();
}

void GraphOptimizer::ApplyImplSpecificGraphOptimizations(Graph &graph) {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "GraphOptimizer::ApplyImplSpecificGraphOptimizations");

    DropDoubleReorders(graph);
    graph.RemoveDroppedNodes();

    MergeTransposeAndReorder(graph);
    graph.RemoveDroppedNodes();

    graph.RemoveDroppedEdges();
}

void GraphOptimizer::FuseConvolutionMatMulDeconvAndBias(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const NodePtr& node) {
        const auto deconv = std::dynamic_pointer_cast<Deconvolution>(node);
        // bias should be the first child
        if (!node->getFusedWith().empty())
            return false;
        // no other child other than bias-add
        if (node->getChildEdges().size() != 1)
            return false;

        if (!deconv)
            return (node->getType() == Type::Convolution || node->getType() == Type::MatMul) &&
                   node->getParentEdges().size() == 2;
        else
            return deconv->canFuseBias();
    };

    auto isSuitableChildNode = [&](const NodePtr& parentNode, const NodePtr& childNode) {
        if (childNode->getAlgorithm() != Algorithm::EltwiseAdd
            || !childNode->getFusedWith().empty()
            || childNode->getParentEdges().size() != 2)
            return false;

        const auto biasNode = childNode->getParentEdgesAtPort(1)[0]->getParent();
        if (biasNode->getType() != Type::Input || !biasNode->isConstant() || biasNode->getChildEdges().size() != 1)
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

        for (size_t i = 0; i < biasDims.size(); i++) {
            if (biasDims[i] != 1 && static_cast<int>(i) != channelAxis)
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

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseConvolutionMatMulDeconvAndBias_ParentNode);

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseConvolutionMatMulDeconvAndBias_ChildNode);

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

                    EdgePtr &remEdge = p_edge;
                    int inNum = 0;
                    if (remEdge) {
                        inNum = remEdge->getInputNum();
                        graph.RemoveEdge(remEdge);
                    }
                    remEdge = childs[j].lock();
                    int outNum = 0;
                    if (remEdge) {
                        outNum = remEdge->getOutputNum();
                        graph.RemoveEdge(remEdge);
                    }
                    EdgePtr newEdge(new Edge(parent, child, inNum, outNum));
                    auto &graphEdges = graph.GetEdges();
                    graphEdges.push_back(newEdge);
                    parent->addEdge(newEdge);
                }
            } else {
                EdgePtr &remEdge = p_edge;
                int inNum = 0;
                if (remEdge) {
                    inNum = remEdge->getInputNum();
                    graph.RemoveEdge(remEdge);
                }

                const auto& parentEltwise = parentNode;
                EdgePtr newEdge(new Edge(parent, parentEltwise, inNum, parentEltwise->getParentEdges().size()));
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

void GraphOptimizer::FuseDeconvolutionAndSimpleOperation(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr node) {
        if (node->getType() != Type::Deconvolution || node->getChildEdges().size() != 1)
            return false;
        const auto deconv = std::dynamic_pointer_cast<Deconvolution>(node);
        if (deconv == nullptr)
            IE_THROW() << "Cannot cast to deconvolution node " << node->getName();

        if (deconv->getAlgorithm() != Algorithm::DeconvolutionCommon) {
            return true;
        }

        const auto& strides = deconv->getStride();
        const auto& kernel = deconv->getWeightDims();
        // WA oneDNN doesn't support fusing post ops after deconvolution with strides over kernel size
        bool isSupportedParams = strides[strides.size() - 1] <= static_cast<dnnl_dim_t>(kernel[kernel.size() - 1]);
        if (strides.size() > 1)
            isSupportedParams &= strides[strides.size() - 2] <= static_cast<dnnl_dim_t>(kernel[kernel.size() - 2]);
        if (strides.size() > 2)
            isSupportedParams &= strides[strides.size() - 3] <= static_cast<dnnl_dim_t>(kernel[kernel.size() - 3]);
        return isSupportedParams;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseDeconvolutionAndSimpleOperation_ParentNode);

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseDeconvolutionAndSimpleOperation_ChildNode);

        childNode->fuseInto(parentNode);

        auto parentEdges = childNode->parentEdges;
        for (auto &parentEdge : parentEdges) {
            auto p_edge = parentEdge.lock();
            if (p_edge->getParent()->getType() == Type::Deconvolution)
                continue;

            graph.RemoveEdge(p_edge);
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseMultiplyAndAdd(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableSecondInput = [](const NodePtr& node, VectorDims dataDims) {
        if (node->getType() != Type::Input || !node->isConstant())
            return false;
        const auto secondInputDims = node->getOutputShapeAtPort(0).getStaticDims();
        if (secondInputDims.size() != dataDims.size() || secondInputDims.size() < 2)
            return false;

        auto getChannelAxis = [](const VectorDims& dims) {
            auto channelAxis = -1;
            for (size_t i = 0; i < dims.size(); i ++) {
                if (dims[i] != 1) {
                    if (channelAxis != -1) // more than one axis is != 1
                        return -1;
                    else
                        channelAxis = i;
                }
            }
            return channelAxis;
        };

        const auto channelAxis = getChannelAxis(secondInputDims);
        if (channelAxis == -1)
            return false;

        if (secondInputDims[0] != 1 || !dimsEqualWeak(secondInputDims[channelAxis], dataDims[channelAxis]))
            return false;

        return true;
    };

    auto isSuitableParentNode = [&](const NodePtr& node) {
        if (node->getAlgorithm() != Algorithm::EltwiseMultiply || !node->getFusedWith().empty() ||
            node->getParentEdges().size() != 2 || node->getChildEdges().size() != 1)
            return false;

        return isSuitableSecondInput(node->getParentEdgesAtPort(1)[0]->getParent(), node->getInputShapeAtPort(0).getDims());
    };

    auto isSuitableChildNode = [&](const NodePtr& parentNode, const NodePtr& childNode) {
        if (childNode->getAlgorithm() != Algorithm::EltwiseAdd || !childNode->getFusedWith().empty() || childNode->getParentEdges().size() != 2)
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

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseMultiplyAndAdd_ParentNode);

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseMultiplyAndAdd_ChildNode);

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

                    EdgePtr &remEdge = p_edge;
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
                    EdgePtr newEdge(new Edge(parent, child, inNum, outNum));
                    auto &graphEdges = graph.GetEdges();
                    graphEdges.push_back(newEdge);
                    parent->addEdge(newEdge);
                }
            } else {
                EdgePtr &remEdge = p_edge;
                int inNum = 0;
                if (remEdge) {
                    inNum = remEdge->getInputNum();
                    remEdge->drop();
                    graph.RemoveEdge(remEdge);
                }

                auto& parentEltwise = parentNode;
                EdgePtr newEdge(new Edge(parent, parentEltwise, inNum, parentEltwise->getParentEdges().size()));
                auto &graphEdges = graph.GetEdges();
                graphEdges.push_back(newEdge);
                parent->addEdge(newEdge);

                parentEltwise->inputShapes.push_back(parent->getOutputShapeAtPort(0));
            }
        }

        parentNode->addOriginalInputPrecision(childNode->getOriginalInputPrecisionAtPort(1));
        parentNode->setAlgorithm(Algorithm::EltwiseMulAdd);
        parentNode->setTypeStr("MulAdd");
        parentNode->addOriginalLayer(childNode->getOriginalLayers());
        graph.DropNode(childNode);
    }
}

void GraphOptimizer::MergeConvertAndScaleShift(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr parentNode) {
        return parentNode->getType() == Type::Convert && parentNode->getChildEdges().size() == 1 &&
               (parentNode->getOriginalInputPrecisionAtPort(0) == Precision::U8 ||
                parentNode->getOriginalInputPrecisionAtPort(0) == Precision::I8) &&
               parentNode->getOriginalOutputPrecisionAtPort(0) == Precision::FP32;
    };

    auto isSuitableChildNode = [](NodePtr childNode) {
        return childNode->getType() == Type::Eltwise && childNode->getParentEdges().size() != 2;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(MergeConvertAndScaleShift_ParentNode);

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildNode(childNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(MergeConvertAndScaleShift_ChildNode);

        auto parents = parentNode->parentEdges;
        for (size_t i = 0; i < parents.size(); i++) {
            auto p_edge = parents[i].lock();
            if (!p_edge) continue;
            auto parent = p_edge->getParent();
            if (!parent) continue;

            if (!parentNode->childEdges[0].lock())
                continue;
            auto child = parentNode->childEdges[0].lock()->getChild();
            if (!child)
                continue;

            EdgePtr& remEdge = p_edge;
            int inNum = 0;
            if (remEdge) {
                inNum = remEdge->getInputNum();
                remEdge->drop();
                graph.RemoveEdge(remEdge);
            }
            remEdge = parentNode->childEdges[0].lock();
            int outNum = 0;
            if (remEdge) {
                outNum = remEdge->getOutputNum();
                remEdge->drop();
                graph.RemoveEdge(remEdge);
            }
            EdgePtr newEdge(new Edge(parent, child, inNum, outNum));
            auto& graphEdges = graph.GetEdges();
            graphEdges.push_back(newEdge);
            parent->addEdge(newEdge);
        }

        childNode->setOriginalInputPrecisionAtPort(0, parentNode->getOriginalInputPrecisionAtPort(0));
        childNode->addOriginalLayer(parentNode->getOriginalLayers());
        graph.DropNode(parentNode);
    }
}

void GraphOptimizer::FuseConvolutionAndZeroPoints(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableConvNode = [](NodePtr node) {
        bool retVal = false;
        if (node->getType() == Type::Convolution) {
            if (auto convNode = std::dynamic_pointer_cast<Convolution>(node)) {
                auto rank = convNode->getInputShapeAtPort(0).getRank();
                // int8 depthwise convolution does not support fusing zero points in 3D case
                if (implication(convNode->isDepthWise(), rank < 5)) {
                    retVal = true;
                }
            }
        }
        return retVal;
    };

    auto initializeInputZeroPoints = [](NodePtr node, NodePtr parent0, NodePtr parent1) {
        auto* convNode = dynamic_cast<Convolution*>(node.get());
        if (convNode == nullptr)
            IE_THROW() << "Cannot get convolution node " << node->getName();

        auto IC = node->getInputShapeAtPort(0).getDims()[1];
        auto OC = node->getOutputShapeAtPort(0).getDims()[1];

        if (Shape::UNDEFINED_DIM == IC || Shape::UNDEFINED_DIM == OC)
            return false;
        if (parent0->getType() != Type::Eltwise)
            return false;
        if (!parent0->getFusedWith().empty() || !parent1->getFusedWith().empty())
            return false;

        // The plug-in doesn't support FP32 convolution with input/weights zero points.
        // In case weights are in FP32 (or we have zero points on weights which are not supported by INT8 convolution) we cannot use
        // INT8 implementation so we have to disable input zero points fusing as well.
        if (parent1->getType() != Type::Input || !parent1->isConstant() || parent1->getOriginalOutputPrecisionAtPort(0) != Precision::I8) {
            return false;
        }

        if (parent0->getAlgorithm() != Algorithm::EltwiseSubtract)
            return false;

        if (parent0->getParentEdges().size() != 2)
            return false;

        auto subtractArg1 = parent0->getParentEdgesAtPort(1)[0]->getParent();
        if (subtractArg1->getType() != Type::Input || !subtractArg1->isConstant())
            return false;

        if (subtractArg1->getOriginalOutputPrecisionAtPort(0) != Precision::U8)
            return false;

        if (parent0->getInputShapeAtPort(1).getRank() < 2) {
            return false;
        }

        auto zpDims = parent0->getInputShapeAtPort(1).getDims();
        if (zpDims[0] != 1 || !dimsEqualStrong(zpDims[1], IC))
            return false;

        for (size_t i = 2; i < zpDims.size(); i++) {
            if (zpDims[i] != 1)
                return false;
        }

        auto subtractArg0 = parent0->getParentEdgesAtPort(0)[0]->getParent();
        if (subtractArg0->getOriginalOutputPrecisionAtPort(0) != Precision::U8)
            return false;

        auto zeroPointsConstant = dynamic_cast<node::Input*>(subtractArg1.get());
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
        convNode->initializeInputZeroPoints(zeroPointsData, zeroPointDataSize);
        return true;
    };

    auto initializeOutputCompensation = [](NodePtr node) {
        auto* convNode = dynamic_cast<Convolution*>(node.get());
        if (convNode == nullptr)
            IE_THROW() << "Cannot get convolution node " << node->getName();

        if (convNode->legacyInputZeroPoints.empty())
            return;
        if (convNode->legacyOutputCompensation.empty())
            convNode->legacyOutputCompensation.resize(convNode->getOutputShapeAtPort(0).getDims()[1]);

        auto weightsConstant = dynamic_cast<node::Input*>(convNode->getParentEdgesAtPort(1)[0]->getParent().get());
        if (!weightsConstant || !weightsConstant->isConstant())
            return;

        auto weightsBlob = weightsConstant->getMemoryPtr();
        if (weightsBlob == nullptr)
            IE_THROW() << "Cannot cast to TBlob internal weights blob";

        auto weightsPtr = static_cast<const int8_t*>(weightsBlob->GetPtr());
        if (weightsPtr == nullptr)
            IE_THROW() << "weightsBlob has not allocated buffer";

        auto G = convNode->getGroupNum();
        const size_t groupOffset = convNode->getAlgorithm() == Algorithm::ConvolutionGrouped ? 1 : 0;
        auto& weightsConstantDims = weightsConstant->outputShapes[0].getStaticDims();

        auto OC = weightsConstantDims[0 + groupOffset];
        auto IC = weightsConstantDims[1 + groupOffset];
        auto KD = weightsConstantDims.size() == (5 + groupOffset) ? weightsConstantDims[weightsConstantDims.size() - 3] : 1;
        auto KH = weightsConstantDims.size() == (3 + groupOffset) ? 1 : weightsConstantDims[weightsConstantDims.size() - 2];
        auto KW = weightsConstantDims[weightsConstantDims.size() - 1];

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

                                auto izp = !convNode->legacyInputZeroPoints.empty() ? static_cast<int32_t>(convNode->legacyInputZeroPoints[g * IC + ic]) : 0;
                                a += w * izp;

                                auto wzp = !convNode->legacyWeightsZeroPoints.empty() ?
                                            static_cast<int32_t>(convNode->legacyWeightsZeroPoints[g * OC + oc]) : 0;
                                a -= wzp * izp;
                            }
                        }
                    }
                }
                convNode->legacyOutputCompensation[g * OC + oc] = -a;
            }
        }
    };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto conv = graphNodes[i];
        if (!isSuitableConvNode(conv)) continue;

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseConvolutionAndZeroPoints_ConvNode);

        auto dataEltwise = conv->getParentEdgesAtPort(0)[0]->getParent();
        auto weightsEltwise = conv->getParentEdgesAtPort(1)[0]->getParent();
        if (initializeInputZeroPoints(conv, dataEltwise, weightsEltwise)) {
            auto p_edge = dataEltwise->getParentEdgesAtPort(1)[0];
            graph.RemoveEdge(p_edge);
            graph.DropNode(dataEltwise);
            initializeOutputCompensation(conv);
        }
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
static bool BF16QuantizeNodeFusing(const NodePtr& parentNode, const NodePtr& childNode) {
    return childNode->getType() == Type::FakeQuantize &&
        one_of(Precision::BF16,
            parentNode->getOriginalOutputPrecisionAtPort(0),
            childNode->getOriginalOutputPrecisionAtPort(0));
}

void GraphOptimizer::FuseFullyConnectedAndSimpleOperation(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr node) {
        return node->getType() == Type::FullyConnected && node->getChildEdges().size() == 1;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseFullyConnectedAndSimpleOperation_ParentNode);

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

        if (childNode->getType() == Type::FakeQuantize || childNode->getType() == Type::Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Type::FullyConnected)
                    continue;

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseMatMulAndSimpleOperation(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](const NodePtr& node) {
        return node->getType() == Type::MatMul && node->getChildEdges().size() == 1;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSutableParentNode(parentNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseMatMulAndSimpleOperation_ParentNode);

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == Type::FakeQuantize || childNode->getType() == Type::Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Type::MatMul)
                    continue;

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseConvolutionAndDWConvolution(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isConvolutionNode = [](const NodePtr &node) {
        return node->getType() == Type::Convolution;
    };

    auto is1x1Convolution = [](const std::shared_ptr<Convolution> &conv) {
        const auto weightRank = conv->getWeightDims().size();
        return conv->getWeightDims()[weightRank - 1] == 1 && conv->getWeightDims()[weightRank - 2] == 1;
    };

    auto isSuitableParentConvolution = [&](NodePtr node) {
        if (node->isDropped())
            return false;

        if (node->isDynamicNode())
            return false;

        const auto conv = std::dynamic_pointer_cast<Convolution>(node);
        if (conv == nullptr)
            IE_THROW() << "Cannot cast to convolution node " << node->getName();

        if (!conv->legacyWeightsZeroPoints.empty())
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
                everyone_is(1u, strides[strides.size() - 1], strides[strides.size() - 2]) &&
                everyone_is(0u, paddings[paddings.size() - 1], paddings[paddings.size() - 2]) &&
                !conv->canBeExecutedInInt8();
        if (!isSupportedParams) return false;

        return node->getChildEdges().size() == 1 && isConvolutionNode(node->getChildEdgeAt(0)->getChild());
    };

    auto isSuitableChildConvolution = [&](const NodePtr &parentNode, const NodePtr &childNode) {
        if (parentNode->isDropped() || childNode->isDropped())
            return false;

        if (childNode->isDynamicNode())
            return false;

        const auto convChild = std::dynamic_pointer_cast<Convolution>(childNode);
        if (convChild == nullptr)
            IE_THROW() << "Cannot cast to convolution node " << childNode->getName();

        const auto convParent = std::dynamic_pointer_cast<Convolution>(parentNode);
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

        if (!convChild->legacyInputZeroPoints.empty() || !convChild->legacyWeightsZeroPoints.empty())
            return false;

        bool withBias = convChild->getOriginalInputPrecisions().size() == 3;

        const auto weightRank = convChild->getWeightDims().size();
        const auto stridesSize = convChild->getStride().size();
        bool isSupportedParams = dimsEqualStrong(convChild->outputShapes[0].getDims()[1], convChild->getGroupNum()) &&
                                 convChild->outputShapes[0].getDims()[1] != 1 &&
                                 everyone_is(3u, convChild->getWeightDims()[weightRank - 1], convChild->getWeightDims()[weightRank - 2]) &&
                                 everyone_is(1u, convChild->getPaddingL()[stridesSize - 1], convChild->getPaddingL()[stridesSize - 2]) &&
                                 everyone_is(1u, convChild->getPaddingR()[stridesSize - 1], convChild->getPaddingR()[stridesSize - 2]) &&
                                 everyone_is(1u, convChild->getDilation()[stridesSize - 1] + 1, convChild->getDilation()[stridesSize - 2] + 1) &&
                                 convChild->getStride()[stridesSize - 1] == convChild->getStride()[stridesSize - 2] &&
                                 withBias &&
                                 one_of(convChild->getStride()[stridesSize - 1], 1u, 2u) &&
                                 childNode->getOutputShapeAtPort(0).getRank() == 4;

        return isSupportedParams;
    };

    auto isFusingWorthwhile = [&](const NodePtr &parentNode, const NodePtr &childNode) {
        if (!childNode->inputShapes[0].isStatic() || !childNode->outputShapes[0].isStatic()) {
            return false;
        }

        auto inDims = childNode->inputShapes[0].getStaticDims();
        auto outDims = childNode->outputShapes[0].getStaticDims();
        int elemSize = childNode->getOriginalOutputPrecisionAtPort(0).size();

        int L3_cache_size = utils::get_cache_size(3, false);
        int dw_conv_input_size = inDims[0] * inDims[1] * inDims[2] * inDims[3] * elemSize;
        int dw_conv_output_size = outDims[0] * outDims[1]* outDims[2] * outDims[3] * elemSize;

        auto parentConvolutionNode = std::dynamic_pointer_cast<Convolution>(parentNode);
        if (parentConvolutionNode == nullptr)
            IE_THROW() << "Cannot get convolution node " << parentNode->getName();

        if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx2) || impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core))
            return false;

        return (dw_conv_input_size + dw_conv_output_size > L3_cache_size / 2);
    };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        if (!isConvolutionNode(graphNodes[i])) continue;

        auto parentConvNode = graphNodes[i];
        if (!isSuitableParentConvolution(parentConvNode)) continue;

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseConvolutionAndDWConvolution_ParentConv);

        auto childConvNode = parentConvNode->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildConvolution(parentConvNode, childConvNode)) continue;

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseConvolutionAndDWConvolution_ChildConv);

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
void GraphOptimizer::FuseConvolutionAndSimpleOperationThroughMaxPool(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr node) {
        return (node->getType() == Type::Convolution || node->getType() == Type::BinaryConvolution) && node->getChildEdges().size() == 1 &&
               node->getOriginalOutputPrecisionAtPort(0) == Precision::FP32;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseConvolutionAndSimpleOperationThroughMaxPool_ParentNode);

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (childNode->getAlgorithm() != Algorithm::PoolingMax || childNode->getChildEdges().size() != 1) {
            parent++;
            continue;
        }

        auto fuseCandidate = childNode->getChildEdgeAt(0)->getChild();
        if (parentNode->getType() == Type::BinaryConvolution && !parentNode->canFuse(fuseCandidate)) {
            parent++;
            continue;
        }

        if (!one_of(fuseCandidate->getAlgorithm(), Algorithm::EltwiseRelu,
                                                   Algorithm::EltwiseGelu,
                                                   Algorithm::EltwiseElu,
                                                   Algorithm::EltwiseSigmoid,
                                                   Algorithm::EltwiseClamp,
                                                   Algorithm::EltwiseTanh,
                                                   Algorithm::EltwiseSwish,
                                                   Algorithm::EltwiseHswish,
                                                   Algorithm::EltwiseMish,
                                                   Algorithm::EltwiseHsigmoid,
                                                   Algorithm::EltwiseRoundHalfToEven,
                                                   Algorithm::EltwiseRoundHalfAwayFromZero,
                                                   Algorithm::EltwiseAbs,
                                                   Algorithm::EltwiseSqrt,
                                                   Algorithm::EltwiseSoftRelu)) {
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

void GraphOptimizer::FuseConvolutionAndSimpleOperation(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr node) {
        return (node->getType() == Type::Convolution || node->getType() == Type::BinaryConvolution) && node->getChildEdges().size() == 1;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseConvolutionAndSimpleOperation_ParentNode);

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

        if (childNode->getType() == Type::FakeQuantize || childNode->getType() == Type::Eltwise) {
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

void GraphOptimizer::FusePoolingAndFakeQuantize(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr node) {
        if (node->getType() == Type::Pooling) {
            if (!one_of(node->getOriginalInputPrecisionAtPort(0), Precision::U8, Precision::I8))
                return false;
            return node->getChildEdges().size() == 1 && node->getAlgorithm() == Algorithm::PoolingAvg;
        }
        return false;
    };

    auto isSuitableChildNode = [](NodePtr node) {
        return node->getType() == Type::FakeQuantize && node->getAlgorithm() != Algorithm::FQBinarization;
    };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSuitableParentNode(parent)) continue;

        CPU_GRAPH_OPTIMIZER_SCOPE(FusePoolingAndFakeQuantize_ParentNode);

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildNode(child)) continue;

        CPU_GRAPH_OPTIMIZER_SCOPE(FusePoolingAndFakeQuantize_ChildNode);

        child->fuseInto(parent);

        auto parents = child->parentEdges;
        for (size_t i = 0; i < parents.size(); i++) {
            auto p_edge = parents[i].lock();
            if (p_edge->getParent()->getType() == Type::Pooling)
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
static bool is_data_dependency(const std::shared_ptr<Node> &parent,
                               const std::shared_ptr<Node> &child) {
    std::set<Node*> visited;
    std::list<Node*> nextLayers {parent.get()};

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

void GraphOptimizer::FuseConvolutionSumAndConvolutionSumActivation(Graph &graph) {
    auto &graphNodes = graph.GetNodes();

    auto isFusingSupported = [&](NodePtr conv, NodePtr child) {
        return child->getType() == Type::Eltwise &&
                one_of(child->getAlgorithm(), Algorithm::EltwiseRelu,
                                              Algorithm::EltwiseElu,
                                              Algorithm::EltwiseSigmoid,
                                              Algorithm::EltwiseClamp,
                                              Algorithm::EltwiseSwish,
                                              Algorithm::EltwiseHswish,
                                              Algorithm::EltwiseMish,
                                              Algorithm::EltwiseHsigmoid,
                                              Algorithm::EltwiseRoundHalfToEven,
                                              Algorithm::EltwiseRoundHalfAwayFromZero,
                                              Algorithm::EltwiseSoftRelu);
    };

    for (auto &graphNode : graphNodes) {
        const auto eltwiseNode = std::dynamic_pointer_cast<Eltwise>(graphNode);
        if (graphNode->getType() != Type::Eltwise || graphNode->getAlgorithm() != Algorithm::EltwiseAdd ||
            !eltwiseNode || eltwiseNode->isWithBroadcast())
            continue;

        // TODO: Enlarge to several inputs
        bool isSuitableNode = graphNode->getParentEdges().size() == 2;
        if (!isSuitableNode)
            continue;

        auto parent1 = graphNode->getParentEdgesAtPort(0)[0]->getParent();
        auto parent2 = graphNode->getParentEdgesAtPort(1)[0]->getParent();

        bool isSuitableParent1 = parent1->getType() == Type::Convolution
                                    || parent1->getType() == Type::BinaryConvolution;
        bool isSuitableParent2 = parent2->getType() == Type::Convolution
                                    || parent2->getType() == Type::BinaryConvolution;

        auto canFuseSum = [](node::BinaryConvolution *binConv, NodePtr fuseCandidate) {
            if (binConv->getImplType() == impl_desc_type::ref)
                return false;

            if (binConv->isFusedWith(Type::FakeQuantize))
                return false;

            if (fuseCandidate->getAlgorithm() == Algorithm::EltwiseAdd) {
                for (auto& fusedNode : binConv->fusedWith) {
                    const auto eltwise = std::dynamic_pointer_cast<Eltwise>(fusedNode);
                    if (eltwise && eltwise->isSpecialConvolutionAddFusing()) {
                        return false;
                    }
                }
                return true;
            }
            return false;
        };

        auto* binConvNode1 = dynamic_cast<node::BinaryConvolution *>(parent1.get());
        if (binConvNode1) {
            isSuitableParent1 = isSuitableParent1 && canFuseSum(binConvNode1, graphNode);
        }

        auto* binConvNode2 = dynamic_cast<node::BinaryConvolution *>(parent2.get());
        if (binConvNode2) {
            isSuitableParent2 = isSuitableParent2 && canFuseSum(binConvNode2, graphNode);
        }

        auto checkFusedWithSum = [](Convolution* conv) -> bool {
            for (const auto& node : conv->getFusedWith()) {
                const auto eltwise = std::dynamic_pointer_cast<Eltwise>(node);
                if (eltwise && eltwise->isSpecialConvolutionAddFusing())
                    return true;
            }
            return false;
        };

        auto* convNode1 = dynamic_cast<Convolution *>(parent1.get());
        if (convNode1) {
            if (!convNode1->canBeExecutedInInt8()) {
                isSuitableParent1 = isSuitableParent1 && convNode1->getFusedWith().empty();
            } else {
                isSuitableParent1 = isSuitableParent1 && !checkFusedWithSum(convNode1);
            }
        }

        auto* convNode2 = dynamic_cast<Convolution *>(parent2.get());
        if (convNode2) {
            if (!convNode2->canBeExecutedInInt8()) {
                isSuitableParent2 = isSuitableParent2 && convNode2->getFusedWith().empty();
            } else {
                isSuitableParent2 = isSuitableParent2 && !checkFusedWithSum(convNode2);
            }
        }

        if (!isSuitableParent1 && !isSuitableParent2)
            continue;

        std::shared_ptr<Node> mergedConv;
        std::shared_ptr<Node> peerNode;

        if (isSuitableParent1 && isSuitableParent2) {
            // not merged operation (peerNode) has to be in low precision
            const auto isBranchQuantized = [](const NodePtr& branchParent) {
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
                mergedConv = parent1;
                peerNode = parent2;
            }
        } else {
            mergedConv = isSuitableParent1 ? parent1 : parent2;
            peerNode = isSuitableParent1 ? parent2 : parent1;
        }

        if (isSuitableParent1 && isSuitableParent2) {
            if ((peerNode->getType() == Type::Convolution || peerNode->getType() == Type::BinaryConvolution) &&
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
           (mergedConv->fusedWith[0]->getType() == Type::Convolution || mergedConv->fusedWith[0]->getType() == Type::BinaryConvolution)) {
            // Merged with DW_conv. Shape may change
            mergedConv->inputShapes.push_back(mergedConv->fusedWith[0]->getOutputShapeAtPort(0));
        } else {
            size_t secondTermPort = sum->getFusingPort() == 0 ? 1 : 0;
            mergedConv->inputShapes.push_back(sum->getInputShapeAtPort(secondTermPort));
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
        auto* mergedConvNode = dynamic_cast<Convolution*>(mergedConv.get());
        if (mergedConvNode != nullptr)
            childPort = mergedConvNode->getParentEdges().size();

        auto* mergedBinConvNode = dynamic_cast<node::BinaryConvolution*>(mergedConv.get());
        if (mergedBinConvNode != nullptr)
            childPort = mergedBinConvNode->getParentEdges().size();

        EdgePtr edgePtr(new Edge(peerNode, mergedConv, peer_port, childPort));
        graph.GetEdges().push_back(edgePtr);

        mergedConv->addEdge(edgePtr);

        std::vector<EdgeWeakPtr> edges_to_reconnect = lastNode->getChildEdges();
        for (auto &edge_w : edges_to_reconnect) {
            auto edge = edge_w.lock();
            auto child = edge->getChild();
            int idxParent = edge->getInputNum();
            int idxChild = edge->getOutputNum();

            // reconnect after  activation/sum. Port index must be 0
            IE_ASSERT(idxParent == 0);

            edge->drop();

            EdgePtr newEdge(new Edge(mergedConv, child, idxParent, idxChild));
            graph.GetEdges().push_back(newEdge);
            child->addEdge(newEdge);
        }

        if (lastNode != sum) {
            lastNode->remove();
        }
        sum->remove();
    }
}

void GraphOptimizer::FuseMVNAndSimpleOperation(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr node) {
        return (node->getType() == Type::MVN) && (node->getChildEdges().size() == 1);
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseMVNAndSimpleOperation_ParentNode);

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == Type::FakeQuantize || childNode->getType() == Type::Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Type::MVN)
                    continue;

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseInterpolateAndSimpleOperation(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr node) {
        return node->getType() == Type::Interpolate && node->getChildEdges().size() == 1;
    };

    auto isSuitableChildNode = [&](NodePtr parentNode, NodePtr childNode) {
        // Avoid cycle dependencies
        for (auto &childParentEdge : childNode->getParentEdges()) {
            for (auto &parentParentEdge : parentNode->getParentEdges()) {
                if (childParentEdge.lock()->getParent() == parentParentEdge.lock()->getParent())
                    return false;
            }
        }
        if (!childNode->getFusedWith().empty())
            return false;
        auto interpolateNode = dynamic_cast<Interpolate*>(parentNode.get());
        if (!interpolateNode) {
            IE_THROW() << "Cannot cast " << parentNode->getName() << " to Interpolate";
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

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseInterpolateAndSimpleOperation_ParentNode);

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseInterpolateAndSimpleOperation_ChildNode);

        childNode->fuseInto(parentNode);

        if (childNode->getType() == Type::FakeQuantize || childNode->getType() == Type::Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Type::Interpolate)
                    continue;

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseNormalizeL2AndSimpleOperation(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr node) {
        return node->getType() == Type::NormalizeL2 && node->getChildEdges().size() == 1;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseNormalizeL2AndSimpleOperation_ParentNode);

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == Type::FakeQuantize || childNode->getType() == Type::Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Type::NormalizeL2)
                    continue;

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseReduceAndSimpleOperation(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr node) {
        return node->getType() == Type::Reduce && node->getChildEdges().size() == 1;
    };

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        auto parentNode = *parent;
        if (!isSuitableParentNode(parentNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseReduceAndSimpleOperation_ParentNode);

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!parentNode->canFuse(childNode)) {
            parent++;
            continue;
        }

        childNode->fuseInto(parentNode);

        if (childNode->getType() == Type::FakeQuantize || childNode->getType() == Type::Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge == nullptr)
                    IE_THROW() << "Cannot get parent edge " << childNode->getName();
                if (p_edge->getParent()->getType() == Type::Reduce)
                    continue;

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseEltwiseAndSimple(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr node) {
        return node->getType() == Type::Eltwise && node->getChildEdges().size() == 1;
    };

    auto isSuitableChildNode = [&](NodePtr parentNode, NodePtr childNode) {
        if (parentNode->isConstant() && !childNode->isConstant())
            return false;
        for (auto &childParentEdge : childNode->getParentEdges()) {
            // WA to prevent unsupported reorder exception issue in some cases
            if (childParentEdge.lock()->getParent()->getType() == Type::Split) {
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

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseEltwiseAndSimple_ParentNode);

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();

        if ((parentNode->isDynamicNode() && !childNode->isDynamicNode()) || (!parentNode->isDynamicNode() && childNode->isDynamicNode())) {
            parent++;
            continue;
        }

        if (!isSuitableChildNode(parentNode, childNode)) {
            parent++;
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseEltwiseAndSimple_ChildNode);

        childNode->fuseInto(parentNode);

        if (childNode->getType() == Type::FakeQuantize) {
            auto parentEdges = childNode->parentEdges;
            for (auto &parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Type::Eltwise)
                    continue;

                graph.RemoveEdge(p_edge);
            }

            graph.DropNode(childNode);
        } else if (childNode->getType() == Type::Eltwise) {
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

                        EdgePtr &remEdge = p_edge;
                        int inNum = 0;
                        if (remEdge) {
                            inNum = remEdge->getInputNum();
                            graph.RemoveEdge(remEdge);
                        }
                        remEdge = children[j].lock();
                        int outNum = 0;
                        if (remEdge) {
                            outNum = remEdge->getOutputNum();
                            graph.RemoveEdge(remEdge);
                        }
                        EdgePtr newEdge(new Edge(parent, child, inNum, outNum));
                        auto &graphEdges = graph.GetEdges();
                        graphEdges.push_back(newEdge);
                        parent->addEdge(newEdge);

                        parent->outputShapes[inNum] = child->inputShapes[outNum];
                    }
                } else {
                    EdgePtr &remEdge = p_edge;
                    int inNum = 0;
                    int outNum = parentNode->getParentEdges().size();
                    if (remEdge) {
                        inNum = remEdge->getInputNum();
                        // Need to keep order for these algorithms
                        if (childNode->getAlgorithm() == Algorithm::EltwiseMulAdd ||
                            childNode->getAlgorithm() == Algorithm::EltwiseSelect) {
                            outNum = initialParentInNum + remEdge->getOutputNum() - 1;
                        }
                        graph.RemoveEdge(remEdge);
                    }

                    EdgePtr newEdge(new Edge(parent, parentNode, inNum, outNum));
                    auto &graphEdges = graph.GetEdges();
                    graphEdges.push_back(newEdge);
                    parent->addEdge(newEdge);

                    parentNode->inputShapes.push_back(parent->getOutputShapeAtPort(inNum));
                }
            }

            graph.DropNode(childNode);
        } else {
            graph.DropNode(childNode);
        }
    }
}

void GraphOptimizer::DropDoubleReorders(Graph &graph) {
    std::set<NodePtr> processed;
    std::size_t graphNodesSize = graph.GetNodes().size();
    for (std::size_t i = 0; i < graphNodesSize; i++) {
        NodePtr& node = graph.GetNodes()[i];
        if (processed.find(node) == processed.end() && node->getType() == Type::Reorder
            && node->getChildEdges().size() == 1
            && node->getChildEdgeAt(0)->getChild()->getType() == Type::Reorder ) {
            auto nextNode = node->getChildEdgeAt(0)->getChild();
            Reorder* n = dynamic_cast<Reorder*>(node.get());
            if (n == nullptr)
                IE_THROW() << "Cannot get reorder layer " << node->getName();
            Reorder* nn = dynamic_cast<Reorder*>(nextNode.get());
            if (nn == nullptr)
                IE_THROW() << "Cannot get reorder layer " << nextNode->getName();

            NodePtr p = n->getParentEdgesAtPort(0)[0]->getParent();
            NodePtr c = nn->getChildEdgesAtPort(0)[0]->getChild();

            auto oldEdgeNum = n->getParentEdgesAtPort(0)[0]->getInputNum();

            graph.DropNode(node);
            graph.DropNode(nextNode);

            processed.insert(node);
            processed.insert(nextNode);

            EdgePtr edge;
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

void GraphOptimizer::FuseBroadcastAndEltwise(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    for (auto &graphNode : graphNodes) {
        if (graphNode->getType() != Type::Generic
                || graphNode->getTypeStr() != "Broadcast"
                || graphNode->getChildEdges().size() != 1lu
                || graphNode->getChildEdgeAt(0)->getChild()->getType() != Type::Eltwise)
            continue;

        NodePtr& broadcastNode = graphNode;
        NodePtr eltwiseNode = broadcastNode->getChildEdgeAt(0)->getChild();
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

void GraphOptimizer::FuseClampAndFakeQuantize(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableClampNode = [](NodePtr node) {
        return node->getType() == Type::Eltwise && node->getChildEdges().size() == 1 && node->getAlgorithm() == Algorithm::EltwiseClamp;
    };

    auto isSuitableFakeQuantizeNode = [](NodePtr node) {
        return node->getType() == Type::FakeQuantize && node->getAlgorithm() != Algorithm::FQBinarization;
    };

    auto fuseClampAndFakeQuantizeNodes = [](NodePtr parent, NodePtr child) {
        auto* eltwiseNode = dynamic_cast<Eltwise *>(parent.get());
        if (eltwiseNode == nullptr)
            IE_THROW() << "Cannot cast " << parent->getName() << " to Eltwise node";

        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(child.get());
        if (fakeQuantizeNode == nullptr)
            IE_THROW() << "Cannot cast " << child->getName() << " to FakeQuantize node";

        const std::vector<float>& cropLowData = fakeQuantizeNode->getCropLow();
        const std::vector<float>& cropHighData = fakeQuantizeNode->getCropHigh();

        std::vector<float> newCropLow(cropLowData.size());
        std::vector<float> newCropHigh(cropHighData.size());
        for (size_t i = 0; i < cropLowData.size(); i++)
            newCropLow[i] = std::max(cropLowData[i], eltwiseNode->getAlpha());
        for (size_t i = 0; i < cropHighData.size(); i++)
            newCropHigh[i] = std::min(cropHighData[i], eltwiseNode->getBeta());

        fakeQuantizeNode->setCropLow(newCropLow);
        fakeQuantizeNode->setCropHigh(newCropHigh);

        return true;
    };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSuitableClampNode(parent)) continue;

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseClampAndFakeQuantize_ClalmpNode);

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSuitableFakeQuantizeNode(child)) continue;

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseClampAndFakeQuantize_QuantizeNode);

        if (fuseClampAndFakeQuantizeNodes(parent, child)) {
            graph.DropNode(parent);
        }
    }
}

void GraphOptimizer::FusePerformedAsScaleShiftAndFakeQuantize(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto getNonConstPort = [](const NodePtr& node) {
        std::vector<int> nonConstPorts;
        for (size_t i = 0; i < node->getParentEdges().size(); i++) {
            const auto& parent = node->getParentEdgeAt(i)->getParent();
            if (!(parent->getType() == Type::Input && parent->isConstant()))
                nonConstPorts.push_back(i);
        }
        // there are more than 1 nonconst port or missed
        if (nonConstPorts.size() != 1)
            return -1;

        return nonConstPorts[0];
    };

    auto isSuitableScaleShiftNode = [getNonConstPort](const NodePtr& node) {
        if (!one_of(node->getAlgorithm(), Algorithm::EltwiseAdd,
                                          Algorithm::EltwiseSubtract,
                                          Algorithm::EltwiseMultiply,
                                          Algorithm::EltwiseDivide,
                                          Algorithm::EltwiseMulAdd))
            return false;

        const auto nonConstPort = getNonConstPort(node);
        if (nonConstPort == -1)
            return false;

        const NodePtr eltwiseInput = node->getParentEdgeAt(nonConstPort)->getParent();
        return node->getChildEdges().size() == 1 && node->canBePerformedAsScaleShift(eltwiseInput.get());
    };

    auto isSuitableFakeQuantizeNode = [](const NodePtr& node) {
        return node->getType() == Type::FakeQuantize && node->getAlgorithm() != Algorithm::FQBinarization;
    };

    auto fuseScaleShiftAndFakeQuantizeNodes = [getNonConstPort](const NodePtr& parent, const NodePtr& child) {
        auto fakeQuantizeNode = std::dynamic_pointer_cast<FakeQuantize>(child);
        if (fakeQuantizeNode == nullptr)
            IE_THROW() << "Cannot cast " << child->getName() << " to FakeQuantize node";

        std::vector<float> scalesBuffer;
        std::vector<float> shiftsBuffer;
        auto parentEltwise = std::dynamic_pointer_cast<Eltwise>(parent);
        if (!parentEltwise) {
            IE_THROW() << "Cannot cast " << parent->getName() << " to Eltwise node";
        }

        const NodePtr eltwiseInput = parentEltwise->getParentEdgeAt(getNonConstPort(parent))->getParent();
        std::tie(scalesBuffer, shiftsBuffer) = parentEltwise->getScalesAndShifts(eltwiseInput.get());

        const auto &outputShape = child->getOutputShapeAtPort(0);
        VectorDims outputDims = outputShape.getDims();
        const auto channelPos = parent->getParentEdgeAt(0)->getParent()->getFusingAxis();

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

        for (size_t i = 0; i < scalesBuffer.size(); i++)
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

        for (size_t i = 0; i < newCropLow.size(); i++) {
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

        for (size_t i = 0; i < newInputScale.size(); i++) {
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

        for (size_t i = 0; i < newInputShift.size(); i++) {
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

    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSuitableScaleShiftNode(parent)) continue;

        CPU_GRAPH_OPTIMIZER_SCOPE(FusePerformedAsScaleShiftAndFakeQuantize_ShiftNode);

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSuitableFakeQuantizeNode(child)) continue;

        CPU_GRAPH_OPTIMIZER_SCOPE(FusePerformedAsScaleShiftAndFakeQuantize_QuantizeNode);

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

void GraphOptimizer::MergeTransposeAndReorder(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr node) {
        // WA: to avoid broken memory pointer for conv + sum
        auto prevNodeIsConvSum = [](NodePtr node) -> bool {
            const auto parent = node->getParentEdgesAtPort(0)[0]->getParent();
            if (parent->getType() == Type::Convolution) {
                for (const auto& fusedNode : parent->getFusedWith()) {
                    if (fusedNode->getAlgorithm() == Algorithm::EltwiseAdd) {
                        const auto addNode = std::dynamic_pointer_cast<Eltwise>(fusedNode);
                        if (addNode && addNode->isSpecialConvolutionAddFusing()) {
                            return true;
                        }
                    }
                }
            }
            return false;
        };

        return node->getType() == Type::Transpose
                && node->getChildEdges().size() == 1
                && !node->isDynamicNode() // TODO [DS]: enable for dynamic shapes when inPlace in the dynamic case is available (CVS-74863)
                && !prevNodeIsConvSum(node);
    };

    auto isSuitableChildNode = [](NodePtr node) {
        return node->getType() == Type::Reorder
                && node->getChildEdges().size() == 1
                && !node->isDynamicNode();   // TODO [DS]: enable for dynamic shapes when inPlace in the dynamic case is available (CVS-74863)
    };

    // Method checkAscendingSummaryOrder() checks that after the sequential execution of Transpose and Reorder nodes,
    // the order of the elements in the memory will not change. In other words, that Transpose+Reorder is identical permutation.
    auto checkAscendingSummaryOrder = [](std::shared_ptr<Node> &parentNode, std::shared_ptr<Node> &childNode) -> bool {
        auto* transposeNode = dynamic_cast<Transpose*>(parentNode.get());
        auto* reorderNode = dynamic_cast<Reorder*>(childNode.get());
        if (!transposeNode || !reorderNode) {
            return false;
        }

        auto& transposeOrder = transposeNode->getOrder();
        auto layoutOrder = transposeNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].getMemDesc()->as<BlockedMemoryDesc>()->getOrder();

        auto inBlockedDesc = reorderNode->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].getMemDesc()->as<BlockedMemoryDesc>();
        auto outBlockedDesc = reorderNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].getMemDesc()->as<BlockedMemoryDesc>();

        auto& inOrder = inBlockedDesc->getOrder();
        auto& outOrder = outBlockedDesc->getOrder();

        if (transposeOrder.size() != layoutOrder.size() || layoutOrder.size() != inOrder.size() || inOrder.size() != outOrder.size()) {
            return false;
        }

        // revLayoutOrder - reverse permutation for layoutOrder
        auto revLayoutOrder = VectorDims(layoutOrder.size());
        for (size_t i = 0; i < revLayoutOrder.size(); i++) {
            revLayoutOrder[layoutOrder[i]] = i;
        }

        // newTransposeOrder - Transpose layout-aware permutation
        auto newTransposeOrder = VectorDims(transposeOrder.size());
        for (size_t i = 0; i < newTransposeOrder.size(); i++) {
            newTransposeOrder[i] = layoutOrder[transposeOrder[revLayoutOrder[i]]];
        }

        // reorderOrder - Reorder layout-aware permutation
        auto reorderOrder = VectorDims(outOrder.size());
        for (size_t i = 0; i < reorderOrder.size(); i++) {
            for (size_t j = 0; j < reorderOrder.size(); j++) {
                if (outOrder[i] == inOrder[j]) {
                    reorderOrder[i] = j;
                    continue;
                }
            }
        }

        // summaryOrder - resulting Transpose+Reorder permutation
        auto summaryOrder = VectorDims(transposeOrder.size());
        for (size_t i = 0; i < summaryOrder.size(); i++) {
            summaryOrder[i] = reorderOrder[newTransposeOrder[i]];
        }

        // check that Transpose+Reorder is the identical permutation
        for (size_t i = 0; i < summaryOrder.size(); i++) {
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
    auto mergeTransposeAndReorder = [&](std::shared_ptr<Node>& parentNode, std::shared_ptr<Node>& childNode) {
        auto parentParentNode = parentNode->getParentEdgesAtPort(0)[0]->getParent();
        auto parentParentConstNode = parentNode->getParentEdgesAtPort(1)[0]->getParent();
        auto childChildNode = childNode->getChildEdgeAt(0)->getChild();

        auto remEdge = parentNode->getParentEdgesAtPort(1)[0];
        remEdge->drop();
        auto& edges = graph.GetEdges();
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if ((*it) == remEdge) {
                edges.erase(it);
                if (parentParentConstNode->getChildEdges().empty())
                    parentParentConstNode->remove();
                break;
            }
        }

        graph.DropNode(parentNode);
        graph.DropNode(childNode);

        auto inDesc = parentNode->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].getMemDesc();
        auto outDesc = childNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].getMemDesc();

        auto inPrec = inDesc->getPrecision();
        auto outPrec = outDesc->getPrecision();

        auto reorderInDesc = inDesc;
        auto reorderOutDesc = outDesc->cloneWithNewPrecision(inPrec);

        std::string reorderlayerName = parentParentNode->getName() + "_" +
                Reorder::getReorderArgs(*reorderInDesc, *reorderOutDesc) + "_" + "fake";

        EdgePtr edge;
        for (auto &childEdge : parentParentNode->getChildEdges()) {
            if (childEdge.lock()->getChild() == childChildNode) {
                edge = childEdge.lock();
                break;
            }
        }
        if (!edge) {
            IE_THROW() << "Transpose node '" << parentNode->getName() << "' has invalid edges.";
        }

        bool isOptimized = true;
        std::vector<int> srcPerm;
        auto configReorder = [&]() {
            // transposeNode support blocked input & non-blocked output, in the case, the reorder
            // cannot be optimized
            auto* transposeNode = dynamic_cast<Transpose*>(parentNode.get());
            auto inOrder = transposeNode->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].getMemDesc()->as<BlockedMemoryDesc>()->getOrder();

            if (inOrder.size() > reorderOutDesc->as<BlockedMemoryDesc>()->getOrder().size()) {
                isOptimized = false;
                // inDesc should be permuted before calling reorder
                auto & ord = transposeNode->getOrder();
                srcPerm = std::vector<int>(ord.size());
                for (size_t i = 0; i < ord.size(); i++) {
                    srcPerm[ord[i]] = i;
                }
            }
        };

        configReorder();

        auto reorderNode = graph.InsertReorder(edge, reorderlayerName, *reorderInDesc, *reorderOutDesc, isOptimized, srcPerm);

        // case 2
        if (inPrec != outPrec) {
            auto reorderInDesc2 = reorderOutDesc;
            auto reorderOutDesc2 = outDesc;

            std::string reorderLayerName2 = reorderNode->getName() + "_" +
                                    Reorder::getReorderArgs(*reorderInDesc2, *reorderOutDesc2) + "_" + childChildNode->getName();

            graph.InsertReorder(reorderNode->getChildEdgeAt(0), reorderLayerName2, *reorderInDesc2, *reorderOutDesc2, false);
        }
    };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto parentNode = graphNodes[i];
        if (!isSuitableParentNode(parentNode)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(MergeTransposeAndReorder_ParentNode);

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildNode(childNode)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(MergeTransposeAndReorder_ChildNode);

        if (checkAscendingSummaryOrder(parentNode, childNode)) {
            mergeTransposeAndReorder(parentNode, childNode);
        }
    }
}

void GraphOptimizer::reshapeRnnSeq(Graph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr node) {
        if (node->type != Type::RNNSeq)
            return false;
        auto rnnNode = std::dynamic_pointer_cast<RNN>(node);
        return rnnNode && !rnnNode->hasNativeOrder() && node->outputShapes[0].getRank() == 4 && node->outputShapes[0].getDims()[1] == 1;
    };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto parentNode = graphNodes[i];
        if (!isSuitableParentNode(parentNode)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(reshapeRnnSeq_ParentNode);

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

            const auto cpuUnsqueeze = std::make_shared<Reshape>(unsqueeze, graph.getGraphContext());
            graph.InsertNode(parentNode, childNode, cpuUnsqueeze, edge->getInputNum(), edge->getOutputNum(), false);

            const auto cpuConstant = std::make_shared<node::Input>(secondInput, graph.getGraphContext());
            EdgePtr newEdge(new Edge(cpuConstant, cpuUnsqueeze, 0, 1));
            cpuUnsqueeze->addEdge(newEdge);
            auto &graphEdges = graph.GetEdges();
            graphEdges.push_back(newEdge);
            graphNodes.push_back(cpuConstant);

            graph.RemoveEdge(edge);
        }
    }
}

/*
Remove Redundant Convert Node
Example: BF16 model output is forced by post-procesing API
Node [FP32] -> Convert[BF16] -> Outputs[BF16]
After EnforceBF16 routine the subgraph becomes:
Node [BF16] -> Convert [BF16] -> Outputs [BF16]
So Convert is redundant."
*/

void GraphOptimizer::RemoveSameConvert(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](NodePtr parentNode) {
        return parentNode->getType() == Type::Convert &&
               (parentNode->getOriginalOutputPrecisionAtPort(0) == parentNode->getOriginalInputPrecisionAtPort(0));
    };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto parentNode = graphNodes[i];
        if (!isSuitableParentNode(parentNode)) {
            continue;
        }
        graph.DropNode(parentNode);
    }
}

}   // namespace intel_cpu
}   // namespace ov
