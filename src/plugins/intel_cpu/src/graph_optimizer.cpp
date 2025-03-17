// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_optimizer.h"

#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "nodes/bin_conv.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/conv.h"
#include "nodes/deconv.h"
#include "nodes/eltwise.h"
#include "nodes/fake_quantize.h"
#include "nodes/fullyconnected.h"
#include "nodes/input.h"
#include "nodes/interpolate.h"
#include "nodes/memory.hpp"
#include "nodes/reorder.h"
#include "nodes/reshape.h"
#include "nodes/rnn.h"
#include "nodes/scaled_attn.h"
#include "nodes/transpose.h"
#include "onednn/dnnl.h"
#include "openvino/opsets/opset1.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

// WA for xbyak.h
#ifdef _WIN32
#    ifndef _WINSOCKAPI_
#        define _WINSOCKAPI_
#    endif
#    ifndef _WINSOCK2API_
#        define _WINSOCK2API_
#    endif
#endif
#include <algorithm>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "itt.h"
#include "memory_desc/cpu_memory_desc_utils.h"

using namespace dnnl;
using namespace ov::intel_cpu::node;

// Many GraphOptimizer passes are changing graphNodes collection while iterating which is not completely safe. So, it is
// impossible to use range based for loops.
// NOLINTBEGIN(modernize-loop-convert)

namespace ov::intel_cpu {

GraphOptimizer::GraphOptimizer() = default;

void GraphOptimizer::ApplyCommonGraphOptimizations(Graph& graph) {
    // For conv with input zp, canBeExecutedInInt8() check has dependency on input zero point check.
    // Also zero point node is the input of computing-intensive nodes. Most others fusing are the output of
    // computing-intensive nodes. So Locate the FuseConvolutionAndZeroPoints() as the first optimization.
    OV_ITT_SCOPE_CHAIN(FIRST_INFERENCE,
                       taskChain,
                       itt::domains::intel_cpu_LT,
                       "ApplyCommonGraphOptimizations",
                       "FuseConvolutionAndZeroPoints");
    FuseConvolutionAndZeroPoints(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseConvMatmulFCDeconvAndDQScales");
    FuseConvMatmulFCDeconvAndDQScales(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseConvolutionAndBias");
    FuseConvolutionMatMulDeconvAndBias(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseMultiplyAndAdd");
    FuseMultiplyAndAdd(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "MergeConvertAndEltwise");
    MergeConvertAndEltwise(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseFCAndConvertOnWeights");
    FuseFCAndConvertOnWeights(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseFCAndTransposeOnWeights");
    FuseFCAndTransposeOnWeights(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseDeconvolutionAndSimpleOperation");
    FuseDeconvolutionAndSimpleOperation(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FuseClampAndFakeQuantize");
    FuseClampAndFakeQuantize(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FusePerformedAsScaleShiftAndFakeQuantize");
    FusePerformedAsScaleShiftAndFakeQuantize(graph);
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

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "MergeEltwiseAndConvert");
    MergeEltwiseAndConvert(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "reshapeRnnSeq");
    reshapeRnnSeq(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "RemoveSameConvert");
    RemoveSameConvert(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "RemoveMemoryInputConvert");
    RemoveMemoryInputConvert(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "RemoveConvertMemoryOutput");
    RemoveConvertMemoryOutput(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "MatchSdpaKvCache");
    MatchSdpaKvCache(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "DropRedundantMemoryOutput");
    DropRedundantMemoryOutput(graph);
    graph.RemoveDroppedNodes();

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "RemoveDroppedEdges");
    graph.RemoveDroppedEdges();
}

void GraphOptimizer::ApplyImplSpecificGraphOptimizations(Graph& graph) {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "GraphOptimizer::ApplyImplSpecificGraphOptimizations");

    DropDoubleReorders(graph);
    graph.RemoveDroppedNodes();

    MergeTransposeAndReorder(graph);
    graph.RemoveDroppedNodes();

    MergeReorderAndTranspose(graph);
    graph.RemoveDroppedNodes();

    graph.RemoveDroppedEdges();
}

void GraphOptimizer::FuseConvMatmulFCDeconvAndDQScales(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isDQScaleGraphPattern = [](const NodePtr& node) {
        if (node->getType() != Type::Eltwise || node->getAlgorithm() != Algorithm::EltwiseMultiply) {
            return false;
        }
        auto parentNode = node->getParentEdgeAt(0)->getParent();
        auto scaleNode = node->getParentEdgeAt(1)->getParent();
        if (!(parentNode->getType() == Type::Convolution || parentNode->getType() == Type::MatMul ||
              parentNode->getType() == Type::Deconvolution)) {
            return false;
        }
        if (!scaleNode->isConstant()) {
            return false;
        }
        // Only Fusing scales for INT8 precision.
        if (!parentNode->canBeExecutedInInt8()) {
            return false;
        }
        return (parentNode->getParentEdges().size() == 2);
    };

    auto scaleDimsCheck = [](const NodePtr& node, const NodePtr& scales) {
        const auto nodeOutDims = node->getOutputShapeAtPort(0).getDims();
        const auto channelAxis = node->getFusingAxis();
        auto OC = nodeOutDims[channelAxis];

        if (Shape::UNDEFINED_DIM == OC) {
            return false;
        }
        if (!node->getFusedWith().empty() || !scales->getFusedWith().empty()) {
            return false;
        }

        const auto scalesDims = getNormalizedDimsBySize(scales->getOutputShapeAtPort(0).getDims(), nodeOutDims.size());
        if (nodeOutDims.size() != scalesDims.size() || scalesDims.size() < 2) {
            return false;
        }

        if (!dimsEqualStrong(scalesDims[channelAxis], nodeOutDims[channelAxis]) && scalesDims[channelAxis] != 1) {
            return false;
        }

        for (size_t i = 0; i < scalesDims.size(); i++) {
            if (scalesDims[i] != 1 && static_cast<int>(i) != channelAxis) {
                return false;
            }
        }
        return true;
    };

    auto initializeDeQuantizedScales = [](const NodePtr& node, const NodePtr& scales) {
        auto scalesConstant = dynamic_cast<node::Input*>(scales.get());
        if (scalesConstant == nullptr) {
            OPENVINO_THROW("Cannot cast to Input node");
        }

        auto scalesBlob = scalesConstant->getMemoryPtr();
        if (scalesBlob == nullptr) {
            OPENVINO_THROW("Cannot cast to TBlob internal scales blob");
        }

        auto scalesData = static_cast<const float*>(scalesBlob->getData());
        if (scalesData == nullptr) {
            OPENVINO_THROW("scalesBlob has not allocated buffer");
        }
        auto scalesDims = getNormalizedDimsBySize(scales->getOutputShapeAtPort(0).getDims(),
                                                  node->getOutputShapeAtPort(0).getDims().size());
        auto scaleSize = std::accumulate(scalesDims.begin(), scalesDims.end(), 1, std::multiplies<>());
        node->fuseDQScales(scalesData, scaleSize);
        return true;
    };

    for (const auto& mul : graphNodes) {
        if (!isDQScaleGraphPattern(mul)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseConvMatmulFCDeconvAndDQScales);

        auto node = mul->getParentEdgeAt(0)->getParent();
        auto scales = mul->getParentEdgeAt(1)->getParent();
        if (!scaleDimsCheck(node, scales)) {
            continue;
        }

        if (initializeDeQuantizedScales(node, scales)) {
            DEBUG_LOG("GraphOptimizer##FusingDQ: Node ##",
                      mul->getName(),
                      " optimized as DQ scales of Node ##",
                      node->getName());
            node->addOriginalLayer(mul->getOriginalLayers());
            auto p_edge = mul->getParentEdgeAt(1);
            graph.RemoveEdge(p_edge);
            graph.DropNode(mul);
        }
    }
}

void GraphOptimizer::FuseConvolutionMatMulDeconvAndBias(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const NodePtr& node) {
        const auto deconv = std::dynamic_pointer_cast<Deconvolution>(node);
        // bias should be the first child
        if (!node->getFusedWith().empty()) {
            return false;
        }
        // no other child other than bias-add
        if (node->getChildEdges().size() != 1) {
            return false;
        }

        if (!deconv) {
            return (one_of(node->getType(), Type::Convolution, Type::MatMul) && node->getParentEdges().size() == 2);
        }
        return deconv->canFuseBias();
    };

    auto isSuitableChildNode = [&](const NodePtr& parentNode, const NodePtr& childNode) {
        if (childNode->getAlgorithm() != Algorithm::EltwiseAdd || !childNode->getFusedWith().empty() ||
            childNode->getParentEdges().size() != 2) {
            return false;
        }

        auto biasPort = childNode->getParentEdgeAt(0)->getParent() == parentNode ? 1 : 0;
        const auto biasNode = childNode->getParentEdgeAt(biasPort)->getParent();
        if (biasNode->getType() != Type::Input || !biasNode->isConstant() || biasNode->getChildEdges().size() != 1) {
            return false;
        }

        const auto parentOutDims = parentNode->getOutputShapeAtPort(0).getDims();
        const auto biasDims =
            getNormalizedDimsBySize(biasNode->getOutputShapeAtPort(0).getDims(), parentOutDims.size());
        // TODO [NM]: Legacy ConvBias fusion transformation supports both per-tensor (via explicit broadcasing) and
        // per-channel cases. Most of the real models contain per-channel bias, so we need to reavaluate the need to
        // support per-tensor variant.
        if (parentOutDims.size() != biasDims.size() || biasDims.size() < 2) {
            return false;
        }

        const auto channelAxis = parentNode->getFusingAxis();
        if (!dimsEqualStrong(biasDims[channelAxis], parentOutDims[channelAxis])) {
            return false;
        }

        for (size_t i = 0; i < biasDims.size(); i++) {
            if (biasDims[i] != 1 && static_cast<int>(i) != channelAxis) {
                return false;
            }
        }

        return true;
    };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto parentNode = graphNodes[i];
        if (!isSuitableParentNode(parentNode)) {
            continue;
        }
        CPU_GRAPH_OPTIMIZER_SCOPE(FuseConvolutionMatMulDeconvAndBias_ParentNode);

        auto childNode = parentNode->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildNode(parentNode, childNode)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseConvolutionMatMulDeconvAndBias_ChildNode);

        auto childs = childNode->childEdges;
        auto parents = childNode->parentEdges;
        const auto biasPort = childNode->getParentEdgeAt(0)->getParent() == parentNode ? 1 : 0;

        for (const auto& i : parents) {
            auto p_edge = i.lock();
            if (!p_edge) {
                continue;
            }
            auto parent = p_edge->getParent();
            if (!parent) {
                continue;
            }

            if (parent == parentNode) {
                for (const auto& j : childs) {
                    if (!j.lock()) {
                        continue;
                    }
                    auto child = j.lock()->getChild();
                    if (!child) {
                        continue;
                    }

                    EdgePtr& remEdge = p_edge;
                    int inNum = 0;
                    if (remEdge) {
                        inNum = remEdge->getInputNum();
                        graph.RemoveEdge(remEdge);
                    }
                    remEdge = j.lock();
                    int outNum = 0;
                    if (remEdge) {
                        outNum = remEdge->getOutputNum();
                        graph.RemoveEdge(remEdge);
                    }
                    graph.CreateEdge(parent, child, inNum, outNum);
                }
            } else {
                EdgePtr& remEdge = p_edge;
                int inNum = 0;
                if (remEdge) {
                    inNum = remEdge->getInputNum();
                    graph.RemoveEdge(remEdge);
                }

                auto& targetNode = parentNode;
                const auto& biasNode = parent;
                auto biasOutputShape = biasNode->getOutputShapeAtPort(0);
                int outNum = targetNode->getParentEdges().size();
                // ONEDNN Conv, Deconv, FC would need the bias to be flatten into 1D tensor.
                // Usually the bias output shape would be normalized to align rank with Conv/Deconv/FC output.
                // To avoid duplicate reshape WA code in nodes, here we flatten the shape.
                // Most bias nodes are const Input and bias memory primitive has been initialized as const memory when
                // constructing CPU Input node. Const memory is not allowed to be modified after initialized. It means
                // we can't redefine const bias memory primitive. So let's insert a reshape node to flatten the bias
                // shape into 1D and const folding node will be executed during the compiling stage.
                const bool needReshape = (targetNode->getType() != Type::MatMul && biasOutputShape.getRank() != 1);
                if (needReshape) {
                    // Bias -> Reshape -> Conv/Deconv/FC
                    const VectorDims flattenShape = {biasOutputShape.getElementsCount()};
                    // Construct Ngraph Reshape node and CPU Reshape node.
                    auto reshapeConstInput =
                        std::make_shared<ov::opset1::Constant>(ov::element::i32, ov::Shape{1}, flattenShape);
                    auto reshapeDummyInput =
                        std::make_shared<ov::opset1::Parameter>(biasNode->getOriginalOutputPrecisionAtPort(0),
                                                                biasOutputShape.toPartialShape());
                    const auto reshape =
                        std::make_shared<ov::opset1::Reshape>(reshapeDummyInput, reshapeConstInput, false);
                    reshape->set_friendly_name(biasNode->getName() + "_flatten_reshape");
                    const auto cpuReshapeNode =
                        std::make_shared<ov::intel_cpu::node::Reshape>(reshape, graph.getGraphContext());
                    // Insert Reshape between bias node and Conv/Deconv/FC
                    graph.InsertNode(biasNode, targetNode, cpuReshapeNode, inNum, outNum, false);
                    // Insert the Reshape const input node and edge into CPU graph.
                    const auto cpuReshapeConstInput =
                        std::make_shared<node::Input>(reshapeConstInput, graph.getGraphContext());
                    graph.AddNode(cpuReshapeConstInput);
                    graph.CreateEdge(cpuReshapeConstInput, cpuReshapeNode, 0, 1);
                    DEBUG_LOG("GraphOptimizer##FusingBias:Flatten Bias node from shape ",
                              PartialShape{biasOutputShape.getDims()},
                              "  to  ",
                              PartialShape{flattenShape});
                    // Update bias output shape to be flatten shape.
                    biasOutputShape = Shape{flattenShape};
                } else {
                    // Bias is connected as input edge.
                    graph.CreateEdge(biasNode, targetNode, inNum, outNum);
                }
                // Add the Bias inputshape into conv/FC/Deconv/Matmul.
                targetNode->inputShapes.push_back(biasOutputShape);
            }
        }
        DEBUG_LOG("GraphOptimizer##FusingBias:Node ##: ",
                  childNode->getName(),
                  " initialize as Bias of Node ##",
                  parentNode->getName());
        parentNode->addOriginalLayer(childNode->getOriginalLayers());
        parentNode->addOriginalInputPrecision(childNode->getOriginalInputPrecisionAtPort(biasPort));
        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseDeconvolutionAndSimpleOperation(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const NodePtr& node) {
        if (node->getType() != Type::Deconvolution || node->getChildEdges().size() != 1) {
            return false;
        }
        const auto deconv = std::dynamic_pointer_cast<Deconvolution>(node);
        if (deconv == nullptr) {
            OPENVINO_THROW("Cannot cast to deconvolution node ", node->getName());
        }

        if (deconv->getAlgorithm() != Algorithm::DeconvolutionCommon) {
            return true;
        }

        const auto& strides = deconv->getStride();
        const auto& kernel = deconv->getWeightDims();
        // WA oneDNN doesn't support fusing post ops after deconvolution with strides over kernel size
        bool isSupportedParams = strides[strides.size() - 1] <= static_cast<dnnl_dim_t>(kernel[kernel.size() - 1]);
        if (strides.size() > 1) {
            isSupportedParams &= strides[strides.size() - 2] <= static_cast<dnnl_dim_t>(kernel[kernel.size() - 2]);
        }
        if (strides.size() > 2) {
            isSupportedParams &= strides[strides.size() - 3] <= static_cast<dnnl_dim_t>(kernel[kernel.size() - 3]);
        }
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
        for (auto& parentEdge : parentEdges) {
            auto p_edge = parentEdge.lock();
            if (p_edge->getParent()->getType() == Type::Deconvolution) {
                continue;
            }

            graph.RemoveEdge(p_edge);
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseMultiplyAndAdd(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableSecondInput = [](const NodePtr& node, VectorDims dataDims) {
        if (node->getType() != Type::Input || !node->isConstant()) {
            return false;
        }
        const auto secondInputDims = node->getOutputShapeAtPort(0).getStaticDims();
        if (secondInputDims.size() != dataDims.size() || secondInputDims.size() < 2) {
            return false;
        }

        auto getChannelAxis = [](const VectorDims& dims) {
            auto channelAxis = -1;
            for (size_t i = 0; i < dims.size(); i++) {
                if (dims[i] != 1) {
                    if (channelAxis != -1) {  // more than one axis is != 1
                        return -1;
                    }
                    channelAxis = i;
                }
            }
            return channelAxis;
        };

        const auto channelAxis = getChannelAxis(secondInputDims);
        if (channelAxis == -1) {
            return false;
        }

        if (secondInputDims[0] != 1 || !dimsEqualWeak(secondInputDims[channelAxis], dataDims[channelAxis])) {
            return false;
        }

        return true;
    };

    auto isSuitableParentNode = [&](const NodePtr& node) {
        if (node->getAlgorithm() != Algorithm::EltwiseMultiply || !node->getFusedWith().empty() ||
            node->getParentEdges().size() != 2 || node->getChildEdges().size() != 1) {
            return false;
        }

        return isSuitableSecondInput(node->getParentEdgeAt(1)->getParent(), node->getInputShapeAtPort(0).getDims());
    };

    auto isSuitableChildNode = [&](const NodePtr& parentNode, const NodePtr& childNode) {
        if (childNode->getAlgorithm() != Algorithm::EltwiseAdd || !childNode->getFusedWith().empty() ||
            childNode->getParentEdges().size() != 2) {
            return false;
        }

        return isSuitableSecondInput(childNode->getParentEdgeAt(1)->getParent(),
                                     childNode->getInputShapeAtPort(0).getDims()) &&
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

        for (const auto& i : parents) {
            auto p_edge = i.lock();
            if (!p_edge) {
                continue;
            }
            auto parent = p_edge->getParent();
            if (!parent) {
                continue;
            }

            if (parent == parentNode) {
                for (const auto& j : childs) {
                    if (!j.lock()) {
                        continue;
                    }
                    auto child = j.lock()->getChild();
                    if (!child) {
                        continue;
                    }

                    EdgePtr& remEdge = p_edge;
                    int inNum = 0;
                    if (remEdge) {
                        inNum = remEdge->getInputNum();
                        graph.RemoveEdge(remEdge);
                    }
                    remEdge = j.lock();
                    int outNum = 0;
                    if (remEdge) {
                        outNum = remEdge->getOutputNum();
                        graph.RemoveEdge(remEdge);
                    }
                    graph.CreateEdge(parent, child, inNum, outNum);
                }
            } else {
                EdgePtr& remEdge = p_edge;
                int inNum = 0;
                if (remEdge) {
                    inNum = remEdge->getInputNum();
                    graph.RemoveEdge(remEdge);
                }

                auto& parentEltwise = parentNode;

                parentEltwise->inputShapes.push_back(parent->getOutputShapeAtPort(0));
                graph.CreateEdge(parent, parentEltwise, inNum, parentEltwise->getParentEdges().size());
            }
        }

        parentNode->addOriginalInputPrecision(childNode->getOriginalInputPrecisionAtPort(1));
        parentNode->setAlgorithm(Algorithm::EltwiseMulAdd);
        parentNode->setTypeStr("MulAdd");
        parentNode->addOriginalLayer(childNode->getOriginalLayers());
        graph.DropNode(childNode);
    }
}

void GraphOptimizer::MergeEltwiseAndConvert(Graph& graph) {
// The pass is enabled on arm platforms only, however it might be usefull for other platforms as well
// It requires additional perf validation. Ticket: 163388
#if !defined(OPENVINO_ARCH_ARM64)
    return;
#endif
    auto& graphNodes = graph.GetNodes();

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        CPU_GRAPH_OPTIMIZER_SCOPE(MergeEltwiseAndConvert);
        auto parentNode = *parent;
        if (parentNode->getType() != Type::Eltwise) {
            parent++;
            continue;
        }

        const auto& childEdges = parentNode->getChildEdges();
        if (childEdges.size() != 1) {
            parent++;
            continue;
        }

        const auto edge = childEdges[0].lock();
        auto childNode = edge->getChild();
        if (childNode->getType() != Type::Convert) {
            parent++;
            continue;
        }

        const auto eltwise = dynamic_cast<ov::intel_cpu::node::Eltwise*>(parentNode.get());
        if (!eltwise->canFuseConvert(childNode)) {
            parent++;
            continue;
        }

        // WA: Eltwise node uses precision of last fused node as output precision
        auto fusedOps = parentNode->getFusedWith();
        if (!fusedOps.empty()) {
            fusedOps[fusedOps.size() - 1]->setOriginalOutputPrecisionAtPort(
                0,
                childNode->getOriginalOutputPrecisionAtPort(0));
        }
        parentNode->setOriginalOutputPrecisionAtPort(0, childNode->getOriginalOutputPrecisionAtPort(0));
        parentNode->addOriginalLayer(childNode->getOriginalLayers());
        graph.DropNode(childNode);
    }
}

void GraphOptimizer::MergeConvertAndEltwise(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto parent = graphNodes.begin();
    while (parent != graphNodes.end()) {
        CPU_GRAPH_OPTIMIZER_SCOPE(MergeConvertAndEltwise);
        auto parentNode = *parent;
        if (parentNode->getType() != Type::Convert) {
            parent++;
            continue;
        }

        const auto& childEdges = parentNode->getChildEdges();
        if (childEdges.size() != 1) {
            parent++;
            continue;
        }

        const auto edge = childEdges[0].lock();
        auto childNode = edge->getChild();
        if (childNode->getType() != Type::Eltwise) {
            parent++;
            continue;
        }

        const auto eltwise = dynamic_cast<ov::intel_cpu::node::Eltwise*>(childNode.get());
        if (!eltwise->canFuseParent(parentNode)) {
            parent++;
            continue;
        }

        const auto parents = parentNode->parentEdges;
        for (const auto& i : parents) {
            auto p_edge = i.lock();
            if (!p_edge) {
                continue;
            }
            auto parent = p_edge->getParent();
            if (!parent) {
                continue;
            }

            if (!parentNode->childEdges[0].lock()) {
                continue;
            }
            auto child = parentNode->childEdges[0].lock()->getChild();
            if (!child) {
                continue;
            }

            EdgePtr& remEdge = p_edge;
            int inNum = 0;
            if (remEdge) {
                inNum = remEdge->getInputNum();
                graph.RemoveEdge(remEdge);
            }
            remEdge = parentNode->childEdges[0].lock();
            int outNum = 0;
            if (remEdge) {
                outNum = remEdge->getOutputNum();
                graph.RemoveEdge(remEdge);
            }
            graph.CreateEdge(parent, child, inNum, outNum);
        }

        childNode->setOriginalInputPrecisionAtPort(0, parentNode->getOriginalInputPrecisionAtPort(0));
        childNode->addOriginalLayer(parentNode->getOriginalLayers());
        graph.DropNode(parentNode);
    }
}

void GraphOptimizer::FuseFCAndConvertOnWeights(Graph& graph) {
#if defined(OV_CPU_WITH_SHL)
    return;
#endif

    // This optimization fuses Convert (fp16 -> bf16/fp32) on weights directly to FC input to allow precision conversion
    // handling based on internal logic (e.g. fuse conversion with weights reordering)

    auto isSuitableTranspose = [](const NodePtr& node) {
        return node->getType() == Type::Transpose && node->getChildEdges().size() == 1 && node->isConstant();
    };
    auto isSuitableConvert = [&](const NodePtr& node) {
        return node->getType() == Type::Convert && node->isConstant() &&
               one_of(node->getOriginalInputPrecisionAtPort(0), ov::element::f16, ov::element::bf16) &&
               one_of(node->getOriginalOutputPrecisionAtPort(0), ov::element::f32, ov::element::bf16);
    };

    auto& graphNodes = graph.GetNodes();
    for (const auto& fullyConnected : graphNodes) {
        if (fullyConnected->getType() != Type::FullyConnected) {
            continue;
        }

        NodePtr transpose = nullptr;
        auto parent = fullyConnected->getParentEdgeAt(1)->getParent();
        if (parent->getType() == Type::Transpose) {
            if (!isSuitableTranspose(parent)) {
                continue;
            }

            transpose = parent;
            parent = transpose->getParentEdgeAt(0)->getParent();
        }

        const auto convert = parent;
        if (!isSuitableConvert(convert)) {
            continue;
        }

        const auto weights = convert->getParentEdgeAt(0)->getParent();
        const auto weights_out_edge = weights->getChildEdges()[0].lock();
        const auto fc_weights_path_edge =
            transpose ? transpose->getParentEdgeAt(0) : fullyConnected->getParentEdgeAt(1);
        const auto inNum = weights_out_edge->getInputNum();
        const auto outNum = fc_weights_path_edge->getOutputNum();
        const auto originalPrecision = convert->getOriginalInputPrecisionAtPort(0);
        fullyConnected->setOriginalInputPrecisionAtPort(1, originalPrecision);
        if (transpose) {
            transpose->setOriginalInputPrecisionAtPort(0, originalPrecision);
            transpose->setOriginalOutputPrecisionAtPort(0, originalPrecision);
        }
        graph.RemoveEdge(fc_weights_path_edge);
        graph.CreateEdge(weights, transpose ? transpose : fullyConnected, inNum, outNum);
        if (convert->getChildEdges().empty()) {
            graph.DropNode(convert);
        }
    }
}

void GraphOptimizer::FuseFCAndTransposeOnWeights(Graph& graph) {
#if defined(OV_CPU_WITH_SHL)
    return;
#endif

    // This optimization allows us to avoid transposing the weights in Transpose node and do it directly along with
    // reordering in FC node
    auto& graphNodes = graph.GetNodes();

    auto isSuitablePattern = [](const NodePtr& parent) {
        bool res = true && parent->getType() == Type::Transpose && parent->getChildEdges().size() == 1 &&
                   parent->getChildEdgeAt(0)->getChild()->getType() == Type::FullyConnected && parent->isConstant();
        return res;
    };

    for (const auto& parent : graphNodes) {
        if (isSuitablePattern(parent)) {
            CPU_GRAPH_OPTIMIZER_SCOPE(FuseFCAndTransposeOnWeights);
            auto fcNode = std::dynamic_pointer_cast<FullyConnected>(parent->getChildEdgeAt(0)->getChild());
            fcNode->keepWeightsNonTransposed(true);
            auto transposeNode = std::dynamic_pointer_cast<Transpose>(parent);
            transposeNode->setOptimized(true);
        }
    }
}

void GraphOptimizer::FuseConvolutionAndZeroPoints(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableConvNode = [](const NodePtr& node) {
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

    auto initializeInputZeroPoints = [](const NodePtr& node, const NodePtr& parent0, const NodePtr& parent1) {
        auto* convNode = dynamic_cast<Convolution*>(node.get());
        if (convNode == nullptr) {
            OPENVINO_THROW("Cannot get convolution node ", node->getName());
        }

        auto IC = node->getInputShapeAtPort(0).getDims()[1];
        auto OC = node->getOutputShapeAtPort(0).getDims()[1];

        if (Shape::UNDEFINED_DIM == IC || Shape::UNDEFINED_DIM == OC) {
            return false;
        }
        if (parent0->getType() != Type::Eltwise) {
            return false;
        }
        if (!parent0->getFusedWith().empty() || !parent1->getFusedWith().empty()) {
            return false;
        }

        // The plug-in doesn't support FP32 convolution with input/weights zero points.
        // In case weights are in FP32 (or we have zero points on weights which are not supported by INT8 convolution)
        // we cannot use INT8 implementation so we have to disable input zero points fusing as well.
        if (parent1->getType() != Type::Input || !parent1->isConstant() ||
            parent1->getOriginalOutputPrecisionAtPort(0) != ov::element::i8) {
            return false;
        }

        if (parent0->getAlgorithm() != Algorithm::EltwiseSubtract) {
            return false;
        }

        if (parent0->getParentEdges().size() != 2) {
            return false;
        }

        auto subtractArg1 = parent0->getParentEdgeAt(1)->getParent();
        if (subtractArg1->getType() != Type::Input || !subtractArg1->isConstant()) {
            return false;
        }

        if (subtractArg1->getOriginalOutputPrecisionAtPort(0) != ov::element::u8) {
            return false;
        }

        if (parent0->getInputShapeAtPort(1).getRank() < 2) {
            return false;
        }

        auto zpDims = parent0->getInputShapeAtPort(1).getDims();
        if (zpDims[0] != 1 || !dimsEqualStrong(zpDims[1], IC)) {
            return false;
        }

        for (size_t i = 2; i < zpDims.size(); i++) {
            if (zpDims[i] != 1) {
                return false;
            }
        }

        const auto& parentEdge = parent0->getParentEdgeAt(0);
        const auto& subtractArg0 = parentEdge->getParent();
        const size_t portNum = parentEdge->getInputNum();
        if (subtractArg0->getOriginalOutputPrecisionAtPort(portNum) != ov::element::u8) {
            return false;
        }

        auto zeroPointsConstant = dynamic_cast<node::Input*>(subtractArg1.get());
        if (zeroPointsConstant == nullptr) {
            OPENVINO_THROW("Cannot cast to Input node");
        }

        auto zeroPointsBlob = zeroPointsConstant->getMemoryPtr();
        if (zeroPointsBlob == nullptr) {
            OPENVINO_THROW("Cannot cast to TBlob internal zero points blob");
        }

        auto zeroPointsData = static_cast<const uint8_t*>(zeroPointsBlob->getData());
        if (zeroPointsData == nullptr) {
            OPENVINO_THROW("zeroPointsBlob has not allocated buffer");
        }

        auto zeroPointDataSize = parent0->getInputShapeAtPort(1).getDims()[1];
        if (Shape::UNDEFINED_DIM == zeroPointDataSize) {
            return false;
        }
        convNode->initializeInputZeroPoints(zeroPointsData, zeroPointDataSize);
        return true;
    };

    auto initializeOutputCompensation = [](const NodePtr& node) {
        auto* convNode = dynamic_cast<Convolution*>(node.get());
        if (convNode == nullptr) {
            OPENVINO_THROW("Cannot get convolution node ", node->getName());
        }

        if (convNode->legacyInputZeroPoints.empty()) {
            return;
        }
        if (convNode->legacyOutputCompensation.empty()) {
            convNode->legacyOutputCompensation.resize(convNode->getOutputShapeAtPort(0).getDims()[1]);
        }

        auto weightsConstant = dynamic_cast<node::Input*>(convNode->getParentEdgeAt(1)->getParent().get());
        if (!weightsConstant || !weightsConstant->isConstant()) {
            return;
        }

        auto weightsBlob = weightsConstant->getMemoryPtr();
        if (weightsBlob == nullptr) {
            OPENVINO_THROW("Cannot cast to TBlob internal weights blob");
        }

        auto weightsPtr = static_cast<const int8_t*>(weightsBlob->getData());
        if (weightsPtr == nullptr) {
            OPENVINO_THROW("weightsBlob has not allocated buffer");
        }

        auto G = convNode->getGroupNum();
        const size_t groupOffset = convNode->getAlgorithm() == Algorithm::ConvolutionGrouped ? 1 : 0;
        auto& weightsConstantDims = weightsConstant->outputShapes[0].getStaticDims();

        auto OC = weightsConstantDims[0 + groupOffset];
        auto IC = weightsConstantDims[1 + groupOffset];
        auto KD =
            weightsConstantDims.size() == (5 + groupOffset) ? weightsConstantDims[weightsConstantDims.size() - 3] : 1;
        auto KH =
            weightsConstantDims.size() == (3 + groupOffset) ? 1 : weightsConstantDims[weightsConstantDims.size() - 2];
        auto KW = weightsConstantDims[weightsConstantDims.size() - 1];

        for (size_t g = 0; g < G; g++) {
            for (size_t oc = 0; oc < OC; oc++) {
                int32_t a = 0;
                for (size_t ic = 0; ic < IC; ic++) {
                    for (size_t kd = 0; kd < KD; kd++) {
                        for (size_t kh = 0; kh < KH; kh++) {
                            for (size_t kw = 0; kw < KW; kw++) {
                                size_t widx = g * OC * IC * KD * KH * KW + oc * IC * KD * KH * KW + ic * KD * KH * KW +
                                              kd * KH * KW + kh * KW + kw;

                                auto w = static_cast<int32_t>(weightsPtr[widx]);

                                auto izp = !convNode->legacyInputZeroPoints.empty()
                                               ? static_cast<int32_t>(convNode->legacyInputZeroPoints[g * IC + ic])
                                               : 0;
                                a += w * izp;

                                auto wzp = !convNode->legacyWeightsZeroPoints.empty()
                                               ? static_cast<int32_t>(convNode->legacyWeightsZeroPoints[g * OC + oc])
                                               : 0;
                                a -= wzp * izp;
                            }
                        }
                    }
                }
                convNode->legacyOutputCompensation[g * OC + oc] = -a;
            }
        }
    };

    for (const auto& conv : graphNodes) {
        if (!isSuitableConvNode(conv)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseConvolutionAndZeroPoints_ConvNode);

        auto dataEltwise = conv->getParentEdgeAt(0)->getParent();
        auto weightsEltwise = conv->getParentEdgeAt(1)->getParent();
        if (initializeInputZeroPoints(conv, dataEltwise, weightsEltwise)) {
            auto p_edge = dataEltwise->getParentEdgeAt(1);
            DEBUG_LOG("[GraphOptimizer##FusingZeorPoint]:Eltwise Subtract Node ##",
                      dataEltwise->getName(),
                      " is optimized as zeropoint of Conv ##",
                      conv->getName());
            graph.RemoveEdge(p_edge);
            graph.DropNode(dataEltwise);
            initializeOutputCompensation(conv);
        }
    }
}

void GraphOptimizer::FuseFullyConnectedAndSimpleOperation(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const NodePtr& node) {
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

        childNode->fuseInto(parentNode);

        if (childNode->getType() == Type::FakeQuantize || childNode->getType() == Type::Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto& parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Type::FullyConnected) {
                    continue;
                }

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseMatMulAndSimpleOperation(Graph& graph) {
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
            for (auto& parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Type::MatMul) {
                    continue;
                }

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseConvolutionAndDWConvolution(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isConvolutionNode = [](const NodePtr& node) {
        return node->getType() == Type::Convolution;
    };

    auto is1x1Convolution = [](const std::shared_ptr<Convolution>& conv) {
        const auto weightRank = conv->getWeightDims().size();
        return conv->getWeightDims()[weightRank - 1] == 1 && conv->getWeightDims()[weightRank - 2] == 1;
    };

    auto isSuitableParentConvolution = [&](const NodePtr& node) {
        if (node->isDropped()) {
            return false;
        }

        if (node->isDynamicNode()) {
            return false;
        }

        const auto conv = std::dynamic_pointer_cast<Convolution>(node);
        if (conv == nullptr) {
            OPENVINO_THROW("Cannot cast to convolution node ", node->getName());
        }

        if (!conv->legacyWeightsZeroPoints.empty()) {
            return false;
        }

        const auto& strides = conv->getStride();
        const auto& paddings = conv->getPaddingL();
        const auto& inDims = node->getInputShapeAtPort(0).getDims();
        const auto& outDims = node->getOutputShapeAtPort(0).getDims();
        bool isSupportedParams =
            conv->getGroupNum() == 1 && inDims.size() == 4 &&
            dimsEqualStrong(inDims[inDims.size() - 1], outDims[outDims.size() - 1]) &&
            dimsEqualStrong(inDims[inDims.size() - 2], outDims[outDims.size() - 2]) &&
            is1x1Convolution(conv) &&  // TODO [oneDNN] : fusing is permitted only with 1x1 convolutions
            everyone_is(1u,
                        static_cast<unsigned int>(strides[strides.size() - 1]),
                        static_cast<unsigned int>(strides[strides.size() - 2])) &&
            everyone_is(0u,
                        static_cast<unsigned int>(paddings[paddings.size() - 1]),
                        static_cast<unsigned int>(paddings[paddings.size() - 2])) &&
            !conv->canBeExecutedInInt8();
        if (!isSupportedParams) {
            return false;
        }

        return node->getChildEdges().size() == 1 && isConvolutionNode(node->getChildEdgeAt(0)->getChild());
    };

    auto isSuitableChildConvolution = [&](const NodePtr& parentNode, const NodePtr& childNode) {
        if (parentNode->isDropped() || childNode->isDropped()) {
            return false;
        }

        if (childNode->isDynamicNode()) {
            return false;
        }

        const auto convChild = std::dynamic_pointer_cast<Convolution>(childNode);
        if (convChild == nullptr) {
            OPENVINO_THROW("Cannot cast to convolution node ", childNode->getName());
        }

        const auto convParent = std::dynamic_pointer_cast<Convolution>(parentNode);
        if (convParent == nullptr) {
            OPENVINO_THROW("Cannot cast to convolution node ", parentNode->getName());
        }

        if (!everyone_is(ov::element::f32,
                         convParent->getOriginalOutputPrecisionAtPort(0),
                         convChild->getOriginalInputPrecisionAtPort(0),
                         convChild->getOriginalOutputPrecisionAtPort(0))) {
            return false;
        }

        auto parentOutputPrecision =
            !parentNode->fusedWith.empty()
                ? parentNode->fusedWith[parentNode->fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0)
                : parentNode->getOriginalOutputPrecisionAtPort(0);

        auto childOutputPrecision =
            !childNode->fusedWith.empty()
                ? childNode->fusedWith[childNode->fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0)
                : childNode->getOriginalOutputPrecisionAtPort(0);

        if (!everyone_is(ov::element::f32, parentOutputPrecision, childOutputPrecision)) {
            return false;
        }

        if (!convChild->legacyInputZeroPoints.empty() || !convChild->legacyWeightsZeroPoints.empty()) {
            return false;
        }

        bool withBias = convChild->getOriginalInputPrecisions().size() == 3;

        const auto weightRank = convChild->getWeightDims().size();
        const auto stridesSize = convChild->getStride().size();
        bool isSupportedParams =
            dimsEqualStrong(convChild->outputShapes[0].getDims()[1], convChild->getGroupNum()) &&
            convChild->outputShapes[0].getDims()[1] != 1 &&
            everyone_is(3u,
                        static_cast<unsigned int>(convChild->getWeightDims()[weightRank - 1]),
                        static_cast<unsigned int>(convChild->getWeightDims()[weightRank - 2])) &&
            everyone_is(1u,
                        static_cast<unsigned int>(convChild->getPaddingL()[stridesSize - 1]),
                        static_cast<unsigned int>(convChild->getPaddingL()[stridesSize - 2])) &&
            everyone_is(1u,
                        static_cast<unsigned int>(convChild->getPaddingR()[stridesSize - 1]),
                        static_cast<unsigned int>(convChild->getPaddingR()[stridesSize - 2])) &&
            everyone_is(1u,
                        static_cast<unsigned int>(convChild->getDilation()[stridesSize - 1] + 1),
                        static_cast<unsigned int>(convChild->getDilation()[stridesSize - 2] + 1)) &&
            convChild->getStride()[stridesSize - 1] == convChild->getStride()[stridesSize - 2] && withBias &&
            one_of(convChild->getStride()[stridesSize - 1], 1u, 2u) &&
            childNode->getOutputShapeAtPort(0).getRank() == 4;

        return isSupportedParams;
    };

    auto isFusingWorthwhile = [&](const NodePtr& parentNode, const NodePtr& childNode) {
        if (!childNode->inputShapes[0].isStatic() || !childNode->outputShapes[0].isStatic()) {
            return false;
        }

        auto inDims = childNode->inputShapes[0].getStaticDims();
        auto outDims = childNode->outputShapes[0].getStaticDims();
        int elemSize = childNode->getOriginalOutputPrecisionAtPort(0).size();

        int L3_cache_size = dnnl::utils::get_cache_size(3, false);
        int dw_conv_input_size = inDims[0] * inDims[1] * inDims[2] * inDims[3] * elemSize;
        int dw_conv_output_size = outDims[0] * outDims[1] * outDims[2] * outDims[3] * elemSize;

        auto parentConvolutionNode = std::dynamic_pointer_cast<Convolution>(parentNode);
        if (parentConvolutionNode == nullptr) {
            OPENVINO_THROW("Cannot get convolution node ", parentNode->getName());
        }

        if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx2) || impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core)) {
            return false;
        }

        return (dw_conv_input_size + dw_conv_output_size > L3_cache_size / 2);
    };

    for (const auto& graphNode : graphNodes) {
        if (!isConvolutionNode(graphNode)) {
            continue;
        }

        const auto& parentConvNode = graphNode;
        if (!isSuitableParentConvolution(parentConvNode)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseConvolutionAndDWConvolution_ParentConv);

        auto childConvNode = parentConvNode->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildConvolution(parentConvNode, childConvNode)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseConvolutionAndDWConvolution_ChildConv);

        if (!isFusingWorthwhile(parentConvNode, childConvNode)) {
            continue;
        }

        parentConvNode->addFusedNode(childConvNode);

        for (auto& node : childConvNode->getFusedWith()) {
            parentConvNode->addFusedNode(node);
        }
        childConvNode->clearFusedWith();

        graph.DropDWConvNode(childConvNode);
    }
}

// TODO [NM]: unite with FuseConvolutionAndSimpleOperation
void GraphOptimizer::FuseConvolutionAndSimpleOperationThroughMaxPool(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const NodePtr& node) {
        return (node->getType() == Type::Convolution || node->getType() == Type::BinaryConvolution) &&
               node->getChildEdges().size() == 1 && node->getOriginalOutputPrecisionAtPort(0) == ov::element::f32;
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

#if defined(OV_CPU_WITH_ACL)
        if (!parentNode->getFusedWith().empty()) {
            parent++;
            continue;
        }
#endif

        if (!DnnlExtensionUtils::isUnarySupportedAsPostOp(fuseCandidate->getAlgorithm())) {
            parent++;
            continue;
        }
        parentNode->addFusedNode(fuseCandidate);
        parentNode->addOriginalLayer(fuseCandidate->getOriginalLayers());
        auto parentEdges = fuseCandidate->parentEdges;
        for (auto& parentEdge : parentEdges) {
            auto p_edge = parentEdge.lock();
            if (p_edge->getParent() == childNode) {
                continue;
            }

            graph.RemoveEdge(p_edge);
        }
        graph.DropNode(fuseCandidate);
    }
}

void GraphOptimizer::FuseConvolutionAndSimpleOperation(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const NodePtr& node) {
        return (node->getType() == Type::Convolution || node->getType() == Type::BinaryConvolution) &&
               node->getChildEdges().size() == 1;
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

        childNode->fuseInto(parentNode);

        if (childNode->getType() == Type::FakeQuantize || childNode->getType() == Type::Eltwise) {
            auto parentEdges = childNode->parentEdges;
            for (auto& parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == parentNodeType) {
                    continue;
                }

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FusePoolingAndFakeQuantize(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const NodePtr& node) {
        if (node->getType() == Type::Pooling) {
            if (!one_of(node->getOriginalInputPrecisionAtPort(0), ov::element::u8, ov::element::i8)) {
                return false;
            }
            return node->getChildEdges().size() == 1 && node->getAlgorithm() == Algorithm::PoolingAvg;
        }
        return false;
    };

    auto isSuitableChildNode = [](const NodePtr& node) {
        return node->getType() == Type::FakeQuantize && node->getAlgorithm() != Algorithm::FQBinarization;
    };

    for (auto parent : graphNodes) {
        if (!isSuitableParentNode(parent)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FusePoolingAndFakeQuantize_ParentNode);

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSuitableChildNode(child)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FusePoolingAndFakeQuantize_ChildNode);

        child->fuseInto(parent);

        auto parents = child->parentEdges;
        for (const auto& i : parents) {
            auto p_edge = i.lock();
            if (p_edge->getParent()->getType() == Type::Pooling) {
                continue;
            }

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
static bool is_data_dependency(const std::shared_ptr<Node>& parent, const std::shared_ptr<Node>& child) {
    std::set<Node*> visited;
    std::list<Node*> nextLayers{parent.get()};

    for (; !nextLayers.empty();) {
        auto layer = *nextLayers.begin();
        if (layer == child.get()) {
            return true;
        }
        for (auto& oe : layer->getChildEdges()) {
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

void GraphOptimizer::FuseConvolutionSumAndConvolutionSumActivation(Graph& graph) {
#if !defined(OPENVINO_ARCH_X86) && !defined(OPENVINO_ARCH_X86_64)
    return;
#endif

    auto& graphNodes = graph.GetNodes();

    auto isFusingSupported = [&](const NodePtr& conv, const NodePtr& child) {
        return child->getType() == Type::Eltwise && DnnlExtensionUtils::isUnarySupportedAsPostOp(child->getAlgorithm());
    };

    for (auto& graphNode : graphNodes) {
        const auto eltwiseNode = std::dynamic_pointer_cast<Eltwise>(graphNode);
        if (graphNode->getType() != Type::Eltwise || graphNode->getAlgorithm() != Algorithm::EltwiseAdd ||
            !eltwiseNode || eltwiseNode->isWithBroadcast()) {
            continue;
        }

        // TODO: Enlarge to several inputs
        bool isSuitableNode = graphNode->getParentEdges().size() == 2;
        if (!isSuitableNode) {
            continue;
        }

        auto parent1 = graphNode->getParentEdgeAt(0)->getParent();
        auto parent2 = graphNode->getParentEdgeAt(1)->getParent();

        bool isSuitableParent1 =
            parent1->getType() == Type::Convolution || parent1->getType() == Type::BinaryConvolution;
        bool isSuitableParent2 =
            parent2->getType() == Type::Convolution || parent2->getType() == Type::BinaryConvolution;

        auto canFuseSum = [](node::BinaryConvolution* binConv, const NodePtr& fuseCandidate) {
            if (binConv->getImplType() == impl_desc_type::ref) {
                return false;
            }

            if (binConv->isFusedWith(Type::FakeQuantize)) {
                return false;
            }

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

        auto* binConvNode1 = dynamic_cast<node::BinaryConvolution*>(parent1.get());
        if (binConvNode1) {
            isSuitableParent1 = isSuitableParent1 && canFuseSum(binConvNode1, graphNode);
        }

        auto* binConvNode2 = dynamic_cast<node::BinaryConvolution*>(parent2.get());
        if (binConvNode2) {
            isSuitableParent2 = isSuitableParent2 && canFuseSum(binConvNode2, graphNode);
        }

        auto checkFusedWithSum = [](Convolution* conv) -> bool {
            for (const auto& node : conv->getFusedWith()) {
                const auto eltwise = std::dynamic_pointer_cast<Eltwise>(node);
                if (eltwise && eltwise->isSpecialConvolutionAddFusing()) {
                    return true;
                }
            }
            return false;
        };

        auto* convNode1 = dynamic_cast<Convolution*>(parent1.get());
        if (convNode1) {
            if (!convNode1->canBeExecutedInInt8()) {
                isSuitableParent1 = isSuitableParent1 && convNode1->getFusedWith().empty();
            } else {
                isSuitableParent1 = isSuitableParent1 && !checkFusedWithSum(convNode1);
            }
        }

        auto* convNode2 = dynamic_cast<Convolution*>(parent2.get());
        if (convNode2) {
            if (!convNode2->canBeExecutedInInt8()) {
                isSuitableParent2 = isSuitableParent2 && convNode2->getFusedWith().empty();
            } else {
                isSuitableParent2 = isSuitableParent2 && !checkFusedWithSum(convNode2);
            }
        }

        if (!isSuitableParent1 && !isSuitableParent2) {
            continue;
        }

        std::shared_ptr<Node> mergedConv;
        std::shared_ptr<Node> peerNode;

        if (isSuitableParent1 && isSuitableParent2) {
            // not merged operation (peerNode) has to be in low precision
            const auto isBranchQuantized = [](const NodePtr& branchParent) {
                const auto& fused = branchParent->getFusedWith();
                const auto branchPrecision = fused.empty()
                                                 ? branchParent->getOriginalOutputPrecisionAtPort(0)
                                                 : fused[fused.size() - 1]->getOriginalOutputPrecisionAtPort(0);
                return (branchPrecision == ov::element::i8) || (branchPrecision == ov::element::u8);
            };

            const auto isBranch1Quantized = isBranchQuantized(graphNode->getParentEdgeAt(0)->getParent());
            const auto isBranch2Quantized = isBranchQuantized(graphNode->getParentEdgeAt(1)->getParent());
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
        if (peerNode->isConstant()) {
            continue;
        }
        const auto& sum = graphNode;

        if (mergedConv->isConstant() && !sum->isConstant()) {
            continue;
        }

        // Disable fusing for Add with broadcasing in case of known data ranges. Add with brodcasting triggers
        // non-optimal code path inside Convolution node, so better to avoid fusing at all.
        const auto& shape1 = sum->getInputShapeAtPort(0);
        const auto& shape2 = sum->getInputShapeAtPort(1);
        if (shape1.getRank() != shape2.getRank()) {
            continue;
        }

        const auto& dims1 = shape1.getDims();
        const auto& dims2 = shape2.getDims();
        bool dynamic_bcast_pattern = false;
        for (size_t d = 2; d < shape1.getRank(); d++) {
            bool cond1 = (dims1[d] == Shape::UNDEFINED_DIM) && (dims2[d] == 1U);
            bool cond2 = (dims2[d] == Shape::UNDEFINED_DIM) && (dims1[d] == 1U);
            if (cond1 || cond2) {
                dynamic_bcast_pattern = true;
                break;
            }
        }
        if (dynamic_bcast_pattern) {
            continue;
        }

        auto lastNode = sum;

        bool fuse_allowed = mergedConv->getChildEdges().size() == 1;
        for (size_t j = 0; fuse_allowed && j < mergedConv->getParentEdges().size(); j++) {
            if (mergedConv->getParentEdgeAt(j)->getParent() == peerNode) {
                fuse_allowed = false;
            }
        }

        // Fused Conv+Sum prim will be used inplace. That's mean that input blob will
        // be overwritten. Should verify that all other consumer already read it and
        // we can spoil input data.
        // TODO: rewrite once we add "Inplace" reporting mechanism
        for (auto& edge : peerNode->getChildEdges()) {
            if (!fuse_allowed) {
                break;
            }
            fuse_allowed &= is_data_dependency(edge.lock()->getChild(), sum);
        }
        if (!fuse_allowed) {
            continue;
        }

        if (graphNode->getChildEdges().size() == 1 &&
            isFusingSupported(graphNode, graphNode->getChildEdgeAt(0)->getChild())) {
            auto relu_shared = graphNode->getChildEdgeAt(0)->getChild();
            lastNode = relu_shared;
            if (mergedConv->isConstant() && !lastNode->isConstant()) {
                continue;
            }
            sum->fuseInto(mergedConv);
        }

        lastNode->fuseInto(mergedConv);

        if (mergedConv->fusedWith.size() > 0 && (mergedConv->fusedWith[0]->getType() == Type::Convolution ||
                                                 mergedConv->fusedWith[0]->getType() == Type::BinaryConvolution)) {
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

        auto peerEdge = peerNode->getChildEdgeAt(childIdx);
        const int peer_port = peerEdge->getInputNum();
        graph.RemoveEdge(peerEdge);

        int childPort = 1;
        auto* mergedConvNode = dynamic_cast<Convolution*>(mergedConv.get());
        if (mergedConvNode != nullptr) {
            childPort = mergedConvNode->getParentEdges().size();
        }

        auto* mergedBinConvNode = dynamic_cast<node::BinaryConvolution*>(mergedConv.get());
        if (mergedBinConvNode != nullptr) {
            childPort = mergedBinConvNode->getParentEdges().size();
        }

        graph.CreateEdge(peerNode, mergedConv, peer_port, childPort);

        std::vector<EdgeWeakPtr> edges_to_reconnect = lastNode->getChildEdges();
        for (auto& edge_w : edges_to_reconnect) {
            auto edge = edge_w.lock();
            auto child = edge->getChild();
            int idxParent = edge->getInputNum();
            int idxChild = edge->getOutputNum();

            // reconnect after  activation/sum. Port index must be 0
            OPENVINO_ASSERT(idxParent == 0);

            graph.RemoveEdge(edge);
            graph.CreateEdge(mergedConv, child, idxParent, idxChild);
        }

        if (lastNode != sum) {
            lastNode->remove();
        }
        sum->remove();
    }
}

void GraphOptimizer::FuseMVNAndSimpleOperation(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const NodePtr& node) {
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
            for (auto& parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Type::MVN) {
                    continue;
                }

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseInterpolateAndSimpleOperation(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const NodePtr& node) {
        return node->getType() == Type::Interpolate && node->getChildEdges().size() == 1;
    };

    auto isSuitableChildNode = [&](const NodePtr& parentNode, const NodePtr& childNode) {
        // Avoid cycle dependencies
        for (auto& childParentEdge : childNode->getParentEdges()) {
            for (auto& parentParentEdge : parentNode->getParentEdges()) {
                if (childParentEdge.lock()->getParent() == parentParentEdge.lock()->getParent()) {
                    return false;
                }
            }
        }
        if (!childNode->getFusedWith().empty()) {
            return false;
        }
        auto interpolateNode = dynamic_cast<Interpolate*>(parentNode.get());
        if (!interpolateNode) {
            OPENVINO_THROW("Cannot cast ", parentNode->getName(), " to Interpolate");
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
            for (auto& parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Type::Interpolate) {
                    continue;
                }

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseNormalizeL2AndSimpleOperation(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const NodePtr& node) {
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
            for (auto& parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Type::NormalizeL2) {
                    continue;
                }

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseReduceAndSimpleOperation(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const NodePtr& node) {
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
            for (auto& parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge == nullptr) {
                    OPENVINO_THROW("Cannot get parent edge ", childNode->getName());
                }
                if (p_edge->getParent()->getType() == Type::Reduce) {
                    continue;
                }

                graph.RemoveEdge(p_edge);
            }
        }

        graph.DropNode(childNode);
    }
}

void GraphOptimizer::FuseEltwiseAndSimple(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const NodePtr& node) {
        return node->getType() == Type::Eltwise && node->getChildEdges().size() == 1;
    };

    auto isSuitableChildNode = [&](const NodePtr& parentNode, const NodePtr& childNode) {
        if (parentNode->isConstant() && !childNode->isConstant()) {
            return false;
        }
        for (auto& childParentEdge : childNode->getParentEdges()) {
            // WA to prevent unsupported reorder exception issue in some cases
            if (childParentEdge.lock()->getParent()->getType() == Type::Split) {
                return false;
            }

            // Avoid cycle dependencies
            for (auto& parentParentEdge : parentNode->getParentEdges()) {
                if (childParentEdge.lock()->getParent() == parentParentEdge.lock()->getParent()) {
                    return false;
                }
            }
        }

        if (!childNode->getFusedWith().empty()) {
            return false;
        }

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

        if ((parentNode->isDynamicNode() && !childNode->isDynamicNode()) ||
            (!parentNode->isDynamicNode() && childNode->isDynamicNode())) {
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
            for (auto& parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (p_edge->getParent()->getType() == Type::Eltwise) {
                    continue;
                }

                graph.RemoveEdge(p_edge);
            }

            graph.DropNode(childNode);
        } else if (childNode->getType() == Type::Eltwise) {
            auto children = childNode->childEdges;
            auto parents = childNode->parentEdges;
            auto initialParentInNum = parentNode->getParentEdges().size();

            for (const auto& i : parents) {
                auto p_edge = i.lock();
                if (!p_edge) {
                    continue;
                }
                auto parent = p_edge->getParent();
                if (!parent) {
                    continue;
                }

                if (parent == parentNode) {
                    for (const auto& j : children) {
                        if (!j.lock()) {
                            continue;
                        }
                        auto child = j.lock()->getChild();
                        if (!child) {
                            continue;
                        }

                        EdgePtr& remEdge = p_edge;
                        int inNum = 0;
                        if (remEdge) {
                            inNum = remEdge->getInputNum();
                            graph.RemoveEdge(remEdge);
                        }
                        remEdge = j.lock();
                        int outNum = 0;
                        if (remEdge) {
                            outNum = remEdge->getOutputNum();
                            graph.RemoveEdge(remEdge);
                        }
                        parent->outputShapes[inNum] = child->inputShapes[outNum];
                        graph.CreateEdge(parent, child, inNum, outNum);
                    }
                } else {
                    EdgePtr& remEdge = p_edge;
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

                    if (parentNode->inputShapes.size() < static_cast<size_t>(outNum) + 1) {
                        parentNode->inputShapes.resize(outNum + 1);
                    }
                    parentNode->inputShapes[outNum] = parent->getOutputShapeAtPort(inNum);

                    graph.CreateEdge(parent, parentNode, inNum, outNum);
                }
            }

            graph.DropNode(childNode);
        } else {
            graph.DropNode(childNode);
        }
    }
}

void GraphOptimizer::ShareReorders(Graph& graph) {
    auto getSuitableReorder = [](const NodePtr& node) -> Reorder* {
        if (node->getType() != Type::Reorder) {
            return nullptr;
        }
        auto* reorder = dynamic_cast<Reorder*>(node.get());
        if (reorder == nullptr) {
            OPENVINO_THROW("Cannot get reorder layer ", node->getName());
        }

        // inplace children cannot be safely shared with each other
        auto reorderConsumers = reorder->getChildEdgesAtPort(0);
        if (std::any_of(reorderConsumers.begin(), reorderConsumers.end(), [](const EdgePtr& e) {
                return e->inPlace(Edge::LOOK_DOWN);
            })) {
            return nullptr;
        }
        return reorder;
    };

    std::set<NodePtr> dropped;
    for (const auto& node : graph.GetNodes()) {
        if (dropped.find(node) != dropped.end()) {
            continue;
        }

        Reorder* reorder = getSuitableReorder(node);
        if (!reorder) {
            continue;
        }

        // find shareable sibling
        auto dataEdge = reorder->getParentEdgeAt(0);
        auto parentNode = dataEdge->getParent();
        auto parentPort = dataEdge->getInputNum();
        for (auto& edge : parentNode->getChildEdgesAtPort(parentPort)) {
            auto siblingNode = edge->getChild();
            if (siblingNode == node) {
                continue;
            }
            Reorder* siblingReorder = getSuitableReorder(siblingNode);
            if (!siblingReorder) {
                continue;
            }
            if (!reorder->getOutput().isCompatible(siblingReorder->getOutput())) {
                continue;
            }

            DEBUG_LOG(node->getName(), " is shared by ", siblingNode->getName());

            // siblingReorder can share output with current reorder
            for (const auto& pwEdge : siblingReorder->getParentEdges()) {
                auto pEdge = pwEdge.lock();
                if (pEdge) {
                    graph.RemoveEdge(pEdge);
                }
            }

            for (const auto& pwEdge : siblingReorder->getChildEdges()) {
                auto pEdge = pwEdge.lock();
                if (pEdge) {
                    graph.RemoveEdge(pEdge);
                    if (pEdge->getInputNum() == 0) {
                        graph.CreateEdge(node, pEdge->getChild(), 0, pEdge->getOutputNum());
                    }
                }
            }

            dropped.insert(siblingNode);
        }
    }
}

void GraphOptimizer::DropDoubleReorders(Graph& graph) {
    std::set<NodePtr> processed;

    auto& nodes = graph.GetNodes();
    for (const auto& node : nodes) {
        if (processed.find(node) == processed.end() && node->getType() == Type::Reorder &&
            node->getChildEdges().size() == 1 && node->getChildEdgeAt(0)->getChild()->getType() == Type::Reorder) {
            auto nextNode = node->getChildEdgeAt(0)->getChild();
            auto* n = dynamic_cast<Reorder*>(node.get());
            if (n == nullptr) {
                OPENVINO_THROW("Cannot get reorder layer ", node->getName());
            }
            auto* nn = dynamic_cast<Reorder*>(nextNode.get());
            if (nn == nullptr) {
                OPENVINO_THROW("Cannot get reorder layer ", nextNode->getName());
            }

            NodePtr p = n->getParentEdgeAt(0)->getParent();
            NodePtr c = nn->getChildEdgeAt(0)->getChild();

            auto oldEdgeNum = n->getParentEdgeAt(0)->getInputNum();

            graph.DropNode(node);
            graph.DropNode(nextNode);

            processed.insert(node);
            processed.insert(nextNode);

            EdgePtr edge;
            for (auto& cur : p->getChildEdgesAtPort(oldEdgeNum)) {
                if (cur->getChild() == c) {
                    edge = cur;
                }
            }
            if (!edge) {
                OPENVINO_THROW("Inappropriate graph processing");
            }

            std::string layerName = edge->getParent()->getName() + "_ScaleReorder_" + edge->getChild()->getName();
            graph.InsertReorder(edge, layerName, n->getInput(), nn->getOutput(), false);
            graph.RemoveEdge(edge);
        }
    }
}

void GraphOptimizer::FuseClampAndFakeQuantize(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableClampNode = [](const NodePtr& node) {
        return node->getType() == Type::Eltwise && node->getChildEdges().size() == 1 &&
               node->getAlgorithm() == Algorithm::EltwiseClamp;
    };

    auto isSuitableFakeQuantizeNode = [](const NodePtr& node) {
        return node->getType() == Type::FakeQuantize && node->getAlgorithm() != Algorithm::FQBinarization;
    };

    auto fuseClampAndFakeQuantizeNodes = [](const NodePtr& parent, const NodePtr& child) {
        auto* eltwiseNode = dynamic_cast<Eltwise*>(parent.get());
        if (eltwiseNode == nullptr) {
            OPENVINO_THROW("Cannot cast ", parent->getName(), " to Eltwise node");
        }

        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(child.get());
        if (fakeQuantizeNode == nullptr) {
            OPENVINO_THROW("Cannot cast ", child->getName(), " to FakeQuantize node");
        }

        const std::vector<float>& cropLowData = fakeQuantizeNode->getCropLow();
        const std::vector<float>& cropHighData = fakeQuantizeNode->getCropHigh();

        std::vector<float> newCropLow(cropLowData.size());
        std::vector<float> newCropHigh(cropHighData.size());
        for (size_t i = 0; i < cropLowData.size(); i++) {
            newCropLow[i] = std::max(cropLowData[i], eltwiseNode->getAlpha());
        }
        for (size_t i = 0; i < cropHighData.size(); i++) {
            newCropHigh[i] = std::min(cropHighData[i], eltwiseNode->getBeta());
        }

        fakeQuantizeNode->setCropLow(newCropLow);
        fakeQuantizeNode->setCropHigh(newCropHigh);

        return true;
    };

    for (const auto& parent : graphNodes) {
        if (!isSuitableClampNode(parent)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseClampAndFakeQuantize_ClalmpNode);

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSuitableFakeQuantizeNode(child)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FuseClampAndFakeQuantize_QuantizeNode);

        if (fuseClampAndFakeQuantizeNodes(parent, child)) {
            graph.DropNode(parent);
        }
    }
}

void GraphOptimizer::FusePerformedAsScaleShiftAndFakeQuantize(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto getNonConstPort = [](const NodePtr& node) {
        std::vector<int> nonConstPorts;
        for (size_t i = 0; i < node->getParentEdges().size(); i++) {
            const auto& parent = node->getParentEdgeAt(i)->getParent();
            if (!(parent->getType() == Type::Input && parent->isConstant())) {
                nonConstPorts.push_back(i);
            }
        }
        // there are more than 1 nonconst port or missed
        if (nonConstPorts.size() != 1) {
            return -1;
        }

        return nonConstPorts[0];
    };

    auto isSuitableScaleShiftNode = [getNonConstPort](const NodePtr& node) {
        if (!one_of(node->getAlgorithm(),
                    Algorithm::EltwiseAdd,
                    Algorithm::EltwiseSubtract,
                    Algorithm::EltwiseMultiply,
                    Algorithm::EltwiseDivide,
                    Algorithm::EltwiseMulAdd)) {
            return false;
        }

        const auto nonConstPort = getNonConstPort(node);
        if (nonConstPort == -1) {
            return false;
        }

        const NodePtr eltwiseInput = node->getParentEdgeAt(nonConstPort)->getParent();
        return node->getChildEdges().size() == 1 && node->canBePerformedAsScaleShift(eltwiseInput.get());
    };

    auto isSuitableFakeQuantizeNode = [](const NodePtr& node) {
        return node->getType() == Type::FakeQuantize && node->getAlgorithm() != Algorithm::FQBinarization;
    };

    auto fuseScaleShiftAndFakeQuantizeNodes = [getNonConstPort](const NodePtr& parent, const NodePtr& child) {
        auto fakeQuantizeNode = std::dynamic_pointer_cast<FakeQuantize>(child);
        if (fakeQuantizeNode == nullptr) {
            OPENVINO_THROW("Cannot cast ", child->getName(), " to FakeQuantize node");
        }

        std::vector<float> scalesBuffer;
        std::vector<float> shiftsBuffer;
        auto parentEltwise = std::dynamic_pointer_cast<Eltwise>(parent);
        if (!parentEltwise) {
            OPENVINO_THROW("Cannot cast ", parent->getName(), " to Eltwise node");
        }

        const NodePtr eltwiseInput = parentEltwise->getParentEdgeAt(getNonConstPort(parent))->getParent();
        std::tie(scalesBuffer, shiftsBuffer) = parentEltwise->getScalesAndShifts(eltwiseInput.get());

        const auto& outputShape = child->getOutputShapeAtPort(0);
        VectorDims outputDims = outputShape.getDims();

        // We need to compute explicitly port with unfolded parent,
        // because there is no guarantee, that the order of operands will be invariant
        // (i.e. zero) after all transformations, which may cause wrong channel-dim in
        // [Const-Schift -> Add <- Mul] topology with constant-folded schift,
        // (Const node return 1 by default as channel dim.)
        // Look into FQScaleshiftWithConstantShift test
        const auto nonConstPort = (parent->getParentEdgeAt(0)->getParent()->isConstant() ? 1 : 0);
        const auto channelPos = parent->getParentEdgeAt(nonConstPort)->getParent()->getFusingAxis();

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

        for (float i : scalesBuffer) {
            if (i == 0.f) {
                return false;
            }
        }

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
            const auto* u32data = reinterpret_cast<const uint32_t*>(&value);
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

    for (const auto& parent : graphNodes) {
        if (!isSuitableScaleShiftNode(parent)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FusePerformedAsScaleShiftAndFakeQuantize_ShiftNode);

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSuitableFakeQuantizeNode(child)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(FusePerformedAsScaleShiftAndFakeQuantize_QuantizeNode);

        if (fuseScaleShiftAndFakeQuantizeNodes(parent, child)) {
            auto parentEdges = parent->parentEdges;
            for (auto& parentEdge : parentEdges) {
                auto p_edge = parentEdge.lock();
                if (!p_edge->getParent()->isConstant()) {
                    continue;
                }

                graph.RemoveEdge(p_edge);
            }

            graph.DropNode(parent);
        }
    }
}

bool GraphOptimizer::canBeInplaced(const NodePtr& parentNode, const NodePtr& childNode) {
    const auto parentInPlace = parentNode->getParentEdgeAt(0)->inPlace(Edge::LOOK_UP);
    const auto& childEdges = childNode->getChildEdgesAtPort(0);
    const auto childInPlace = std::any_of(childEdges.begin(), childEdges.end(), [](const EdgePtr& edge) {
        return edge->inPlace(Edge::LOOK_DOWN);
    });
    return !(parentInPlace && childInPlace);
}

bool GraphOptimizer::checkAscendingFinalOrder(const VectorDims& transposeOrder,
                                              const VectorDims& layoutOrder,
                                              const VectorDims& reorderInOrder,
                                              const VectorDims& reorderOutOrder) {
    if (transposeOrder.size() != layoutOrder.size() || layoutOrder.size() != reorderInOrder.size() ||
        reorderInOrder.size() != reorderOutOrder.size()) {
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
    auto reorderOrder = VectorDims(reorderOutOrder.size());
    for (size_t i = 0; i < reorderOrder.size(); i++) {
        for (size_t j = 0; j < reorderOrder.size(); j++) {
            if (reorderOutOrder[i] == reorderInOrder[j]) {
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
}

void GraphOptimizer::mergeTransposeReshapeReorder(Graph& graph,
                                                  const NodePtr& transposeNode,
                                                  const NodePtr& reshapeNode,
                                                  const NodePtr& reorderNode,
                                                  const bool reverseOrder) {
    const auto& parentNode = reverseOrder ? reorderNode : transposeNode;
    const auto& childNode = reverseOrder ? transposeNode : reorderNode;
    auto nodeBeforeSequence = parentNode->getParentEdgeAt(0)->getParent();
    auto nodeBeforeSequencePort = parentNode->getParentEdgeAt(0)->getInputNum();
    auto nodeAfterSequence = childNode->getChildEdgeAt(0)->getChild();

    graph.RemoveEdge(transposeNode->getParentEdgeAt(1));
    if (reshapeNode) {
        graph.RemoveEdge(reshapeNode->getParentEdgeAt(1));
    }

    // To prevent inPlace conflict, we must check that the memory reference is unidirectional
    // or inPlace memory is not used
    // Note: this value must be computed before detaching nodes
    bool isOptimized = canBeInplaced(parentNode, childNode);

    // hold references to all children before dropping reorder_node
    std::vector<std::pair<NodePtr, int>> reorderChildren;
    for (const auto& ccEdge : childNode->getChildEdgesAtPort(0)) {
        reorderChildren.emplace_back(ccEdge->getChild(), ccEdge->getOutputNum());
    }

    // detach nodes from graph by remove all of their edges
    // they will be removed in future graph.RemoveDroppedNodes() call
    auto detachNode = [&](const std::shared_ptr<Node>& node) {
        std::vector<EdgeWeakPtr> edges;
        edges = node->getParentEdges();
        for (auto& edge : edges) {
            graph.RemoveEdge(edge.lock());
        }
        edges = node->getChildEdges();
        for (auto& edge : edges) {
            graph.RemoveEdge(edge.lock());
        }
    };
    detachNode(transposeNode);
    detachNode(reorderNode);
    if (reshapeNode) {
        detachNode(reshapeNode);
    }

    auto reorderInDesc = parentNode->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].getMemDesc();
    auto finalDesc = childNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].getMemDesc();
    auto reorderOutDesc = finalDesc->cloneWithNewPrecision(reorderInDesc->getPrecision());

    std::vector<int> srcPerm;
    auto* castedTranspose = dynamic_cast<Transpose*>(transposeNode.get());
    OPENVINO_ASSERT(castedTranspose,
                    "[CPU] parent node of type:",
                    transposeNode->getTypeStr(),
                    " with name: ",
                    transposeNode->getName(),
                    " is not a transpose node");

    const auto& inOrder = transposeNode->getSelectedPrimitiveDescriptor()
                              ->getConfig()
                              .inConfs[0]
                              .getMemDesc()
                              ->as<BlockedMemoryDesc>()
                              ->getOrder();
    const auto& outOrder = reorderOutDesc->as<BlockedMemoryDesc>()->getOrder();
    // Permutation must be set and reorder mustn't be optimized in 2 cases:
    // 1. Transpose has blocked input & non-blocked output
    // 2. Transpose and Reorder do opposite permutation to each other as expected,
    //    but isOptimized is already set to false due to some preliminarily checks.
    if (!isOptimized || inOrder.size() > outOrder.size()) {
        isOptimized = false;
        // inDesc should be permuted before calling reorder
        auto& ord = castedTranspose->getOrder();
        srcPerm = std::vector<int>(ord.size());
        for (size_t i = 0; i < ord.size(); i++) {
            srcPerm[ord[i]] = i;
        }
    }

    std::string reorderName =
        nodeBeforeSequence->getName() + "_" + Reorder::getReorderArgs(*reorderInDesc, *reorderOutDesc);
    if (isOptimized) {
        reorderName += "_fake";
    }
    DEBUG_LOG("mergeTransposeAndReorder ", parentNode->getName(), " and ", childNode->getName(), " -> ", reorderName);
    auto reorder_layout =
        std::make_shared<node::Reorder>(*reorderInDesc, *reorderOutDesc, reorderName, graph.getGraphContext());
    reorder_layout->setOptimized(isOptimized);
    reorder_layout->setSrcPermutation(srcPerm);

    graph.CreateEdge(nodeBeforeSequence, reorder_layout, nodeBeforeSequencePort, 0);

    // If precisions don't match, another reorder must be inserted to perform conversion
    auto reorder_last = reorder_layout;
    if (reorderOutDesc->getPrecision() != finalDesc->getPrecision()) {
        std::string reorderLayerName2 = reorder_layout->getName() + "_" +
                                        Reorder::getReorderArgs(*reorderOutDesc, *finalDesc) + "_" +
                                        nodeAfterSequence->getName();

        reorder_last =
            std::make_shared<node::Reorder>(*reorderOutDesc, *finalDesc, reorderLayerName2, graph.getGraphContext());
        reorder_last->setOptimized(false);
        reorder_last->setSrcPermutation(srcPerm);
        graph.CreateEdge(reorder_layout, reorder_last, 0, 0);
    }

    for (auto& cc : reorderChildren) {
        graph.CreateEdge(reorder_last, cc.first, 0, cc.second);
    }

    // initialize and add nodes into graph
    std::vector<NodePtr> new_nodes;
    new_nodes.push_back(reorder_layout);
    if (reorder_last != reorder_layout) {
        new_nodes.push_back(reorder_last);
    }
    for (auto& node : new_nodes) {
        graph.AddNode(node);
    }

    // multiple nodes must be initialized in specific order
    for (auto& node : new_nodes) {
        node->init();
    }
    for (auto& node : new_nodes) {
        node->getSupportedDescriptors();
        node->initSupportedPrimitiveDescriptors();
        node->filterSupportedPrimitiveDescriptors();
    }
    for (auto& node : new_nodes) {
        node->selectOptimalPrimitiveDescriptor();
    }
    for (auto& node : new_nodes) {
        node->resolveInPlaceDirection();
    }
    for (auto& node : new_nodes) {
        node->initOptimalPrimitiveDescriptor();
    }
}

void GraphOptimizer::MergeTransposeAndReorder(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableTranspose = [](const NodePtr& node) {
        // WA: to avoid broken memory pointer for conv + sum
        auto prevNodeIsConvSum = [](const NodePtr& node) -> bool {
            const auto parent = node->getParentEdgeAt(0)->getParent();
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

        return node->getType() == Type::Transpose && node->getChildEdges().size() == 1 &&
               !node->isDynamicNode()  // TODO [DS]: enable for dynamic shapes when inPlace in the dynamic case is
                                       // available (CVS-74863)
               && !prevNodeIsConvSum(node);
    };

    auto isSuitableReshape = [](const NodePtr& node) {
        if (node->getChildEdges().size() != 1 || node->getOutputShapeAtPort(0).isDynamic() ||
            node->getInputShapeAtPort(0).isDynamic()) {
            return false;
        }
        // Reshape supported only in one case: if one of the input dims is split into 2 consecutive dims
        const auto& inDims = node->getInputShapeAtPort(0).getDims();
        const auto& outDims = node->getOutputShapeAtPort(0).getDims();
        if (outDims.size() - inDims.size() != 1) {
            return false;
        }

        size_t mismatchCount = 0;
        for (size_t i = 0; i < inDims.size(); ++i) {
            if (i + mismatchCount >= outDims.size()) {
                return false;
            }
            if (inDims[i] != outDims[i + mismatchCount]) {
                mismatchCount++;
            }
        }
        return mismatchCount == 1;
    };

    auto isSuitableReorder = [](const NodePtr& node) {
        return node->getType() == Type::Reorder &&
               !node->isDynamicNode();  // TODO [DS]: enable for dynamic shapes when inPlace in the dynamic case is
                                        // available (CVS-74863)
    };

    auto updateOrder = [](const VectorDims& originalOrder, const NodePtr& reshape) {
        if (!reshape) {
            return originalOrder;
        }

        // Further logic works with transpose order without Reshape.
        // If there is a Reshape node, which splits one of the dimensions into 2 consecutive ones,
        // the order must be updated as if Transpose is done after Reshape
        // Example. For this sequence:
        // [1,12,5] -> Transpose(0,2,1) -> Reshape(1,5,3,4) -> [1,5,3,4]
        // updated order must be (0,3,1,2):
        // - dim with idx=1 is split into 2 parts: 1 and 2
        // - dim idxes which was greater then 1, increments by 1
        const auto& reshapeInShape = reshape->getInputShapeAtPort(0).getDims();
        const auto& reshapeOutShape = reshape->getOutputShapeAtPort(0).getDims();
        const size_t splitDimIdx = [&]() {
            for (size_t i = 0; i < reshapeInShape.size(); ++i) {
                if (reshapeInShape[i] != reshapeOutShape[i]) {
                    for (size_t j = 0; j < originalOrder.size(); ++j) {
                        if (originalOrder[j] == i) {
                            return j;
                        }
                    }
                }
            }
            OPENVINO_THROW("splitDimIdx can not be found");
        }();

        auto transformedOrder = originalOrder;
        auto insertIt = transformedOrder.end();
        for (auto it = transformedOrder.begin(); it != transformedOrder.end(); ++it) {
            auto& elem = *it;
            if (elem > splitDimIdx) {
                elem++;
            } else if (elem == splitDimIdx) {
                insertIt = it + 1;
            }
        }
        transformedOrder.insert(insertIt, splitDimIdx + 1);
        return transformedOrder;
    };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto parentNode = graphNodes[i];
        if (!isSuitableTranspose(parentNode)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(MergeTransposeAndReorder_ParentNode);

        auto childNode = parentNode->getChildEdgesAtPort(0).front()->getChild();
        NodePtr intermNode;
        if (childNode->getType() == Type::Reshape) {
            intermNode = childNode;
            if (!isSuitableReshape(intermNode)) {
                continue;
            }
            childNode = intermNode->getChildEdgesAtPort(0).front()->getChild();
        }
        if (!isSuitableReorder(childNode)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(MergeTransposeAndReorder_ChildNode);

        const auto transposeNode = std::dynamic_pointer_cast<Transpose>(parentNode);
        const auto reorderNode = std::dynamic_pointer_cast<Reorder>(childNode);
        std::shared_ptr<Reshape> reshapeNode =
            intermNode != nullptr ? std::dynamic_pointer_cast<Reshape>(intermNode) : nullptr;
        if (!transposeNode || !reorderNode || (intermNode && !reshapeNode)) {
            continue;
        }

        auto transposeOrder = updateOrder(transposeNode->getOrder(), reshapeNode);
        auto descBeforeReorder = reorderNode->getParentEdgeAt(0)
                                     ->getParent()
                                     ->getSelectedPrimitiveDescriptor()
                                     ->getConfig()
                                     .outConfs[0]
                                     .getMemDesc();
        auto layoutOrder = descBeforeReorder->as<BlockedMemoryDesc>()->getOrder();

        auto inBlockedDesc =
            reorderNode->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].getMemDesc()->as<BlockedMemoryDesc>();
        auto outBlockedDesc = reorderNode->getSelectedPrimitiveDescriptor()
                                  ->getConfig()
                                  .outConfs[0]
                                  .getMemDesc()
                                  ->as<BlockedMemoryDesc>();

        auto& inOrder = inBlockedDesc->getOrder();
        auto& outOrder = outBlockedDesc->getOrder();

        if (checkAscendingFinalOrder(transposeOrder, layoutOrder, inOrder, outOrder)) {
            mergeTransposeReshapeReorder(graph, transposeNode, reshapeNode, reorderNode, false);
        }
    }
}

void GraphOptimizer::MergeReorderAndTranspose(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableTranspose = [](const NodePtr& node) {
        return node->getType() == Type::Transpose && node->getChildEdges().size() == 1 && !node->isDynamicNode();
    };

    auto isSuitableReshape = [](const NodePtr& node) {
        if (node->getChildEdges().size() != 1 || node->getOutputShapeAtPort(0).isDynamic() ||
            node->getInputShapeAtPort(0).isDynamic()) {
            return false;
        }
        // Reshape supported only in one case: if two consecutive input dims are merged into 1
        const auto& inShape = node->getInputShapeAtPort(0).getDims();
        const auto& outShape = node->getOutputShapeAtPort(0).getDims();
        if (inShape.size() - outShape.size() != 1) {
            return false;
        }

        size_t mismatchCount = 0;
        for (size_t i = 0; i < outShape.size(); ++i) {
            if (i + mismatchCount >= inShape.size()) {
                return false;
            }
            if (outShape[i] != inShape[i + mismatchCount]) {
                mismatchCount++;
            }
        }
        return mismatchCount == 1;
    };

    auto isSuitableReorder = [](const NodePtr& node) {
        return node->getType() == Type::Reorder && node->getChildEdges().size() == 1 && !node->isDynamicNode();
    };

    auto updateOrder = [](const VectorDims& originalOrder, const NodePtr& reshape) {
        if (!reshape) {
            return originalOrder;
        }

        // Further logic works with order without Reshape.
        // If there is Reshape node which merges 2 consecutive dims into one,
        // the order must be updated as like Transpose is done before Reshape
        // Example. For this sequence:
        // [1,3,4,5] -> Reshape(1,12,5) -> Transpose(0,2,1) -> [1,5,12]
        // updated order must be (0,3,1,2):
        // - dim with idx=2 is split into 2 parts: 2 and 3
        const auto& reshapeInShape = reshape->getInputShapeAtPort(0).getDims();
        const auto& reshapeOutShape = reshape->getOutputShapeAtPort(0).getDims();
        const size_t mergedDimIdx = [&]() {
            for (size_t i = 0; i < reshapeInShape.size(); ++i) {
                if (reshapeInShape[i] != reshapeOutShape[i]) {
                    return i;
                }
            }
            OPENVINO_THROW("mergedDimIdx can not be found");
        }();

        auto transformedOrder = originalOrder;
        auto insertIt = transformedOrder.end();
        for (auto it = transformedOrder.begin(); it != transformedOrder.end(); ++it) {
            auto& elem = *it;
            if (elem > mergedDimIdx) {
                elem++;
            } else if (elem == mergedDimIdx) {
                insertIt = it + 1;
            }
        }

        transformedOrder.insert(insertIt, mergedDimIdx + 1);
        return transformedOrder;
    };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto parentNode = graphNodes[i];
        if (!isSuitableReorder(parentNode)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(MergeTransposeAndReorder_ParentNode);

        auto childNode = parentNode->getChildEdgesAtPort(0).front()->getChild();
        NodePtr intermNode;
        if (childNode->getType() == Type::Reshape) {
            intermNode = childNode;
            if (!isSuitableReshape(intermNode)) {
                continue;
            }
            childNode = intermNode->getChildEdgeAt(0)->getChild();
        }
        if (!isSuitableTranspose(childNode)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(MergeTransposeAndReorder_ChildNode);

        auto transposeNode = std::dynamic_pointer_cast<Transpose>(childNode);
        auto reorderNode = std::dynamic_pointer_cast<Reorder>(parentNode);
        std::shared_ptr<Reshape> reshapeNode =
            intermNode != nullptr ? std::dynamic_pointer_cast<Reshape>(intermNode) : nullptr;
        if (!transposeNode || !reorderNode || (intermNode && !reshapeNode)) {
            continue;
        }

        auto transposeOrder = updateOrder(transposeNode->getOrder(), reshapeNode);
        auto descAfterTranspose = transposeNode->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].getMemDesc();
        auto layoutOrder = updateOrder(descAfterTranspose->as<BlockedMemoryDesc>()->getOrder(), reshapeNode);

        auto inBlockedDesc =
            reorderNode->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].getMemDesc()->as<BlockedMemoryDesc>();
        auto outBlockedDesc = reorderNode->getSelectedPrimitiveDescriptor()
                                  ->getConfig()
                                  .outConfs[0]
                                  .getMemDesc()
                                  ->as<BlockedMemoryDesc>();

        auto& inOrder = inBlockedDesc->getOrder();
        auto& outOrder = outBlockedDesc->getOrder();

        if (checkAscendingFinalOrder(transposeOrder, layoutOrder, inOrder, outOrder)) {
            // Reorder node doesn't support (with rare exceptions) reordering in case of different ranks on input and
            // output. So the merge can be performed only in the case when the fused reorder will be optimized.
            if (parentNode->getInputShapeAtPort(0).getRank() != childNode->getOutputShapeAtPort(0).getRank() &&
                !canBeInplaced(parentNode, childNode)) {
                continue;
            }
            mergeTransposeReshapeReorder(graph, transposeNode, reshapeNode, reorderNode, true);
        }
    }
}

void GraphOptimizer::reshapeRnnSeq(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableParentNode = [](const NodePtr& node) {
        if (node->type != Type::RNNSeq) {
            return false;
        }
        auto rnnNode = std::dynamic_pointer_cast<RNN>(node);
        return rnnNode && !rnnNode->hasNativeOrder() && node->outputShapes[0].getRank() == 4 &&
               node->outputShapes[0].getDims()[1] == 1;
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

            const auto secondInput =
                std::make_shared<ov::opset1::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{1});
            const auto unsqueeze = std::make_shared<ov::opset1::Unsqueeze>(
                std::make_shared<ov::opset1::Parameter>(parentNode->getOriginalOutputPrecisionAtPort(0),
                                                        parentNode->getOutputShapeAtPort(0).toPartialShape()),
                secondInput);
            unsqueeze->set_friendly_name(parentNode->getName() + "_abc_a1bc_" + std::to_string(j));

            const auto cpuUnsqueeze = std::make_shared<Reshape>(unsqueeze, graph.getGraphContext());
            graph.InsertNode(edge, cpuUnsqueeze, false);

            const auto cpuConstant = std::make_shared<node::Input>(secondInput, graph.getGraphContext());
            graph.AddNode(cpuConstant);
            graph.CreateEdge(cpuConstant, cpuUnsqueeze, 0, 1);
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

    auto isSuitableParentNode = [](const NodePtr& parentNode) {
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

void GraphOptimizer::RemoveMemoryInputConvert(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableNode = [](const NodePtr& node) {
        if (Type::Convert != node->getType()) {
            return false;
        }

        auto parent = node->getParentEdgeAt(0)->getParent();
        if (Type::MemoryInput != parent->getType()) {
            return false;
        }

        return true;
    };

    for (const auto& node : graphNodes) {
        if (!isSuitableNode(node)) {
            continue;
        }
        graph.DropNode(node);
    }
}

void GraphOptimizer::RemoveConvertMemoryOutput(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableNode = [](const NodePtr& node) {
        if (Type::Convert != node->getType()) {
            return false;
        }

        auto&& childEdges = node->getChildEdgesAtPort(0);
        for (auto&& edge : childEdges) {
            if (Type::MemoryOutput != edge->getChild()->getType()) {
                return false;
            }
        }

        return true;
    };

    for (const auto& node : graphNodes) {
        if (!isSuitableNode(node)) {
            continue;
        }
        graph.DropNode(node);
    }
}

void GraphOptimizer::MatchSdpaKvCache(Graph& graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSuitableMemInput = [](const NodePtr& node) -> bool {
        if (Type::MemoryInput != node->getType()) {
            return false;
        }
        NodePtr childSdpa = nullptr;
        auto&& childEdges = node->getChildEdgesAtPort(0);
        for (auto&& item : childEdges) {
            auto childNode = item->getChild();
            if (!one_of(childNode->getType(), Type::ScaledDotProductAttention, Type::ShapeOf)) {
                return false;
            }

            if (Type::ScaledDotProductAttention == childNode->getType()) {
                if (childSdpa && childSdpa != childNode) {
                    // only one child SDPA supported
                    return false;
                }
                childSdpa = childNode;
            }
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(MatchSdpaKvCache_isSuitableMemInput);

        auto memInputNode = std::dynamic_pointer_cast<node::MemoryInputBase>(node);
        OPENVINO_ASSERT(memInputNode, "MemoryInput node ", node->getName(), " has unexpected dynamic type");
        auto& memOutputNode = memInputNode->getOutputNode();
        auto memOutputParent = memOutputNode.getParentEdgeAt(0)->getParent();
        if (memOutputParent != childSdpa) {
            return false;
        }
        return true;
    };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto node = graphNodes[i];
        if (!isSuitableMemInput(node)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(MatchSdpaKvCache_Node);

        // Node is already modified
        if (auto sdpaMemInput = std::dynamic_pointer_cast<node::MemoryInputSDPA>(node)) {
            continue;
        }

        auto memInputNode = std::dynamic_pointer_cast<node::MemoryInputBase>(node);
        OPENVINO_ASSERT(memInputNode, "MemoryInput node ", node->getName(), " has unexpected dynamic type");

        std::optional<std::vector<Shape>> inputShapes;
        std::optional<std::vector<ov::element::Type>> inputPrcs;
        if (!node->getParentEdges().empty()) {
            inputShapes = std::optional<std::vector<Shape>>(std::vector<Shape>{});
            inputPrcs = std::optional<std::vector<ov::element::Type>>(std::vector<ov::element::Type>{});

            auto& input_shape_vec = *inputShapes;
            auto& input_prc_vec = *inputPrcs;

            for (size_t i = 0; i < node->getParentEdges().size(); i++) {
                input_shape_vec.push_back(node->getInputShapeAtPort(i));
                input_prc_vec.push_back(node->getOriginalInputPrecisionAtPort(i));
            }
        }

        // search for SDPA
        std::shared_ptr<ScaledDotProductAttention> sdpa;
        for (auto&& edge : node->getChildEdgesAtPort(0)) {
            auto child = edge->getChild();
            if (Type::ScaledDotProductAttention == child->getType()) {
                sdpa = std::dynamic_pointer_cast<ScaledDotProductAttention>(child);
                if (sdpa) {
                    break;
                }
                OPENVINO_THROW("Couldn't cast node", child->getName(), " to ScaledDotProductAttention type");
            }
        }

        // capture reference to the original mem output before graph transformations
        auto& memOutput = memInputNode->getOutputNode();

        auto memInputSdpa = std::make_shared<MemoryInputSDPA>(memInputNode->getId(),
                                                              memInputNode->getName(),
                                                              memInputNode->getTypeStr(),
                                                              memInputNode->getOutputShapeAtPort(0),
                                                              memInputNode->getOriginalOutputPrecisionAtPort(0),
                                                              graph.getGraphContext(),
                                                              inputShapes,
                                                              inputPrcs,
                                                              sdpa);

        if (!memInputNode->getParentEdges().empty()) {
            auto parentEdge = memInputNode->getParentEdgeAt(0);
            auto parent = parentEdge->getParent();
            const auto inputNum = parentEdge->getInputNum();
            graph.RemoveEdge(parentEdge);
            graph.CreateEdge(parent, memInputSdpa, inputNum, 0);
        }

        for (auto&& edge : memInputNode->getChildEdgesAtPort(0)) {
            auto child = edge->getChild();
            const auto outputNum = edge->getOutputNum();
            graph.RemoveEdge(edge);
            graph.CreateEdge(memInputSdpa, child, 0, outputNum);
        }

        // create a stub memory output
        auto memOutputStub = std::make_shared<MemoryOutputStub>(memOutput.getId(),
                                                                memOutput.getName(),
                                                                memOutput.getTypeStr(),
                                                                memOutput.getInputShapeAtPort(0),
                                                                memOutput.getOriginalInputPrecisionAtPort(0),
                                                                graph.getGraphContext());

        auto memOutputEdge = memOutput.getParentEdgeAt(0);
        const auto inputNum = memOutputEdge->getInputNum();
        graph.RemoveEdge(memOutputEdge);
        graph.CreateEdge(sdpa, memOutputStub, inputNum, 0);

        graph.AddNode(memInputSdpa);
        graph.AddNode(memOutputStub);
    }
}

void GraphOptimizer::DropRedundantMemoryOutput(Graph& graph) {
    // When we have a MemoryInput->MemoryOutput pair, that means that the state is immediately populated with the init
    // subgraph values when the init subgraph exists. In all the other cases the state is simply a read only object.
    // We can optimize such a case removing the MemoryOutput node and transferring the state values update
    // responsibility to a special type of the MemoryInput node - MemoryInputSingle
    auto& graphNodes = graph.GetNodes();

    auto isSuitableMemInput = [](const NodePtr& node) -> bool {
        if (Type::MemoryInput != node->getType()) {
            return false;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(DropRedundantMemoryOutput_isSuitableMemInput);

        auto memInputBase = std::dynamic_pointer_cast<MemoryNode>(node);
        OPENVINO_ASSERT(memInputBase,
                        "Unexpectedly wrong dynamic type of node: ",
                        node->getName(),
                        " of type: ",
                        node->getTypeStr());

        auto id = memInputBase->getId();

        NodePtr MemoryOutput = nullptr;
        auto&& childEdges = node->getChildEdgesAtPort(0);
        for (auto&& item : childEdges) {
            auto childNode = item->getChild();

            if (Type::MemoryOutput == childNode->getType()) {
                auto memOutputBase = std::dynamic_pointer_cast<MemoryNode>(childNode);
                OPENVINO_ASSERT(memInputBase,
                                "Unexpectedly wrong dynamic type of node: ",
                                node->getName(),
                                " of type: ",
                                node->getTypeStr());

                if (memOutputBase->getId() != id) {
                    return false;  // an Assign node from different Variable is attached
                }

                if (MemoryOutput && MemoryOutput != childNode) {
                    // only one child MemoryOutput is expected
                    return false;
                }
                MemoryOutput = childNode;
            }
        }
        return nullptr != MemoryOutput;
    };

    for (size_t i = 0; i < graphNodes.size(); i++) {
        auto node = graphNodes[i];
        if (!isSuitableMemInput(node)) {
            continue;
        }

        CPU_GRAPH_OPTIMIZER_SCOPE(DropRedundantMemoryOutput_Node);

        auto memInputNode = std::dynamic_pointer_cast<node::MemoryInputBase>(node);
        OPENVINO_ASSERT(memInputNode, "MemoryInput node ", node->getName(), " has unexpected dynamic type");

        std::optional<std::vector<Shape>> inputShapes;
        std::optional<std::vector<ov::element::Type>> inputPrcs;
        if (!node->getParentEdges().empty()) {
            inputShapes = std::optional<std::vector<Shape>>(std::vector<Shape>{});
            inputPrcs = std::optional<std::vector<ov::element::Type>>(std::vector<ov::element::Type>{});

            auto& input_shape_vec = *inputShapes;
            auto& input_prc_vec = *inputPrcs;
            for (size_t i = 0; i < node->getParentEdges().size(); i++) {
                input_shape_vec.push_back(node->getInputShapeAtPort(i));
                input_prc_vec.push_back(node->getOriginalInputPrecisionAtPort(i));
            }
        }

        // search for the MemoryOutputNode
        NodePtr memoryOutputNode;
        for (auto&& edge : node->getChildEdgesAtPort(0)) {
            auto child = edge->getChild();
            if (Type::MemoryOutput == child->getType()) {
                memoryOutputNode = child;
                break;
            }
        }
        OPENVINO_ASSERT(memoryOutputNode, "Corresponding MemoryOutput has not been found");

        graph.RemoveEdge(memoryOutputNode->getParentEdgeAt(0));
        // there are no output edges from MemoryOutput nodes

        CPU_GRAPH_OPTIMIZER_SCOPE(DropRedundantMemoryOutput_SubGraph);
        auto memInpNd = std::dynamic_pointer_cast<node::MemoryInput>(node);
        OPENVINO_ASSERT(memInpNd, "MemoryInput node ", node->getName(), " has unexpected dynamic type");

        // now replace the existing MemoryInput with a special type that works without the corresponding MemoryOutput
        auto memInputSingle = std::make_shared<MemoryInputSingle>(memInputNode->getId(),
                                                                  memInputNode->getName(),
                                                                  memInputNode->getTypeStr(),
                                                                  memInputNode->getOutputShapeAtPort(0),
                                                                  memInputNode->getOriginalOutputPrecisionAtPort(0),
                                                                  graph.getGraphContext(),
                                                                  inputShapes,
                                                                  inputPrcs,
                                                                  memInpNd->getSubGraph());
        graph.AddNode(memInputSingle);

        if (!memInputNode->getParentEdges().empty()) {
            auto parentEdgeNum = memInputNode->getParentEdges().size();
            std::vector<ov::intel_cpu::EdgePtr> parentEdges;
            for (size_t i = 0; i < parentEdgeNum; i++) {
                auto parentEdge = memInputNode->getParentEdgeAt(i);
                auto parent = parentEdge->getParent();
                const auto inputNum = parentEdge->getInputNum();
                parentEdges.push_back(parentEdge);
                graph.CreateEdge(parent, memInputSingle, inputNum, parentEdge->getOutputNum());
            }
            for (const auto& parentEdge : parentEdges) {
                graph.RemoveEdge(parentEdge);
            }
        }

        for (auto&& edge : memInputNode->getChildEdgesAtPort(0)) {
            auto child = edge->getChild();
            const auto outputNum = edge->getOutputNum();
            graph.RemoveEdge(edge);
            graph.CreateEdge(memInputSingle, child, 0, outputNum);
        }
    }
}

}  // namespace ov::intel_cpu

// NOLINTEND(modernize-loop-convert)
