// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <blob_factory.hpp>
#include "nodes/mkldnn_reshape_node.h"
#include "mkldnn_graph_optimizer.h"
#include <nodes/mkldnn_activation_node.h>
#include "nodes/mkldnn_pooling_node.h"
#include "nodes/mkldnn_eltwise_node.h"
#include "nodes/mkldnn_depthwise_node.h"
#include "nodes/mkldnn_concat_node.h"
#include "nodes/mkldnn_reorder_node.h"

#include <string>
#include <list>
#include <memory>
#include <set>
#include <ie_layers_internal.hpp>
#include <nodes/mkldnn_bin_conv_node.h>
#include <nodes/mkldnn_quantize_node.h>
#include "cpu_isa_traits.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNGraphOptimizer::MKLDNNGraphOptimizer() {}

void MKLDNNGraphOptimizer::ApplyCommonGraphOptimizations(MKLDNNGraph &graph) {
    MergeGroupConvolution(graph);
    graph.RemoveDroppedNodes();

    FuseConvolutionAndDepthwise(graph);
    graph.RemoveDroppedNodes();

    FuseConvolutionAndActivation(graph);
    graph.RemoveDroppedNodes();

    FuseConvolutionAndDepthwise(graph);
    graph.RemoveDroppedNodes();

    FuseConvolutionAndDWConvolution(graph);
    graph.RemoveDroppedNodes();

    FuseBinaryConvolutionAndQuantize(graph);
    graph.RemoveDroppedNodes();

    FuseBatchNormWithScale(graph);
    graph.RemoveDroppedNodes();

    FuseFullyConnectedAndActivation(graph);
    graph.RemoveDroppedNodes();

    RemoveIdentityOperator(graph);
    graph.RemoveDroppedNodes();

    FuseConvolutionSumAndConvolutionSumActivation(graph);
    graph.RemoveDroppedNodes();


    graph.RemoveDroppedEdges();
}

void MKLDNNGraphOptimizer::ApplyImplSpecificGraphOptimizations(MKLDNNGraph &graph) {
    RemoveIOScaleShifts(graph);
    graph.RemoveDroppedNodes();

    DropDoubleReorders(graph);
    graph.RemoveDroppedNodes();


    graph.RemoveDroppedEdges();
}

void MKLDNNGraphOptimizer::MergeGroupConvolution(MKLDNNGraph &graph) {
    for (auto node : graph.GetNodes()) {
        // Split with at least 2 Convolutions
        if (!IsOneOf(node->getType(), {Split}) || node->getChildEdges().size() < 2 ||
                !IsOneOf(node->getChildEdgeAt(0)->getChild()->getType(), {Convolution, Convolution_Activation})) {
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
        for (size_t i = 1; i < split->getChildEdges().size(); i++) {
            auto peerInEdge = split->getChildEdgeAt(i);
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
             isOneOf(activationNode->getAlgorithm(), {eltwise_elu, eltwise_logistic, eltwise_bounded_relu, eltwise_clamp})));
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        if (graphNodes[i]->getType() == Convolution || graphNodes[i]->getType() == BinaryConvolution) {
            auto conv = graphNodes[i];

            auto fuse = [&] (MKLDNNNodePtr relu) {
                if (graphNodes[i]->getType() != BinaryConvolution)
                    conv->setType(Convolution_Activation);
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
                        bool is_max_pool =
                                dynamic_cast<PoolingLayer *>(pool->getCnnLayer().get())->_type ==
                                PoolingLayer::PoolType::MAX;

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

void MKLDNNGraphOptimizer::FuseConvolutionAndDepthwise(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isSutableParentNode = [](MKLDNNNodePtr node) {
        bool isSutableConv = (node->getType() == Convolution || node->getType() == Convolution_Activation) &&
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
        return ((depthwiseNode->getAlgorithm() == mkldnn::algorithm::depthwise_scale_shift && depthwiseNode->isWithBiases()) ||
                (depthwiseNode->getAlgorithm() == mkldnn::algorithm::depthwise_prelu));
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto conv = graphNodes[i];
        if (!isSutableParentNode(conv)) continue;

        auto depthwise0 = conv->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(depthwise0)) continue;

        conv->fuseWith(depthwise0);
        if (conv->type != BinaryConvolution)
            conv->setType(Convolution_Depthwise);

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

void MKLDNNGraphOptimizer::FuseConvolutionAndDWConvolution(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isConvolutionNode = [](MKLDNNNodePtr node) {
        return node->getType() == Convolution || node->getType() == Convolution_Activation;
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

            bool isSupportedParams = layer->_group == 1;
            if (!isSupportedParams) return false;
        } else {
            auto *layer = dynamic_cast<ConvolutionLayer *>(node->getCnnLayer().get());

            bool isSupportedParams = layer->_group == 1 &&
                                     ((is1x1Convolution(layer) && layer->_stride[X_AXIS] == 1 &&
                                       layer->_stride[Y_AXIS] == 1) || !is1x1Convolution(layer)) &&
                                     (layer->precision == Precision::FP32 || layer->precision == Precision::I8);
            if (!isSupportedParams) return false;
        }

        return node->getChildEdges().size() == 1 && isConvolutionNode(node->getChildEdgeAt(0)->getChild());
    };

    auto isSutableChildConvolution = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        auto* childLayer = dynamic_cast<ConvolutionLayer*>(childNode->getCnnLayer().get());

        if (!isBinaryConvolutionNode(parentNode)) {
            auto* parentLayer = dynamic_cast<ConvolutionLayer*>(parentNode->getCnnLayer().get());
            if (parentLayer->precision != childLayer->precision)
                return false;
        }

        auto allPads = getPaddings(*childLayer);
        bool isSupportedParams = childLayer->_out_depth == childLayer->_group &&
                                 childLayer->_out_depth != 1 &&
                                 // Depthwise convolution output should be multiple of 8
                                 childLayer->_kernel[X_AXIS] == 3 && childLayer->_kernel[Y_AXIS] == 3 &&
                                 allPads.begin[X_AXIS] == 1 && allPads.begin[Y_AXIS] == 1 &&
                                 childLayer->_dilation[X_AXIS] == 1 && childLayer->_dilation[Y_AXIS] == 1 &&
                                 childLayer->_biases != nullptr && childLayer->_biases->size() != 0;

        return isSupportedParams;
    };

    auto isFusingWorthwhile = [&](MKLDNNNodePtr parentNode, MKLDNNNodePtr childNode) {
        if (isBinaryConvolutionNode(parentNode)) {
            return true;
        }

        auto* layer = dynamic_cast<ConvolutionLayer*>(childNode->getCnnLayer().get());

        auto inDims = childNode->inDims[0];
        auto outDims = childNode->outDims[0];
        int elemSize = MKLDNNExtensionUtils::sizeOfDataType(MKLDNNExtensionUtils::IEPrecisionToDataType(layer->precision));

        int L3_cache_size = mkldnn_get_cache_size(3, false);
        int dw_conv_input_size = inDims[0] * inDims[1] * inDims[2] * inDims[3] * elemSize;
        int dw_conv_output_size = outDims[0] * outDims[1]* outDims[2] * outDims[3] * elemSize;

        bool isInt8 = layer->precision == Precision::I8 || layer->precision == Precision::U8;
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
        graph.DropNode(childConvNode);
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

        auto* quantizeLayer = dynamic_cast<QuantizeLayer*>(node->getCnnLayer().get());
        bool isSutableQuantize = node->getType() == Quantize && quantizeLayer->levels == 2;

        return isSutableQuantize;
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        auto parent = graphNodes[i];
        if (!isSutableParentNode(parent)) continue;

        auto child = parent->getChildEdgeAt(0)->getChild();
        if (!isSutableChildNode(child)) continue;

        parent->fuseWith(child);

        auto* binConvNode = dynamic_cast<MKLDNNBinaryConvolutionNode*>(parent.get());

        auto parents = child->parentEdges;
        for (size_t i = 0; i < parents.size(); i++) {
            auto p_edge = parents[i].lock();
            if (p_edge->getParent()->getType() == Input) {
                InferenceEngine::SizeVector dims;
                dims.push_back(binConvNode->getChildEdgeAt(0)->getDims()[1]);

                auto InputLowBlob = dynamic_cast<TBlob<float>*>(p_edge->getParent()->getCnnLayer()->blobs["custom"].get());

                auto inputLowData = InputLowBlob->buffer().as<float*>();
                int inputLowAxis = p_edge->getDims().ndims() == 1 ? 0 : 1;
                bool isInputLowBroadcasted = p_edge->getDims()[inputLowAxis] != dims[0];

                for (int i = 0; i < dims[0]; i++) {
                    binConvNode->pushBinarizationThreshold(inputLowData[isInputLowBroadcasted ? 0 : i]);
                }

                break;
            }
        }

        for (size_t i = 0; i < parents.size(); i++) {
            auto p_edge = parents[i].lock();
            if (p_edge->getParent()->getType() == BinaryConvolution)
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

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(activation.get());

        return activationNode &&
            (activationNode->getAlgorithm() == eltwise_relu ||
            (conv->getCnnLayer()->precision == Precision::FP32 &&
             isOneOf(activationNode->getAlgorithm(), {eltwise_elu, eltwise_logistic, eltwise_bounded_relu, eltwise_clamp})));
    };

    for (auto &graphNode : graphNodes) {
        if (graphNode->getType() != Eltwise)
            continue;

        if (!std::dynamic_pointer_cast<MKLDNNEltwiseNode>(graphNode)->isSum()) continue;
        if (!std::dynamic_pointer_cast<MKLDNNEltwiseNode>(graphNode)->isUnitScales()) continue;

        auto parent1 = graphNode->getParentEdgeAt(0)->getParent();
        auto parent2 = graphNode->getParentEdgeAt(1)->getParent();
        // TODO: Enlarge to several inputs
        if (graphNode->getParentEdges().size() != 2 ||
            (parent1->getType() != Convolution && parent1->getType() != BinaryConvolution &&
             parent2->getType() != Convolution && parent2->getType() != BinaryConvolution))
            continue;

        auto mergedConv = (parent1->getType() == Convolution || parent1->getType() == BinaryConvolution) ? parent1 : parent2;
        auto peerNode = (parent1->getType() == Convolution || parent1->getType() == BinaryConvolution) ? parent2 : parent1;
        if ((peerNode->getType() == Convolution || peerNode->getType() == BinaryConvolution) &&
            mergedConv->getChildEdges().size() != 1) {
            mergedConv = parent2;
            peerNode = parent1;
        }
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
            if (mergedConv->getType() != BinaryConvolution)
                mergedConv->setType(Convolution_Sum_Activation);
            mergedConv->fuseWith(sum);
        } else {
            if (mergedConv->getType() != BinaryConvolution)
                mergedConv->setType(Convolution_Sum);
        }

        mergedConv->fuseWith(lastNode);

        if (mergedConv->fusedWith.size() > 0 &&
           (mergedConv->fusedWith[0]->getType() == Convolution || mergedConv->fusedWith[0]->getType() == BinaryConvolution)) {
            // Merged with DW_conv. Shape may change
            mergedConv->inDims.push_back(mergedConv->fusedWith[0]->outDims[0]);
        } else {
            mergedConv->inDims.push_back(mergedConv->outDims[0]);
        }

        size_t childIdx = 0;
        for (childIdx = 0; childIdx < peerNode->getChildEdges().size(); childIdx++) {
            if (peerNode->getChildEdgeAt(childIdx)->getChild() == sum) {
                break;
            }
        }

        int peer_port = peerNode->getChildEdgeAt(childIdx)->getInputNum();
        peerNode->getChildEdgeAt(childIdx)->drop();

        MKLDNNEdgePtr edgePtr(new MKLDNNEdge(peerNode, mergedConv, peer_port, 1));
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

void MKLDNNGraphOptimizer::FuseFullyConnectedAndActivation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isFusingSupported = [&](MKLDNNNodePtr fc, MKLDNNNodePtr activation) {
        if (!activation->getCnnLayer())
            return false;

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(activation.get());

        // TODO: fuse on fp32 not optimized yet in mkl-dnn
        return activationNode && fc->getCnnLayer()->precision != Precision::FP32 &&
            (activationNode->getAlgorithm() == eltwise_relu);
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        if (graphNodes[i]->getType() == FullyConnected) {
            auto fc = graphNodes[i];

            auto fuse = [&] (MKLDNNNodePtr relu) {
                fc->setType(FullyConnected_Activation);
                fc->fuseWith(relu);
            };

            if (fc->getChildEdges().size() == 1) {
                auto ch1 = fc->getChildEdgeAt(0)->getChild();

                if (isFusingSupported(fc, ch1)) {
                    fuse(ch1);
                    graph.DropNode(ch1);
                }
            }
        }
    }
}

void MKLDNNGraphOptimizer::RemoveIdentityOperator(MKLDNNGraph &graph) {
    for (MKLDNNNodePtr& node : graph.GetNodes()) {
        bool toDrop = false;

        if (node->getType() == Power) {
            PowerLayer* l = dynamic_cast<PowerLayer*>(node->getCnnLayer().get());

            if (l->power == 1.0f && l->scale == 1.0f && l->offset == 0.0f) toDrop = true;
        }

        if (node->getType() == Depthwise && node->getCnnLayer()->type == "ScaleShift") {
            ScaleShiftLayer* l = dynamic_cast<ScaleShiftLayer*>(node->getCnnLayer().get());

            if (l->_weights == nullptr && l->_biases == nullptr) toDrop = true;
        }

        if (node->getType() == Copy) toDrop = true;

        if (toDrop) graph.DropNode(node);
    }
}

void MKLDNNGraphOptimizer::DropDoubleReorders(MKLDNNGraph &graph) {
    std::set<MKLDNNNodePtr> processed;
    std::vector<MKLDNNNodePtr> newNodes;
    for (MKLDNNNodePtr& node : graph.GetNodes()) {
        if (processed.find(node) == processed.end() && node->getType() == Reorder
            && node->getChildEdges().size() == 1
            && node->getChildEdgeAt(0)->getChild()->getType() == Reorder ) {
            auto nextNode = node->getChildEdgeAt(0)->getChild();
            MKLDNNReorderNode* n = dynamic_cast<MKLDNNReorderNode*>(node.get());
            MKLDNNReorderNode* nn = dynamic_cast<MKLDNNReorderNode*>(nextNode.get());

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
            MKLDNNNodePtr newReorder(new MKLDNNReorderNode(layer, graph.getEngine()));
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

void MKLDNNGraphOptimizer::RemoveIOScaleShifts(MKLDNNGraph &graph) {
    for (MKLDNNNodePtr& node : graph.GetNodes()) {
        if (node->getType() == Depthwise && node->getCnnLayer()->type == "ScaleShift") {
            ScaleShiftLayer* l = dynamic_cast<ScaleShiftLayer*>(node->getCnnLayer().get());

            auto cur = l->insData[0].lock();
            if (cur == nullptr) {
                THROW_IE_EXCEPTION << "[MKLDNN] error - invalid input data";
            }
            if (cur->precision != l->outData[0]->precision) {
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


