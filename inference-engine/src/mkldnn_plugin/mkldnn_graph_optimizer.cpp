// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <list>
#include <memory>
#include <set>
#include <nodes/mkldnn_activation_node.h>
#include "mkldnn_graph_optimizer.h"
#include "nodes/mkldnn_pooling_node.h"
#include "nodes/mkldnn_eltwise_node.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNGraphOptimizer::MKLDNNGraphOptimizer() {}

void MKLDNNGraphOptimizer::Optimize(MKLDNNGraph &graph) {
    MergeGroupConvolution(graph);
    RemoveDropped(graph);

    FuseConvolutionAndActivation(graph);
    RemoveDropped(graph);

    FuseConvolutionAndDWConvolution(graph);
    RemoveDropped(graph);

    FuseBatchNormWithScale(graph);
    RemoveDropped(graph);

    RemoveIdentityOperator(graph);
    RemoveDropped(graph);

    FuseConvolutionSumAndConvolutionSumActivation(graph);
    RemoveDropped(graph);

    RemoveDroppedEdges(graph);
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

        DropNode(graph, split);
        DropNode(graph, concat);
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
                        DropNode(graph, child);
                    }
                }
            }
        }
    }
}

void MKLDNNGraphOptimizer::FuseConvolutionAndActivation(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isFusingSupported = [&](MKLDNNNodePtr node) {
        if (!node->getCnnLayer())
            return false;

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());

        return activationNode &&
               (activationNode->getAlgorithm() == mkldnn::algorithm::eltwise_relu           ||
                activationNode->getAlgorithm() == mkldnn::algorithm::eltwise_elu            ||
                activationNode->getAlgorithm() == mkldnn::algorithm::eltwise_logistic       ||
                activationNode->getAlgorithm() == mkldnn::algorithm::eltwise_bounded_relu   ||
                activationNode->getAlgorithm() == mkldnn::algorithm::eltwise_clamp);
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        if (graphNodes[i]->getType() == Convolution) {
            auto conv = graphNodes[i];

            auto fuse = [&] (MKLDNNNodePtr relu) {
                conv->setType(Convolution_Activation);
                conv->fuseWith(relu);
                DropNode(graph, relu);
            };

            if (conv->getChildEdges().size() == 1) {
                auto ch1 = conv->getChildEdgeAt(0)->getChild();

                if (isFusingSupported(ch1)) {
                    fuse(ch1);
                } else {
                    if (ch1->type == Pooling) {
                        auto pool = ch1;
                        bool is_max_pool =
                                dynamic_cast<PoolingLayer *>(pool->getCnnLayer().get())->_type ==
                                PoolingLayer::PoolType::MAX;

                        if (is_max_pool && pool->getChildEdges().size() == 1) {
                            auto ch2 = pool->getChildEdgeAt(0)->getChild();
                            if (isFusingSupported(ch2))
                                fuse(ch2);
                        }
                    }
                }
            }
        }
    }
}

void MKLDNNGraphOptimizer::FuseConvolutionAndDWConvolution(MKLDNNGraph &graph) {
    auto& graphNodes = graph.GetNodes();

    auto isConvolutionNode = [](MKLDNNNodePtr node) {
        return node->getType() == Convolution || node->getType() == Convolution_Activation;
    };

    auto is1x1Convolution = [](ConvolutionLayer* layer) {
        return layer->_kernel_x == 1 && layer->_kernel_y == 1;
    };

    auto isSutableParentConvolution = [&](MKLDNNNodePtr node) {
        auto* layer = dynamic_cast<ConvolutionLayer*>(node->getCnnLayer().get());

        bool isSupportedParams = layer->_group == 1 &&
                                 ((is1x1Convolution(layer) &&
                                  layer->_stride_x == 1 && layer->_stride_y == 1) || !is1x1Convolution(layer));
        if (!isSupportedParams) return false;

        return node->getChildEdges().size() == 1 && isConvolutionNode(node->getChildEdgeAt(0)->getChild());
    };

    auto isSutableChildConvolution = [](MKLDNNNodePtr node) {
        auto* layer = dynamic_cast<ConvolutionLayer*>(node->getCnnLayer().get());

        bool isSupportedParams = layer->_out_depth == layer->_group &&
                                 layer->_kernel_x == 3 && layer->_kernel_y == 3 &&
                                 layer->_padding_x == 1 && layer->_padding_y == 1 &&
                                 layer->_dilation_x == 1 && layer->_dilation_y == 1 &&
                                 layer->_biases != nullptr && layer->_biases->size() != 0;
        return isSupportedParams;
    };

    auto isFusingWorthwhile = [](MKLDNNNodePtr node) {
        auto inDims = node->inDims[0];
        auto outDims = node->outDims[0];

        int L3_cache_size = mkldnn_get_cache_size(3, false);
        int dw_conv_input_size = inDims[0] * inDims[1] * inDims[2] * inDims[3] * sizeof(float);
        int dw_conv_output_size = outDims[0] * outDims[1]* outDims[2] * outDims[3] * sizeof(float);
        return (dw_conv_input_size + dw_conv_output_size > L3_cache_size / 2);
    };

    for (int i = 0; i < graphNodes.size(); i++) {
        if (!isConvolutionNode(graphNodes[i])) continue;

        auto parentConvNode = graphNodes[i];
        if (!isSutableParentConvolution(parentConvNode)) continue;

        auto childConvNode = parentConvNode->getChildEdgeAt(0)->getChild();
        if (!isSutableChildConvolution(childConvNode)) continue;

        if (!isFusingWorthwhile(childConvNode)) continue;

        parentConvNode->fuseWith(childConvNode);
        DropNode(graph, childConvNode);
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

    auto isFusingSupported = [&](MKLDNNNodePtr node) {
        if (!node->getCnnLayer())
            return false;

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());

        return activationNode &&
               (activationNode->getAlgorithm() == mkldnn::algorithm::eltwise_relu           ||
                activationNode->getAlgorithm() == mkldnn::algorithm::eltwise_elu            ||
                activationNode->getAlgorithm() == mkldnn::algorithm::eltwise_logistic       ||
                activationNode->getAlgorithm() == mkldnn::algorithm::eltwise_bounded_relu   ||
                activationNode->getAlgorithm() == mkldnn::algorithm::eltwise_clamp);
    };

    for (auto &graphNode : graphNodes) {
        if (graphNode->getType() != Eltwise)
            continue;

        if (!std::dynamic_pointer_cast<MKLDNNEltwiseNode>(graphNode)->isSum()) continue;
        if (!std::dynamic_pointer_cast<MKLDNNEltwiseNode>(graphNode)->isUnitScales()) continue;

        // TODO: Enlarge to several inputs
        if (graphNode->getParentEdges().size() != 2 ||
            (graphNode->getParentEdgeAt(0)->getParent()->getType() != Convolution &&
                    graphNode->getParentEdgeAt(1)->getParent()->getType() != Convolution))
            continue;

        auto parent1 = graphNode->getParentEdgeAt(0)->getParent();
        auto parent2 = graphNode->getParentEdgeAt(1)->getParent();

        auto mergedConv = (parent1->getType() == Convolution) ? parent1 : parent2;
        auto peerNode = (parent1->getType() == Convolution) ? parent2 : parent1;
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
                isFusingSupported(graphNode->getChildEdgeAt(0)->getChild())) {
            auto relu_shared = graphNode->getChildEdgeAt(0)->getChild();
            lastNode = relu_shared;
            mergedConv->setType(Convolution_Sum_Activation);
            mergedConv->fuseWith(sum);
        } else {
            mergedConv->setType(Convolution_Sum);
        }

        mergedConv->fuseWith(lastNode);

        MKLDNNEdgePtr edgePtr(new MKLDNNEdge(peerNode, mergedConv));
        graph.GetEdges().push_back(edgePtr);

        size_t childIdx = 0;
        for (childIdx = 0; childIdx < peerNode->getChildEdges().size(); childIdx++) {
            if (peerNode->getChildEdgeAt(childIdx)->getChild() == sum) {
                break;
            }
        }

        mergedConv->addEdge(edgePtr, mergedConv->getParentEdges().size(), childIdx);

        for (size_t j = 0; j < lastNode->getChildEdges().size(); j++) {
            auto child = lastNode->getChildEdgeAt(j)->getChild();
            edgePtr = lastNode->getChildEdgeAt(j);
            int idxParent = edgePtr->getOutputNum();
            int idxChild = edgePtr->getInputNum();

            MKLDNNEdgePtr newEdge(new MKLDNNEdge(mergedConv, child));
            graph.GetEdges().push_back(newEdge);
            child->addEdge(newEdge, idxParent, idxChild);
        }

        if (lastNode != sum) {
            lastNode->remove();
        }
        sum->remove();
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

        if (toDrop) DropNode(graph, node);
    }
}

void MKLDNNGraphOptimizer::RemoveDropped(MKLDNNGraph& graph) {
    auto& nodes = graph.GetNodes();

    auto it = nodes.begin();

    while (it != nodes.end()) {
        if ((*it)->isDropped()) {
            it = nodes.erase(it);
        } else {
            it++;
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

void MKLDNNGraphOptimizer::DropNode(MKLDNNGraph &graph, MKLDNNNodePtr &node) {
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
                removeEdge(graph, remEdge);
            }
            inNum += j;
            remEdge = node->childEdges[j].lock();
            int outNum = 0;
            if (remEdge) {
                outNum = remEdge->getOutputNum();
                node->removeEdge(remEdge);
                removeEdge(graph, remEdge);
            }
            MKLDNNEdgePtr newEdge(new MKLDNNEdge(parent, child));
            graph.GetEdges().push_back(newEdge);
            parent->addEdge(newEdge, outNum, inNum);
        }
    }
}

void MKLDNNGraphOptimizer::RemoveDroppedEdges(MKLDNNGraph &graph) {
    auto& edges = graph.GetEdges();

    auto it = edges.begin();

    while (it != edges.end()) {
        if ((*it)->isDropped()) {
            it = edges.erase(it);
        } else {
            it++;
        }
    }
}
