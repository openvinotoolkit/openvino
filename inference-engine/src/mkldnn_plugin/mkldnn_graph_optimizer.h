// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn_graph.h"
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNGraphOptimizer {
public:
    MKLDNNGraphOptimizer();

public:
    void Optimize(MKLDNNGraph& graph);

private:
    void MergeGroupConvolution(MKLDNNGraph& graph);
    void FuseConvolutionAndActivation(MKLDNNGraph &graph);
    void FuseConvolutionAndDWConvolution(MKLDNNGraph &graph);
    void FuseBatchNormWithScale(MKLDNNGraph& graph);
    void FuseConvolutionSumAndConvolutionSumActivation(MKLDNNGraph &graph);
    void RemoveIdentityOperator(MKLDNNGraph& graph);
    void RemoveDropped(MKLDNNGraph& graph);
    void RemoveDroppedEdges(MKLDNNGraph& graph);

    void DropNode(MKLDNNGraph& graph, MKLDNNNodePtr& node);

    bool IsOneOf(Type type, std::vector<Type> types);
};

}  // namespace MKLDNNPlugin
