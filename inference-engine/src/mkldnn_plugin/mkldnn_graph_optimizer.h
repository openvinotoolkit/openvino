// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn_graph.h"
#include "nodes/mkldnn_eltwise_node.h"
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNGraphOptimizer {
public:
    MKLDNNGraphOptimizer();

public:
    void ApplyCommonGraphOptimizations(MKLDNNGraph& graph);
//    void ApplyImplSpecificGraphOptimizations(MKLDNNGraph& graph);

private:
//    void FuseConvolutionAndBias(MKLDNNGraph &graph);
//    void FuseDeconvolutionAndSimpleOperation(MKLDNNGraph &graph);
//    void FuseMultiplyAndAdd(MKLDNNGraph &graph);
//    void FuseFullyConnectedAndSimpleOperation(MKLDNNGraph &graph);
//    void FuseConvolutionAndSimpleOperationThroughMaxPool(MKLDNNGraph &graph);
//    void FuseConvolutionAndSimpleOperation(MKLDNNGraph &graph);
//    void FuseConvolutionAndDWConvolution(MKLDNNGraph &graph);
//    void FusePoolingAndFakeQuantize(MKLDNNGraph &graph);
//    void FuseConvolutionSumAndConvolutionSumActivation(MKLDNNGraph &graph);
//    void FuseMVNAndSimpleOperation(MKLDNNGraph &graph);
//    void FuseInterpolateAndSimpleOperation(MKLDNNGraph &graph);
//    void FuseNormalizeL2AndSimpleOperation(MKLDNNGraph &graph);
//
//    void DropDoubleReorders(MKLDNNGraph& graph);
//    void FuseConvolutionAndZeroPoints(MKLDNNGraph &graph);
//    void FuseBroadcastAndEltwise(MKLDNNGraph &graph);
//    void FuseEltwiseAndSimple(MKLDNNGraph &graph);
//    void FusePerformedAsScaleShiftAndFakeQuantize(MKLDNNGraph &graph);
//    void FuseClampAndFakeQuantize(MKLDNNGraph &graph);
//    void MergeTransposeAndReorder(MKLDNNGraph &graph);
//
//    void removeEdge(MKLDNNGraph &graph, MKLDNNEdgePtr& edge);
};

}  // namespace MKLDNNPlugin
