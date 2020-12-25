// Copyright (C) 2018-2020 Intel Corporation
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
    void ApplyImplSpecificGraphOptimizations(MKLDNNGraph& graph);

private:
    void MergeConversions(MKLDNNGraph& graph);
    void MergeGroupConvolution(MKLDNNGraph& graph);
    void MergeTwoEqualScaleShifts(MKLDNNGraph& graph);
    void FuseConvolutionAndActivation(MKLDNNGraph &graph);
    void FuseFullyConnectedAndSimpleOperation(MKLDNNGraph &graph);
    void FuseConvolutionAndDepthwise(MKLDNNGraph &graph);
    void FuseConvolutionAndSimpleOperation(MKLDNNGraph &graph);
    void FuseConvolutionAndDWConvolution(MKLDNNGraph &graph);
#if defined(COMPILED_CPU_MKLDNN_QUANTIZE_NODE)
    void FuseConvolutionAndQuantize(MKLDNNGraph &graph);
    void FuseBinaryConvolutionAndQuantize(MKLDNNGraph &graph);
    void FusePoolingAndQuantize(MKLDNNGraph &graph);
#endif
    void FuseBatchNormWithScale(MKLDNNGraph& graph);
#if defined(COMPILED_CPU_MKLDNN_ELTWISE_NODE)
    void FuseConvolutionSumAndConvolutionSumActivation(MKLDNNGraph &graph);
#endif
    void FuseMVNAndSimpleOperation(MKLDNNGraph &graph);
    void FuseInterpolateAndSimpleOperation(MKLDNNGraph &graph);
    void FuseNormalizeAndSimpleOperation(MKLDNNGraph &graph);
    void RemoveIdentityOperator(MKLDNNGraph& graph);

    void RemoveIOScaleShifts(MKLDNNGraph& graph);
#if defined (COMPILED_CPU_MKLDNN_REORDER_NODE)
    void DropDoubleReorders(MKLDNNGraph& graph);
    void DropConvertReorder(MKLDNNGraph& graph);
    void ChangeConvertToReorder(MKLDNNGraph &graph);
#endif
    void FuseConvolutionAndZeroPoints(MKLDNNGraph &graph);
    void FuseBroadcastAndEltwise(MKLDNNGraph &graph);
    void FuseEltwiseAndSimple(MKLDNNGraph &graph);
    void FuseScaleShiftAndQuantize(MKLDNNGraph &graph);
    void FuseClampAndQuantize(MKLDNNGraph &graph);
    void MergePermuteAndReorder(MKLDNNGraph &graph);

    bool IsOneOf(Type type, std::vector<Type> types);
    bool IsOneOf(EltwiseOpType alg, std::vector<EltwiseOpType> algs);

    void removeEdge(MKLDNNGraph &graph, MKLDNNEdgePtr& edge);
};

}  // namespace MKLDNNPlugin
