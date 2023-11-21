// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph.h"
#include "nodes/eltwise.h"
#include <vector>

namespace ov {
namespace intel_cpu {

class GraphOptimizer {
public:
    GraphOptimizer();

public:
    void ApplyCommonGraphOptimizations(Graph& graph);
    void ApplyImplSpecificGraphOptimizations(Graph& graph);

private:
    void FuseConvMatmulFCDeconvAndDQScales(Graph &graph);
    void FuseFCAndWeightsDecompression(Graph &graph);
    void FuseConvolutionMatMulDeconvAndBias(Graph &graph);
    void FuseDeconvolutionAndSimpleOperation(Graph &graph);
    void FuseMultiplyAndAdd(Graph &graph);
    void MergeConvertAndScaleShift(Graph& graph);
    void FuseFCAndConvertOnWeights(Graph& graph);
    void FuseFCAndTransposeOnWeights(Graph& graph);
    void FuseFullyConnectedAndSimpleOperation(Graph &graph);
    void FuseMatMulAndSimpleOperation(Graph &graph);
    void FuseConvolutionAndSimpleOperationThroughMaxPool(Graph &graph);
    void FuseConvolutionAndSimpleOperation(Graph &graph);
    void FuseConvolutionAndDWConvolution(Graph &graph);
    void FusePoolingAndFakeQuantize(Graph &graph);
    void FuseConvolutionSumAndConvolutionSumActivation(Graph &graph);
    void FuseMVNAndSimpleOperation(Graph &graph);
    void FuseInterpolateAndSimpleOperation(Graph &graph);
    void FuseNormalizeL2AndSimpleOperation(Graph &graph);
    void FuseReduceAndSimpleOperation(Graph &graph);

    void DropDoubleReorders(Graph& graph);
    void FuseConvolutionAndZeroPoints(Graph &graph);
    void FuseBroadcastAndEltwise(Graph &graph);
    void FuseEltwiseAndSimple(Graph &graph);
    void FusePerformedAsScaleShiftAndFakeQuantize(Graph &graph);
    void FuseClampAndFakeQuantize(Graph &graph);
    void MergeTransposeAndReorder(Graph &graph);
    void reshapeRnnSeq(Graph &graph);
    void RemoveSameConvert(Graph &graph);
    void MatchSdpaKvCache(Graph &graph);
};

}   // namespace intel_cpu
}   // namespace ov
