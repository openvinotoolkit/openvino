// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph.h"
#include "nodes/eltwise.h"
#include <vector>

namespace ov {
namespace intel_cpu {

class MKLDNNGraphOptimizer {
public:
    MKLDNNGraphOptimizer();

public:
    void ApplyCommonGraphOptimizations(MKLDNNGraph& graph);
    void ApplyImplSpecificGraphOptimizations(MKLDNNGraph& graph);

private:
    void FuseConvolutionMatMulAndBias(MKLDNNGraph &graph);
    void FuseDeconvolutionAndSimpleOperation(MKLDNNGraph &graph);
    void FuseMultiplyAndAdd(MKLDNNGraph &graph);
    void FuseFullyConnectedAndSimpleOperation(MKLDNNGraph &graph);
    void FuseMatMulAndSimpleOperation(MKLDNNGraph &graph);
    void FuseConvolutionAndSimpleOperationThroughMaxPool(MKLDNNGraph &graph);
    void FuseConvolutionAndSimpleOperation(MKLDNNGraph &graph);
    void FuseConvolutionAndDWConvolution(MKLDNNGraph &graph);
    void FusePoolingAndFakeQuantize(MKLDNNGraph &graph);
    void FuseConvolutionSumAndConvolutionSumActivation(MKLDNNGraph &graph);
    void FuseMVNAndSimpleOperation(MKLDNNGraph &graph);
    void FuseInterpolateAndSimpleOperation(MKLDNNGraph &graph);
    void FuseNormalizeL2AndSimpleOperation(MKLDNNGraph &graph);
    void FuseReduceAndSimpleOperation(MKLDNNGraph &graph);

    void DropDoubleReorders(MKLDNNGraph& graph);
    void FuseConvolutionAndZeroPoints(MKLDNNGraph &graph);
    void FuseBroadcastAndEltwise(MKLDNNGraph &graph);
    void FuseEltwiseAndSimple(MKLDNNGraph &graph);
    void FusePerformedAsScaleShiftAndFakeQuantize(MKLDNNGraph &graph);
    void FuseClampAndFakeQuantize(MKLDNNGraph &graph);
    void MergeTransposeAndReorder(MKLDNNGraph &graph);
    void reshapeRnnSeq(MKLDNNGraph &graph);
};

}   // namespace intel_cpu
}   // namespace ov
