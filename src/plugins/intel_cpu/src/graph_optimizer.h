// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph.h"

namespace ov {
namespace intel_cpu {

class GraphOptimizer {
public:
    GraphOptimizer();

public:
    void ApplyCommonGraphOptimizations(Graph& graph);
    void ApplyImplSpecificGraphOptimizations(Graph& graph);
    void ShareReorders(Graph &graph);

private:
    void FuseConvMatmulFCDeconvAndDQScales(Graph &graph);
    void FuseFCAndWeightsDecompression(Graph &graph);
    void FuseGatherAndWeightsDecompression(Graph &graph);
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
    void MergeReorderAndTranspose(Graph &graph);
    void reshapeRnnSeq(Graph &graph);
    void RemoveSameConvert(Graph &graph);
    void RemoveMemoryInputConvert(Graph &graph);
    void RemoveConvertMemoryOutput(Graph &graph);
    void MatchSdpaKvCache(Graph &graph);

    // Method checks that after the sequential execution of Transpose and Reorder nodes,
    // the order of the elements in the memory (physical layout) will not change.
    bool checkAscendingFinalOrder(const VectorDims& transposeOrder,
                                  const VectorDims& layoutOrder,
                                  const VectorDims& reorderInOrder,
                                  const VectorDims& reorderOutOrder);
    // Method merges Transpose -> Reshape(optional) -> Reorder sequences which do opposite permutation to each other.
    // Reverse order Reorder -> Reshape(optional) -> Transpose is supported too.
    // Reshape support has the following limitations:
    // - direct order: Only reshape which separates one of the dimension on 2 consecutive ones is supported
    // - reverse order: Only reshape which fuses 2 consecutive dimensions into one is supported
    // Example:
    //      chain [physical layout: NCHW, logical layout: NCHW] -> Transpose(order=0312) -> [physical layout: NWCH, logical layout: NCHW] ->
    //      Reorder(nchw->nhwc) -> [physical layout: NCHW, logical layout: NHWC] can be replaced with Reorder(nchw->nhwc; isOptimized=true)
    //      which will just reinterprets layout without physical change of the memory.
    void mergeTransposeReshapeReorder(Graph& graph,
                                      const NodePtr& transposeNode,
                                      const NodePtr& reshapeNode,
                                      const NodePtr& reorderNode,
                                      const bool reverseOrder);
};

}   // namespace intel_cpu
}   // namespace ov
