// Copyright (C) 2018-2025 Intel Corporation
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
    void ShareReorders(Graph& graph);

private:
    void FuseConvMatmulFCDeconvAndDQScales(Graph& graph);
    void FuseConvolutionMatMulDeconvAndBias(Graph& graph);
    void FuseDeconvolutionAndSimpleOperation(Graph& graph);
    void FuseMultiplyAndAdd(Graph& graph);
    void MergeEltwiseAndConvert(Graph& graph);
    void MergeConvertAndEltwise(Graph& graph);
    void FuseFCAndConvertOnWeights(Graph& graph);
    void FuseFCAndTransposeOnWeights(Graph& graph);
    void FuseFullyConnectedAndSimpleOperation(Graph& graph);
    void FuseMatMulAndSimpleOperation(Graph& graph);
    void FuseConvolutionAndSimpleOperationThroughMaxPool(Graph& graph);
    void FuseConvolutionAndSimpleOperation(Graph& graph);
    void FuseConvolutionAndDWConvolution(Graph& graph);
    void FusePoolingAndFakeQuantize(Graph& graph);
    void FuseConvolutionSumAndConvolutionSumActivation(Graph& graph);
    void FuseMVNAndSimpleOperation(Graph& graph);
    void FuseInterpolateAndSimpleOperation(Graph& graph);
    void FuseNormalizeL2AndSimpleOperation(Graph& graph);
    void FuseReduceAndSimpleOperation(Graph& graph);

    void DropDoubleReorders(Graph& graph);
    void FuseConvolutionAndZeroPoints(Graph& graph);
    void FuseBroadcastAndEltwise(Graph& graph);
    void FuseEltwiseAndSimple(Graph& graph);
    void FusePerformedAsScaleShiftAndFakeQuantize(Graph& graph);
    void FuseClampAndFakeQuantize(Graph& graph);
    void MergeTransposeAndReorder(Graph& graph);
    void MergeReorderAndTranspose(Graph& graph);
    void reshapeRnnSeq(Graph& graph);
    void RemoveSameConvert(Graph& graph);
    void RemoveMemoryInputConvert(Graph& graph);
    void RemoveConvertMemoryOutput(Graph& graph);
    void MatchSdpaKvCache(Graph& graph);
    void DropRedundantMemoryOutput(Graph& graph);

    bool canBeInplaced(const NodePtr& parentNode, const NodePtr& childNode);
    // Method checks that after the sequential execution of Transpose and Reorder nodes,
    // the order of the elements in the memory (physical layout) will not change.
    bool checkAscendingFinalOrder(const VectorDims& transposeOrder,
                                  const VectorDims& layoutOrder,
                                  const VectorDims& reorderInOrder,
                                  const VectorDims& reorderOutOrder);
    // Method merges Transpose -> Reshape(optional) -> Reorder sequences which do opposite permutation to each other.
    // Reverse order Reorder -> Reshape(optional) -> Transpose is supported too.
    // Reshape support has the following limitations:
    // - direct order: Only reshape which split one of the dimension into 2 consecutive ones is supported
    // - reverse order: Only reshape which fuses 2 consecutive dimensions into one is supported
    // Examples:
    // 1. Direct order, no Reshape node.
    //    Before: [N,C,H,W]abcd==>Transpose(0312)==>[N,W,C,H]abcd==>Reorder(abcd->acdb)==>[N,W,C,H]acdb
    //    [N,C,H,W]abcd is equivalent to the [N,W,C,H]acdb, so the Transpose and Reorder can be fused into single
    //    optimized Reorder: After:  [N,C,H,W]abcd==>Reorder(abcd->acdb, isOptimized=true)==>[N,W,C,H]acdb
    // 2. Reverse order, no Reshape node.
    //    Before: [N,W,C,H]acdb==>Reorder(acdb->abcd)==>[N,W,C,H]abcd==>Transpose(0231)==>[N,C,H,W]abcd
    //    [N,W,C,H]acdb is equivalent to the [N,C,H,W]abcd, so the Transpose and Reorder can be fused into single
    //    optimized Reorder: After:  [N,W,C,H]acdb==>Reorder(acdb->abcd, isOptimized=true)==>[N,C,H,W]abcd
    // 3. Direct order with Reshape node (L = H x w).
    //    Before:
    //    [N,L,C]abc==>Transpose(021)==>[N,C,L]abc==>Reshape==>[N,C,H,W]abcd==>Reoder(abcd->acdb)==>[N,C,H,W]acdb After:
    //    [N,L,C]abc==>Reorder(abc->acdb, isOptimized=true)==>[N,C,H,W]acdb
    // 4. Reverse order with Reshape node (L = H x W).
    //    Before:
    //    [N,C,H,W]acdb==>Reorder(acdb->abcd)==>[N,C,H,W]abcd==>Reshape==>[N,C,L]abc==>Transpose(021)==>[N,L,C]abc
    //    After:  [N,C,H,W]acdb==>Reorder(acdb->abc, isOptimized=true)==>[N,L,C]abc
    // Note: in some cases (inplace conflicts or transpose with blocked input and non-blocked output) the merged Reorder
    // can not be optimized.
    void mergeTransposeReshapeReorder(Graph& graph,
                                      const NodePtr& transposeNode,
                                      const NodePtr& reshapeNode,
                                      const NodePtr& reorderNode,
                                      const bool reverseOrder);
};

}  // namespace intel_cpu
}  // namespace ov
