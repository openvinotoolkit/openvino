// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_types.h"
#include "graph.h"
#include "node.h"

namespace ov::intel_cpu {

class GraphOptimizer {
public:
    GraphOptimizer();

    static void ApplyCommonGraphOptimizations(Graph& graph);
    static void ApplyImplSpecificGraphOptimizations(Graph& graph);
    static void ShareReorders(Graph& graph);

private:
    void extracted();
    static void FuseConvMatmulFCDeconvAndDQScales(Graph& graph);
    static void FuseConvolutionMatMulDeconvAndBias(Graph& graph);
    static void FuseDeconvolutionAndSimpleOperation(Graph& graph);
    static void FuseMultiplyAndAdd(Graph& graph);
    static void MergeEltwiseAndConvert(Graph& graph);
    static void MergeConvertAndEltwise(Graph& graph);
    static void FuseFCAndConvertOnWeights(Graph& graph);
    static void FuseFCAndTransposeOnWeights(Graph& graph);
    static void FuseFullyConnectedAndSimpleOperation(Graph& graph);
    static void FuseMatMulAndSimpleOperation(Graph& graph);
    static void FuseConvolutionAndSimpleOperationThroughMaxPool(Graph& graph);
    static void FuseConvolutionAndSimpleOperation(Graph& graph);
    static void FuseConvolutionAndDWConvolution(Graph& graph);
    static void FusePoolingAndFakeQuantize(Graph& graph);
    static void FuseConvolutionSumAndConvolutionSumActivation(Graph& graph);
    static void FuseMVNAndSimpleOperation(Graph& graph);
    static void FuseInterpolateAndSimpleOperation(Graph& graph);
    static void FuseNormalizeL2AndSimpleOperation(Graph& graph);
    static void FuseReduceAndSimpleOperation(Graph& graph);
    static void FuseGatherAndConvert(Graph& graph);

    static void DropDoubleReorders(Graph& graph);
    static void FuseConvolutionAndZeroPoints(Graph& graph);
    void FuseBroadcastAndEltwise(Graph& graph);
    static void FuseEltwiseAndSimple(Graph& graph);
    static void FusePerformedAsScaleShiftAndFakeQuantize(Graph& graph);
    static void FuseClampAndFakeQuantize(Graph& graph);
    static void MergeTransposeAndReorder(Graph& graph);
    static void MergeReorderAndTranspose(Graph& graph);
    static void reshapeRnnSeq(Graph& graph);
    static void RemoveSameConvert(Graph& graph);
    static void RemoveMemoryInputConvert(Graph& graph);
    static void RemoveConvertMemoryOutput(Graph& graph);
    static void MatchSdpaKvCache(Graph& graph);
    static void DropRedundantMemoryOutput(Graph& graph);

    static bool canBeInplaced(const NodePtr& parentNode, const NodePtr& childNode);
    // Method checks that after the sequential execution of Transpose and Reorder nodes,
    // the order of the elements in the memory (physical layout) will not change.
    static bool checkAscendingFinalOrder(const VectorDims& transposeOrder,
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
    static void mergeTransposeReshapeReorder(Graph& graph,
                                             const NodePtr& transposeNode,
                                             const NodePtr& reshapeNode,
                                             const NodePtr& reorderNode,
                                             bool reverseOrder);
};

}  // namespace ov::intel_cpu
