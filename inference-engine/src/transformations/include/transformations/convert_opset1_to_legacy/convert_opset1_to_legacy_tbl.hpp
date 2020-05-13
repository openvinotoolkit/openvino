// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NGRAPH_PASS
#warning "NGRAPH_PASS is not defined"
#define NGRAPH_PASS(A, B)
#endif

// To register new pass you need to define NGRAPH_PASS
// Usage example:
//   ngraph::pass:Manager pm;
//   #define NGRAPH_PASS(NAME, NAMESPACE)   pm.register_pass<NAMESPACE::NAME>();
//   #include <transformations/transformations_tbl.hpp>
//   #undef NGRAPH_PASS

NGRAPH_PASS(ConstantFolding, ::ngraph::pass)
NGRAPH_PASS(ConvertReduceToPooling, ::ngraph::pass)
NGRAPH_PASS(ConvertMod, ::ngraph::pass)
NGRAPH_PASS(ConvertMinimum, ::ngraph::pass)
NGRAPH_PASS(ConvertSubtract, ::ngraph::pass)
NGRAPH_PASS(ConvertDivide, ::ngraph::pass)
NGRAPH_PASS(ConvertNegative, ::ngraph::pass)
NGRAPH_PASS(ConvertDepthToSpace, ::ngraph::pass)
NGRAPH_PASS(ConvertSpaceToDepth, ::ngraph::pass)
NGRAPH_PASS(ConvertConvolutions, ::ngraph::pass)
NGRAPH_PASS(BatchNormDecomposition, ::ngraph::pass)
NGRAPH_PASS(ConstantFolding, ::ngraph::pass)
NGRAPH_PASS(MulAddVerification, ::ngraph::pass)
NGRAPH_PASS(MulAddFusion, ::ngraph::pass)
NGRAPH_PASS(ConstantFolding, ::ngraph::pass)
NGRAPH_PASS(ConvertMatMulToFCorGemm, ::ngraph::pass)
NGRAPH_PASS(PullTransposeThroughFQUp, ::ngraph::pass)
NGRAPH_PASS(ConstantFolding, ::ngraph::pass)
NGRAPH_PASS(ConvFusion, ::ngraph::pass)
NGRAPH_PASS(FullyConnectedBiasFusion, ::ngraph::pass)
NGRAPH_PASS(ConstantFolding, ::ngraph::pass)
NGRAPH_PASS(ReshapeFullyConnected, ::ngraph::pass)
NGRAPH_PASS(ReshapeFullyConnectedFusion, ::ngraph::pass)
NGRAPH_PASS(Reshape1DConvolutions, ::ngraph::pass)
NGRAPH_PASS(ConvertNormalizeL2WithMulToNormalizeIE, ::ngraph::pass)
NGRAPH_PASS(ConvertNormalizeL2ToNormalizeIE, ::ngraph::pass)
NGRAPH_PASS(ConstantEltwiseReduction, ::ngraph::pass)
NGRAPH_PASS(ConvertMulAddToScaleShiftOrPower, ::ngraph::pass)
NGRAPH_PASS(ConvertMulOrAddFinally, ::ngraph::pass)
NGRAPH_PASS(ConstantFolding, ::ngraph::pass)
NGRAPH_PASS(ConvertBroadcastToTiles, ::ngraph::pass)
NGRAPH_PASS(ConvertTileToIETile, ::ngraph::pass)
NGRAPH_PASS(ConvertProposalToProposalIE, ::ngraph::pass)
NGRAPH_PASS(ConvertLRNToLRNIE, ::ngraph::pass)
NGRAPH_PASS(ConvertPadToPadIE, ::ngraph::pass)
NGRAPH_PASS(ConvertHardSigmoidToHardSigmoidIE, ::ngraph::pass)
NGRAPH_PASS(ConvertLSTMCellToLSTMCellIE, ::ngraph::pass)
NGRAPH_PASS(ConvertInterpolateToInterpOrResample, ::ngraph::pass)
NGRAPH_PASS(ConvertStridedSliceToCrop, ::ngraph::pass)
NGRAPH_PASS(ConvertStridedSliceToStridedSliceIE, ::ngraph::pass)
NGRAPH_PASS(ConvertPowerToPowerIE, ::ngraph::pass)
NGRAPH_PASS(ConvertSqrtToPowerIE, ::ngraph::pass)
NGRAPH_PASS(ConvertPReLUToReLUIE, ::ngraph::pass)
NGRAPH_PASS(ConvertGatherToGatherIE, ::ngraph::pass)
NGRAPH_PASS(ConvertSeluToSeluIE, ::ngraph::pass)
NGRAPH_PASS(ConvertOneHotToOneHotIE, ::ngraph::pass)
NGRAPH_PASS(ConvertGatherTreeToGatherTreeIE, ::ngraph::pass)
NGRAPH_PASS(ConvertTopKToTopKIE, ::ngraph::pass)
NGRAPH_PASS(ConvertNMSToNMSIE, ::ngraph::pass)
NGRAPH_PASS(ConstantFolding, ::ngraph::pass)
