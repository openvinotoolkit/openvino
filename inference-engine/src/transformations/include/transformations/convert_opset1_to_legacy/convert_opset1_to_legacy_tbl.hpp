// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NGRAPH_PASS
#warning "NGRAPH_PASS is not defined"
#define NGRAPH_PASS(A, B)
#endif

#ifndef REGISTER_GRAPH_REWRITE_PASS
#warning "REGISTER_GRAPH_REWRITE_PASS is not defined"
#define REGISTER_GRAPH_REWRITE_PASS(A)
#endif

#ifndef REGISTER_MATCHER
#warning "REGISTER_MATCHER is not defined"
#define REGISTER_MATCHER(A, B)
#endif

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
NGRAPH_PASS(Reshape1DOps, ::ngraph::pass)
NGRAPH_PASS(ConvertNormalizeL2WithMulToNormalizeIE, ::ngraph::pass)
NGRAPH_PASS(ConstantEltwiseReduction, ::ngraph::pass)
NGRAPH_PASS(ConvertMulAddToScaleShiftOrPower, ::ngraph::pass)
NGRAPH_PASS(ConvertMulOrAddFinally, ::ngraph::pass)
NGRAPH_PASS(ConstantFolding, ::ngraph::pass)
NGRAPH_PASS(ConvertBroadcastToTiles, ::ngraph::pass)

REGISTER_GRAPH_REWRITE_PASS(ConvertOpSet1ToLegacy)
REGISTER_MATCHER(ConvertNormalizeL2ToNormalizeIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertTileToIETile, ::ngraph::pass)
REGISTER_MATCHER(ConvertProposalToProposalIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertLRNToLRNIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertPadToPadIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertHardSigmoidToHardSigmoidIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertCellsToCellsIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertInterpolateToInterpOrResample, ::ngraph::pass)
REGISTER_MATCHER(ConvertStridedSliceToCrop, ::ngraph::pass)
REGISTER_MATCHER(ConvertPowerToPowerIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertSqrtToPowerIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertPReLUToReLUIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertGatherToGatherIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertSeluToSeluIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertOneHotToOneHotIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertGatherTreeToGatherTreeIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertTopKToTopKIE, ::ngraph::pass)
REGISTER_MATCHER(ConvertNMSToNMSIE, ::ngraph::pass)

NGRAPH_PASS(ConstantFolding, ::ngraph::pass)
