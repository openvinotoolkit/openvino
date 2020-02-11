// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace GNATestIRs {

std::string FCOnlyModel() ;
std::string Fc2DOutputModel();
std::string affineToMemoryModel();
std::string eltwiseToMemoryModel();
std::string eltwiseToMemoryModelNoOutput();
std::string activationAfterSplitModel();
std::string FCWithPaddingAfterSplitModel();
/// @eltwise layer mode
std::string EltwiseAfterSplitModel(int tensor_size, bool bMul = false);
std::string SliceModelWithAlignedOutputs();
std::string FCWithPaddingAfterSliceModel();
std::string FCBeforeSplitModel();
std::string twoFCWithPaddingAfterSliceModel();
std::string eltwiseSummModel();
std::string eltwiseMulModel();
std::string concatModel();
std::string doubleConcatModel();
std::string scaleShiftAffineModel();
std::string clampFollowedByTanhModel();
std::string eltwiseWithMemoryAndActivationInputModel();
std::string AffineWith2AffineOutputsModel();
std::string SigmoidActivationModel();
std::string TanhActivationModel();
std::string ReLUActivationModel();
std::string LeakyReLUActivationModel();
std::string ClampActivationModel();
std::string IdentityActivationModel();
std::string maxpoolAfterRelu();
std::string TFLeakyReluModel();
std::string cropWithoutOffsetModel();
std::string cropWithAlignedOffsetModel();
std::string cropWithOffsetModel();
std::string cropWithMaxOffsetModel();
std::string cropWithOffsetExtendedModel();
std::string twoCropsModel();
std::string threeCropsModel();
std::string threeCropsWithReshapeModel();
std::string copyModel();
std::string two_inputs_to_affine();
std::string two_inputs_to_concat();
std::string affineAfterConvNoPermute();
std::string affineAfterConvWithPermute();
std::string ScaleShift3DModel();
std::string FCOnlyModelFP16();
std::string AffineWithReluSigmoid();
std::string AffineWithReluSigmoidAndIdentity();
std::string concatModelWithConstLayer();
std::string LSTMCellOnlyModel();
std::string eltwiseSumModelWithConstLayer();
std::string eltwiseSumModelWithConstLayer2();
std::string eltwiseMulModelWithConstLayer();
std::string TIModelWithLSTMCell1();
std::string TIModelWithLSTMCell1Aligned();
std::string TIModelWithLSTMCell2();
std::string TIModelWithLSTMCell3Aligned();
std::string InputSplitConcatModel();
std::string InputSplitConcatModelUnaligned();
std::string InputSplitConcatReshapeModelUnaligned();
std::string LSTMCellOnlyModelUnaligned();
std::string TIModelWithLSTMCell1WithoutScaleShift();
std::string SplitToConcatThroughScaleShift();
std::string PowerWithScaleFactor1();
std::string ConcatWithDiffScaleFactor();
std::string TwoOutputs();
std::string TwoOutputsDiffPrecision();

std::string SplitToConcatWith2InputsNotAlignedNoFC();
std::string SplitToConcatWith2InputsAlignedNoFC();
std::string SplitToConcatWith2InputsAlignedNoFCWithInCopyWithOutCopy();
std::string SplitToConcatWith2Inputs1360NotAlignedNoFC();
std::string SplitToConcatWith2InputsNotAlignedWithFC();
std::string SplitToConcatWith2InputsAlignedWithFC();
std::string SplitToConcatWith2InputsAlignedWithFCWithInCopy();

std::string SplitToConcatWith3InputsNotAlignedNoFC();
std::string SplitToConcatWith3InputsAlignedNoFC();
std::string SplitToConcatWith3InputsAlignedNoFCWithInCopyWithOutCopy();
std::string SplitToConcatWith3InputsNotAlignedWithFC();
std::string SplitToConcatWith3InputsAlignedWithFC();
std::string SplitToConcatWith3InputsAlignedWithFCWithInCopy();

std::string SplitToConcatWith4InputsNotAlignedNoFC();
std::string SplitToConcatWith4InputsNotAlignedNoFCWithOutCopy();

std::string SplitToConcatWith10InputsNotAlignedNoFC();
std::string SplitToConcatWith10InputsNotAlignedNoFCWithOutCopy();
std::string SplitToConcatWith10InputsAlignedNoFC();
std::string SplitToConcatWith10InputsAlignedNoFCWithInCopyWithOutCopy();
std::string SplitToConcatWith10InputsNotAlignedWithFC();
std::string SplitToConcatWith10InputsAlignedWithFC();
std::string SplitToConcatWith10InputsAlignedWithFCWithInCopy();

std::string SplitToConcatWith10By1InputsNotAlignedNoFCWithOutCopy();
std::string SplitToConcatWith2By50InputsNotAlignedNoFC();
std::string SplitToConcatWith2By50InputsNotAlignedNoFCWithInCopyWithOutCopy();
std::string SplitToConcatWith2By64InputsAlignedNoFC();
std::string SplitToConcatWith2By64InputsAlignedNoFCWithOutCopy();
std::string SplitToConcatWith3By512InputsWithOutCopy();

std::string ReshapeConvolutionLessThan48Filters();
}  // namespace GNATestIRs
