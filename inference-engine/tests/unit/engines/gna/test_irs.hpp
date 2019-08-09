// Copyright (C) 2018-2019 Intel Corporation
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
std::string SplitToConcatThroughScaleShift();
std::string PowerWithScaleFactor1();
}  // namespace GNATestIRs
