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
std::string copyModel();
std::string two_inputs_to_affine();
std::string two_inputs_to_concat();
std::string affineAfterConvNoPermute();
std::string affineAfterConvWithPermute();
std::string ScaleShift3DModel();
}  // namespace GNATestIRs
