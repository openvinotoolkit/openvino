// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace GNATestIRs {

std::string FCOnlyModel() ;
std::string Fc2DOutputModel();
std::string affineToMemoryModel();
std::string MemoryAfterConcatModel();
std::string MemoryAndConcatAfterOneNode();
std::string eltwiseToMemoryModel();
std::string eltwiseToMemoryModelNoOutput();
std::string FCWithPaddingAfterSplitModel();
/// @eltwise layer mode
std::string SliceModelWithAlignedOutputs();
std::string FCWithPaddingAfterSliceModel();
std::string twoFCWithPaddingAfterSliceModel();
std::string eltwiseSummModel();
std::string eltwiseMulModel();
std::string doubleConcatModel();
std::string scaleShiftAffineModel();
std::string AffineWith2AffineOutputsModel();
std::string maxpoolAfterRelu();
std::string TFLeakyReluModel();
std::string cropWithoutOffsetModel();
std::string cropWithAlignedOffsetModel();
std::string cropWithOffsetModel();
std::string cropWithOffsetAndSecondDimModel();
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
std::string concatModelWithConstLayer();
std::string LSTMCellOnlyModel();
std::string eltwiseSumModelWithConstLayer();
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
std::string TwoOutputs();
std::string TwoOutputsDiffPrecision();
std::string SplitToConcatWith2InputsAlignedNoFC();
std::string SplitToConcatWith2Inputs1360NotAlignedNoFC();
std::string SplitToConcatWith3InputsAlignedNoFC();
std::string SplitToConcatWith10InputsAlignedNoFC();
std::string SplitToConcatWith2By64InputsAlignedNoFC();
std::string ReshapeConvolutionLessThan48Filters();

}  // namespace GNATestIRs
