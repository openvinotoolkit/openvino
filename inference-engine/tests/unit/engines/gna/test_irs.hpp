/*
 * INTEL CONFIDENTIAL
 * Copyright (C) 2018-2019 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */

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
