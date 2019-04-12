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


#include <vector>
#include <gtest/gtest.h>
#include "gna_matcher.hpp"

class PWLAproximationTest : public GNATest {
 protected:
    void SetUp() override  {
    }
};
using namespace GNATestIRs;

// Recursive Algorithm
// Precision Threshold

TEST_F(PWLAproximationTest, forTanhOnRecursiveAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferModel(TanhActivationModel())
	                            .inNotCompactMode()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActTanh)
                                .pwl_quantization_precision_threshold(0.0053);
}

TEST_F(PWLAproximationTest, forSigmoidOnRecursiveAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferModel(SigmoidActivationModel())
                                .inNotCompactMode()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActSigmoid)
                                .pwl_quantization_precision_threshold(0.0027);
}

TEST_F(PWLAproximationTest, forReLUonRecursiveAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferModel(ReLUActivationModel())
                                .inNotCompactMode()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActRelu)
                                .pwl_quantization_precision_threshold(0.0001);
}

TEST_F(PWLAproximationTest, forLeakyReLUonRecursiveAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferModel(LeakyReLUActivationModel())
                                .inNotCompactMode()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActLeakyRelu)
                                .pwl_quantization_precision_threshold(0.0003);
}

TEST_F(PWLAproximationTest, DISABLED_forIdentityOnRecursiveAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferModel(IdentityActivationModel())
                                .inNotCompactMode()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActIdentity)
                                .pwl_quantization_precision_threshold(0.0003);
}

TEST_F(PWLAproximationTest, forClampOnRecursiveAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferModel(ClampActivationModel())
                                .inNotCompactMode()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActKaldiLstmClipping)
                                .pwl_quantization_precision_threshold(0.0001);
}

// Uniform Algorithm
// Precision Threshold

TEST_F(PWLAproximationTest, forTanhOnUniformAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferModel(TanhActivationModel())
                                .inNotCompactMode()
                                .withUniformPWLAlgo()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActTanh)
                                .pwl_quantization_precision_threshold(0.0009);
}

TEST_F(PWLAproximationTest, forSigmoidOnUniformAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferModel(SigmoidActivationModel())
                                .inNotCompactMode()
                                .withUniformPWLAlgo()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActSigmoid)
                                .pwl_quantization_precision_threshold(0.0004);
}

TEST_F(PWLAproximationTest, DISABLED_forIdentityOnUniformAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferModel(IdentityActivationModel())
                                .inNotCompactMode()
                                .withUniformPWLAlgo()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActIdentity)
                                .pwl_quantization_precision_threshold(0.0003);
}

TEST_F(PWLAproximationTest, forClampOnUniformAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferModel(ClampActivationModel())
                                .inNotCompactMode()
                                .withUniformPWLAlgo()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActKaldiLstmClipping)
                                .pwl_quantization_precision_threshold(0.0001);
}

// Recursive Algorithm
// Segment Threshold

TEST_F(PWLAproximationTest, forSigmoidonRecursiveAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferModel(SigmoidActivationModel())
                                .inNotCompactMode()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActSigmoid)
                                .pwl_quantization_segments_threshold(12);
}

TEST_F(PWLAproximationTest, forTanhonRecursiveAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferModel(TanhActivationModel())
                                .inNotCompactMode()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActTanh)
                                .pwl_quantization_segments_threshold(12);
}

TEST_F(PWLAproximationTest, forReLUonRecursiveAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferModel(ReLUActivationModel())
                                .inNotCompactMode()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActRelu)
                                .pwl_quantization_segments_threshold(2);
}

TEST_F(PWLAproximationTest, forLeakyReLUonRecursiveAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferModel(LeakyReLUActivationModel())
                                .inNotCompactMode()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActLeakyRelu)
                                .pwl_quantization_segments_threshold(2);
}

TEST_F(PWLAproximationTest, DISABLED_forIdentityOnRecursiveAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferModel(IdentityActivationModel())
                                .inNotCompactMode()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActIdentity)
                                .pwl_quantization_segments_threshold(3);
}

TEST_F(PWLAproximationTest, forClampOnRecursiveAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferModel(ClampActivationModel())
                                .inNotCompactMode()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActKaldiLstmClipping)
                                .pwl_quantization_segments_threshold(3);
}

// Uniform Algorithm
// Segment Threshold

TEST_F(PWLAproximationTest, forSigmoidonUniformAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferModel(SigmoidActivationModel())
                                .inNotCompactMode()
                                .withUniformPWLAlgo()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActSigmoid)
                                .pwl_quantization_segments_threshold(65);
}

TEST_F(PWLAproximationTest, forTanhonUniformAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferModel(TanhActivationModel())
                                .inNotCompactMode()
                                .withUniformPWLAlgo()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActTanh)
                                .pwl_quantization_segments_threshold(65);
}

TEST_F(PWLAproximationTest, DISABLED_forIdentityOnUniformAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferModel(IdentityActivationModel())
                                .inNotCompactMode()
                                .withUniformPWLAlgo()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActIdentity)
                                .pwl_quantization_segments_threshold(3);
}

TEST_F(PWLAproximationTest, forClampOnUniformAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferModel(ClampActivationModel())
                                .inNotCompactMode()
                                .withUniformPWLAlgo()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActKaldiLstmClipping)
                                .pwl_quantization_segments_threshold(3);
}
