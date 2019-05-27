// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include <inference_engine/layer_transform.hpp>
#include "gna_plugin/quantization/model_quantizer.hpp"
#include "gna_plugin/quantization/layer_quantizer.hpp"
#include "gna_matcher.hpp"

using namespace InferenceEngine;
using namespace GNAPluginNS;
using namespace GNATestIRs;

class FP32NonQuantizedTest : public GNATest {
 protected:

    void SetUp() override  {
    }
};

TEST_F(FP32NonQuantizedTest, SplitFollowedByFCAndEltwiseOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {12.0, 12.0, 12.0, 12.0, 12.0,
                                          12.0, 12.0, 12.0, 12.0, 12.0};
    assert_that().onInferModel(FCWithPaddingAfterSplitModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(FP32NonQuantizedTest, SliceFollowedByFCAndEltwiseOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0};
    assert_that().onInferModel(FCWithPaddingAfterSliceModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(FP32NonQuantizedTest, SliceFollowedByAlignedFCAndEltwiseOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {18.0, 18.0, 18.0, 18.0};
    assert_that().onInferModel(SliceModelWithAlignedOutputs())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(FP32NonQuantizedTest, DISABLED_SliceFollowedBy2FCsAnd2EltwisesOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0};
    assert_that().onInferModel(twoFCWithPaddingAfterSliceModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(FP32NonQuantizedTest, SplitAfterFCFollowedByFCAndEltwiseOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {232.0, 232.0, 232.0, 232.0, 232.0,
                                          232.0, 232.0, 232.0, 232.0, 232.0};
    assert_that().onInferModel(FCBeforeSplitModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}


TEST_F(FP32NonQuantizedTest, ConcatPropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {121.0, 121.0, 121.0, 121.0, 121.0,
                                          121.0, 121.0, 121.0, 121.0, 121.0,
                                          121.0, 121.0, 121.0, 121.0, 121.0,
                                          121.0, 121.0, 121.0, 121.0, 121.0};

    assert_that().onInferModel(concatModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(FP32NonQuantizedTest, DoubleConcatPropageteForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0,
                                          141.0, 141.0, 141.0, 141.0, 141.0};

    assert_that().onInferModel(doubleConcatModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(FP32NonQuantizedTest, multiple_inputs_correct_results) {
    std::vector<float> input_data  = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> input2_data = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    std::vector<float> result      = {30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0};

    assert_that().onInferModel(two_inputs_to_affine())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with().input("input_1", input_data).And().input("input_2", input2_data).result().equal_to(result);
}


TEST_F(FP32NonQuantizedTest, CropWithoutOffsetPropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> expected_result = {11.0, 11.0, 11.0, 11.0, 11.0,
                                          11.0, 11.0, 11.0, 11.0, 11.0};

    assert_that().onInferModel(cropWithoutOffsetModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(FP32NonQuantizedTest, CropWithAlignedOffsetPropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> expected_result = {3.0, 3.0, 3.0, 3.0, 3.0,
                                          3.0, 3.0, 3.0, 3.0, 3.0};

    assert_that().onInferModel(cropWithAlignedOffsetModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(FP32NonQuantizedTest, CropWithOffsetPropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> expected_result = {7.0, 7.0, 7.0, 7.0, 7.0,
                                          7.0, 7.0, 7.0, 7.0, 7.0};

    assert_that().onInferModel(cropWithOffsetModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(FP32NonQuantizedTest, CropWithMaxOffsetPropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> expected_result = {1.0, 1.0, 1.0, 1.0, 1.0,
                                          1.0, 1.0, 1.0, 1.0, 1.0};

    assert_that().onInferModel(cropWithMaxOffsetModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(FP32NonQuantizedTest, CropWithOffsetAfterFCPropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> expected_result = {111.0, 111.0, 111.0, 111.0, 111.0,
                                          111.0, 111.0, 111.0, 111.0, 111.0};

    assert_that().onInferModel(cropWithOffsetExtendedModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(FP32NonQuantizedTest, CopySimpleCasePropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> expected_result = {12.0, 12.0, 12.0, 12.0, 12.0,
                                          12.0, 12.0, 12.0, 12.0, 12.0,
                                          11.0, 11.0, 11.0, 11.0, 11.0,
                                          11.0, 11.0, 11.0, 11.0, 11.0,};

    assert_that().onInferModel(copyModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}


TEST_F(FP32NonQuantizedTest, ScaleShiftWithBroadcastSupported) {
    std::vector<float> input_data (40, 1.0);

    std::vector<float> expected_result = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                                          2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                                          2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                                          2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                                          2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0};

    assert_that().onInferModel(ScaleShift3DModel()).withWeigthsPattern({1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f})
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}