// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <numeric>
#include <gtest/gtest.h>
#include <legacy/layer_transform.hpp>
#include "gna_matcher.hpp"

using namespace InferenceEngine;
using namespace GNAPluginNS;
using namespace GNATestIRs;

class FP32NonQuantizedTest : public GNATest<>{
 protected:

    void SetUp() override  {
    }
};

class FP32TestParams {
 public:
    int nChannels = 1;
    enum tagEltwise : uint8_t {
        eSumm,
        eMul
    } eltwise_type;
    FP32TestParams(int nChannels, uint8_t eltwise) : nChannels(nChannels), eltwise_type((tagEltwise)eltwise) {}
};

/**
 * parameter defines one of key dims size - number of channels in input tensor
 * due to gna-plugin implementation esential it is required to check 64 bits aligned tensors and non aligned
 */
class GNAFP32ParametricTest : public GNATest<::testing::TestWithParam<FP32TestParams>> {

};

static std::string getTestName(testing::TestParamInfo<FP32TestParams> obj) {
    return  "channels_" + std::to_string(obj.param.nChannels) + "_" + (obj.param.eltwise_type == FP32TestParams::eSumm ? "summ" : "mull");
}


TEST_P(GNAFP32ParametricTest, SplitFollowedByEltwiseMulOnAllignedCPU) {
    auto c = GetParam().nChannels;
    auto isMull = GetParam().eltwise_type == FP32TestParams::eMul;
    std::vector<float> input_data1(c, 3.0);
    std::vector<float> input_data2(c, 2.0);
    std::vector<float> input_data;
    input_data.insert(input_data.end(), input_data1.begin(), input_data1.end());
    input_data.insert(input_data.end(), input_data2.begin(), input_data2.end());

    std::vector<float> expected_result(c, isMull ? 6.0 : 5.0);
    assert_that().onInferModel(EltwiseAfterSplitModel(c, isMull))
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with().input("input_1", input_data).equals_to(expected_result);
}

FP32TestParams gna_fp32_test_params[] = {
    {7, FP32TestParams::eMul},
    {7, FP32TestParams::eSumm},
    {10, FP32TestParams::eMul},
    {10, FP32TestParams::eSumm},
    {32, FP32TestParams::eMul},
    {32, FP32TestParams::eSumm}
};

INSTANTIATE_TEST_SUITE_P(GNAFP32Tests, GNAFP32ParametricTest,
    ::testing::ValuesIn(gna_fp32_test_params), getTestName);

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

TEST_F(FP32NonQuantizedTest, SliceFollowedBy2FCsAnd2EltwisesOnCPU) {
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

TEST_F(FP32NonQuantizedTest, DoubleConcatPropagateForwardWithSuccessOnCPU) {
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
        .called_with().input("input_1", input_data).And().input("input_2", input2_data).result().equals_to(result);
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

TEST_F(FP32NonQuantizedTest, CropWithOffsetAndSecondDimPropagateForwardWithSuccessOnCPU) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> expected_result = {7.0, 7.0, 7.0, 7.0, 7.0,
                                          7.0, 7.0, 7.0, 7.0, 7.0};

    assert_that().onInferModel(cropWithOffsetAndSecondDimModel())
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


TEST_F(FP32NonQuantizedTest, TwoCropsWithDistanceInBetween) {
    std::vector<float> input_data ={1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                                    17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
                                    33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0};

    std::vector<float> expected_result = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                                          33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0};

    assert_that().onInferModel(twoCropsModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(FP32NonQuantizedTest, ThreeCropsWithDistanceInBetween) {
    std::vector<float> input_data ={1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                                    17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                    25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
                                    33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0};

    std::vector<float> expected_result = {1.0,   2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                                          17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                          33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0};

    assert_that().onInferModel(threeCropsModel())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input_and_expected_output(input_data, expected_result);
}

TEST_F(FP32NonQuantizedTest, ThreeCropsWithReshapeWithDistanceInBetween) {
    std::vector<float> input_data ={1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                                    17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                    25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
                                    33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0};

    std::vector<float> expected_result = {1.0,   2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                                          17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                          33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0};

    assert_that().onInferModel(threeCropsWithReshapeModel())
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

TEST_F(FP32NonQuantizedTest, ConcatWithConstInputPropagatedForward) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    std::vector<float> expected_result = {121.0, 121.0, 121.0, 121.0, 121.0,
                                          121.0, 121.0, 121.0, 121.0, 121.0,
                                          121.0, 121.0, 121.0, 121.0, 121.0,
                                          121.0, 121.0, 121.0, 121.0, 121.0};

    assert_that().onInferModel(concatModelWithConstLayer())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, InputSplitConcatPropagateForward) {
    std::vector<float> input_data(64, 1.0f);
    std::vector<float> expected_result(10, 64.f);

    assert_that().onInferModel(InputSplitConcatModel())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, InputSplitConcatUnalignedPropagateForward) {
    std::vector<float> input_data(20, 1.0f);
    std::vector<float> expected_result(10, 20.f);

    assert_that().onInferModel(InputSplitConcatModelUnaligned())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, InputSplitConcatReshapeUnalignedPropagateForward) {
    std::vector<float> input_data(20, 1.0f);
    std::vector<float> expected_result(10, 20.f);

    assert_that().onInferModel(InputSplitConcatReshapeModelUnaligned())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, LSTMCellPropagateForward) {
    std::vector<float> input_data(96, 0.10f);
    std::vector<float> expected_result(32, 0.14366889f);

    assert_that().onInferModel(LSTMCellOnlyModel()).withWeigthsPattern({0.1f})
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, LSTMCellUnalignedPropagateForward) {
    std::vector<float> input_data(30, 0.10f);
    std::vector<float> expected_result(10, 0.10488615);

    assert_that().onInferModel(LSTMCellOnlyModelUnaligned()).withWeigthsPattern({0.1f})
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, TI1PropagateForward) {
    std::vector<float> input_data(10, 0.10f);
    std::vector<float> expected_result1(10, 0.22478661f);
    std::vector<float> expected_result2(12, 0.22699516);

    assert_that().onInferModel(TIModelWithLSTMCell1()).withWeigthsPattern("ScaleShift_1", {1.0f}).withWeigthsPattern({0.1f})
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result1).equals_to(expected_result2);
}

TEST_F(FP32NonQuantizedTest, TI1PropagateForwardWithoutScaleShift) {
    std::vector<float> input_data(10, 0.10f);
    std::vector<float> expected_result1(10, 0.22478661f);
    std::vector<float> expected_result2(12, 0.22699516f);

    assert_that().onInferModel(TIModelWithLSTMCell1WithoutScaleShift()).withWeigthsPattern({0.1f})
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result1).equals_to(expected_result2);
}

// DISABLED DUE TO (31901)
TEST_F(FP32NonQuantizedTest, DISABLED_TI1AlignedPropagateForward) {
    std::vector<float> input_data(32, 0.1f);
    std::vector<float> expected_result1(32, 0.25883245);
    std::vector<float> expected_result2(12, 0.59515548f);

    assert_that().onInferModel(TIModelWithLSTMCell1Aligned()).withWeigthsPattern({0.1f})
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result1).And().equals_to(expected_result2);
}

// DISABLED DUE TO (31901)
TEST_F(FP32NonQuantizedTest, DISABLED_TI3AlignedPropagateForward) {
    std::vector<float> input_data(96, 0.1f);
    std::vector<float> expected_result1(32, 0.42592844f);
    std::vector<float> expected_result2(12, 0.97069889f);

    assert_that().onInferModel(TIModelWithLSTMCell3Aligned()).withWeigthsPattern({0.1f})
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result1).And().equals_to(expected_result2);
}

TEST_F(FP32NonQuantizedTest, TI2PropagateForward) {
    std::vector<float> input_data(20, 0.1f);
    std::vector<float> expected_result1(10, 0.22478661f);
    std::vector<float> expected_result2(12, 0.22699516f);

    assert_that().onInferModel(TIModelWithLSTMCell2()).withWeigthsPattern({0.1f})
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result1).equals_to(expected_result2);
}

TEST_F(FP32NonQuantizedTest, EltwiseWithConstInputPropagatedForward) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

    assert_that().onInferModel(eltwiseSumModelWithConstLayer())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, EltwiseWithConstInputReorderPropagatedForward) {
    std::vector<float> input_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_result = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

    assert_that().onInferModel(eltwiseSumModelWithConstLayer2())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, EltwiseMulWithConstInputReorderPropagatedForward) {
    std::vector<float> input_data = {3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0};
    std::vector<float> expected_result = {6.0, 2.0, 6.0, 2.0, 6.0, 2.0, 6.0, 2.0, 6.0, 2.0};

    assert_that().onInferModel(eltwiseMulModelWithConstLayer())
        .inNotCompactMode().withWeigthsPattern({2}).gna().propagate_forward().onCPU()
        .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, PowerWithScaleFactorPropagateForward) {
    std::vector<float> input_data(10, 2.f);
    std::vector<float> expected_result(12, 21.f);

    assert_that().onInferModel(PowerWithScaleFactor1())
            .inNotCompactMode().withWeigthsPattern({1}).gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatThroughScaleShiftPropagateForward) {
    std::vector<float> input_data(30, 1.f);
    std::vector<float> expected_result(20, 41.f);
    expected_result.insert(expected_result.end(), 20, 2.f);

    assert_that().onInferModel(SplitToConcatThroughScaleShift())
            .inNotCompactMode().withWeigthsPattern({1}).gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, TwoOutputsPropagateForward) {
    std::vector<float> input_data(10, 1);
    std::vector<float> result1(20, 11.f);
    std::vector<float> result2(10, 11.f);

    assert_that().onInferModel(TwoOutputs())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with().input("input_1", input_data).result().equals_to(result1).And().equals_to(result2);
}

TEST_F(FP32NonQuantizedTest, TwoOutputsDiffPrecisionPropagateForward) {
    std::vector<float> input_data(10, 1);
    std::vector<float> result1(10, 11.f);
    std::vector<float> result2(20, 11.f);

    assert_that().onInferModel(TwoOutputsDiffPrecision())
        .inNotCompactMode().gna().propagate_forward().onCPU()
        .called_with().input("input_1", input_data).result().equals_to(result1).And().equals_to(result2);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith2InputsNotAlignedNoFC) {
    std::vector<float> input_data(20);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith2InputsNotAlignedNoFC())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith2By50InputsNotAlignedNoFC) {
    std::vector<float> input_data(100);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith2By50InputsNotAlignedNoFC())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith2By50InputsNotAlignedNoFCWithInCopyWithOutCopy) {
    std::vector<float> input_data(100);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith2By50InputsNotAlignedNoFCWithInCopyWithOutCopy())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith3InputsNotAlignedNoFC) {
    std::vector<float> input_data(30);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith3InputsNotAlignedNoFC())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith4InputsNotAlignedNoFC) {
    std::vector<float> input_data(40);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith4InputsNotAlignedNoFC())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith4InputsNotAlignedNoFCWithOutCopy) {
    std::vector<float> input_data(40);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith4InputsNotAlignedNoFCWithOutCopy())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith10InputsNotAlignedNoFC) {
    std::vector<float> input_data(100);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith10InputsNotAlignedNoFC())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith10InputsNotAlignedNoFCWithOutCopy) {
    std::vector<float> input_data(100);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith10InputsNotAlignedNoFCWithOutCopy())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith10By1InputsNotAlignedNoFCWithOutCopy) {
    std::vector<float> input_data(10);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith10By1InputsNotAlignedNoFCWithOutCopy())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith2InputsNotAlignedWithFC) {
    std::vector<float> input_data(20);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    std::vector<float> expected_result(10, 211.0f);
    assert_that().onInferModel(SplitToConcatWith2InputsNotAlignedWithFC())
            .inNotCompactMode().withWeigthsPattern({1}).gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith3InputsNotAlignedWithFC) {
    std::vector<float> input_data(30);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    std::vector<float> expected_result(10, 466.0f);
    assert_that().onInferModel(SplitToConcatWith3InputsNotAlignedWithFC())
            .inNotCompactMode().withWeigthsPattern({1}).gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith3By512InputsWithOutCopy) {
    std::vector<float> input_data(1536);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith3By512InputsWithOutCopy())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith10InputsNotAlignedWithFC) {
    std::vector<float> input_data(100);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    std::vector<float> expected_result(10, 5051.0f);
    assert_that().onInferModel(SplitToConcatWith10InputsNotAlignedWithFC())
            .inNotCompactMode().withWeigthsPattern({1}).gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, DISABLED_SplitToConcatWith2InputsAlignedNoFC) {
    std::vector<float> input_data(64);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith2InputsAlignedNoFC())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, DISABLED_SplitToConcatWith2By64InputsAlignedNoFC) {
    std::vector<float> input_data(128);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith2By64InputsAlignedNoFC())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith2Inputs1360NotAlignedNoFC) {
    std::vector<float> input_data(1360);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith2Inputs1360NotAlignedNoFC())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith2By64InputsAlignedNoFCWithOutCopy) {
    std::vector<float> input_data(128);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith2By64InputsAlignedNoFCWithOutCopy())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith2InputsAlignedNoFCWithInCopyWithOutCopy) {
    std::vector<float> input_data(64);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith2InputsAlignedNoFCWithInCopyWithOutCopy())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, DISABLED_SplitToConcatWith3InputsAlignedNoFC) {
    std::vector<float> input_data(96);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith3InputsAlignedNoFC())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith3InputsAlignedNoFCWithInCopyWithOutCopy) {
    std::vector<float> input_data(96);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith3InputsAlignedNoFCWithInCopyWithOutCopy())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, DISABLED_SplitToConcatWith10InputsAlignedNoFC) {
    std::vector<float> input_data(320);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith10InputsAlignedNoFC())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith10InputsAlignedNoFCWithInCopyWithOutCopy) {
    std::vector<float> input_data(320);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    assert_that().onInferModel(SplitToConcatWith10InputsAlignedNoFCWithInCopyWithOutCopy())
            .inNotCompactMode().gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(input_data);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith2InputsAlignedWithFC) {
    std::vector<float> input_data(64);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    std::vector<float> expected_result(32, 2081.0f);
    assert_that().onInferModel(SplitToConcatWith2InputsAlignedWithFC())
            .inNotCompactMode().withWeigthsPattern({1}).gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith2InputsAlignedWithFCWithInCopy) {
    std::vector<float> input_data(64);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    std::vector<float> expected_result(32, 2081.0f);
    assert_that().onInferModel(SplitToConcatWith2InputsAlignedWithFCWithInCopy())
            .inNotCompactMode().withWeigthsPattern({1}).gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith3InputsAlignedWithFC) {
    std::vector<float> input_data(96);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    std::vector<float> expected_result(32, 4657.0f);
    assert_that().onInferModel(SplitToConcatWith3InputsAlignedWithFC())
            .inNotCompactMode().withWeigthsPattern({1}).gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith3InputsAlignedWithFCWithInCopy) {
    std::vector<float> input_data(96);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    std::vector<float> expected_result(32, 4657.0f);
    assert_that().onInferModel(SplitToConcatWith3InputsAlignedWithFCWithInCopy())
            .inNotCompactMode().withWeigthsPattern({1}).gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith10InputsAlignedWithFC) {
    std::vector<float> input_data(320);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    std::vector<float> expected_result(32, 51361.0f);
    assert_that().onInferModel(SplitToConcatWith10InputsAlignedWithFC())
            .inNotCompactMode().withWeigthsPattern({1}).gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, SplitToConcatWith10InputsAlignedWithFCWithInCopy) {
    std::vector<float> input_data(320);
    std::iota(input_data.begin(), input_data.end(), 1.0f);
    std::vector<float> expected_result(32, 51361.0f);
    assert_that().onInferModel(SplitToConcatWith10InputsAlignedWithFCWithInCopy())
            .inNotCompactMode().withWeigthsPattern({1}).gna().propagate_forward().onCPU()
            .called_with_input(input_data).equals_to(expected_result);
}

TEST_F(FP32NonQuantizedTest, ReshapeConvolutionLessThan48Filters) {
    std::vector<float> input_data(800, 1.f);
    std::vector<float> expected_result(1600, 8.f);

    assert_that().onInferModel(ReshapeConvolutionLessThan48Filters())
            .inNotCompactMode()
            .withWeigthsPattern({1})
            .gna()
            .propagate_forward()
            .onCPU()
            .called_with_input(input_data)
            .equals_to(expected_result);
}

