// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_op_tests/roi_align.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ROIAlignLayerTest;
using ov::test::ROIAlignV9LayerTest;

class ROIAlignLayerGPUTest : public ROIAlignLayerTest {
protected:
    void SetUp() override {
        ROIAlignLayerTest::SetUp();
        auto param = this->GetParam();
        ov::element::Type model_type = std::get<7>(param);
        targetDevice = std::get<8>(param);
        if (targetDevice == "GPU" && model_type == ov::element::f32) {
            abs_threshold = 1e-5;   // ROIAlign may introduce small differences when output values are very small,
            rel_threshold = 5e-2;   // so the threshold should account for this behavior.
        }
    }
};

class ROIAlignV9LayerGPUTest : public ROIAlignV9LayerTest {
protected:
    void SetUp() override {
        ROIAlignV9LayerTest::SetUp();
        auto param = this->GetParam();
        ov::element::Type model_type = std::get<8>(param);
        targetDevice = std::get<9>(param);

        if (targetDevice == "GPU" && model_type == ov::element::f32) {
            abs_threshold = 1e-5;   // ROIAlign may introduce small differences when output values are very small,
            rel_threshold = 5e-2;   // so the threshold should account for this behavior.
        }
    }
};

const std::vector<ov::element::Type> netPRCs = {
    ov::element::f32
    // There is no possibility to test ROIAlign in fp16 precision,
    // because on edge cases where in fp32 version ROI value is
    // a little bit smaller than the nearest integer value,
    // it would be bigger than the nearest integer in fp16 precision.
    // Such behavior leads to completely different results of ROIAlign
    // in fp32 and fp16 precisions.
    // In real AI applications this problem is solved by precision-aware training.

    // ov::element::f16
};

TEST_P(ROIAlignLayerGPUTest, Inference) {
    run();
}

TEST_P(ROIAlignV9LayerGPUTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_average,
                         ROIAlignLayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                            std::vector<std::vector<ov::Shape>>{{{3, 8, 16, 16}},
                                                                                                {{2, 1, 16, 16}},
                                                                                                {{2, 1, 8, 16}}})),
                                            ::testing::Values(std::vector<size_t>{2, 4}),
                                            ::testing::Values(2),
                                            ::testing::Values(2),
                                            ::testing::ValuesIn(std::vector<float>{1, 0.625}),
                                            ::testing::Values(2),
                                            ::testing::Values("avg"),
                                            ::testing::ValuesIn(netPRCs),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ROIAlignLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_max,
                         ROIAlignLayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                            std::vector<std::vector<ov::Shape>>{{{2, 8, 20, 20}},
                                                                                                {{2, 1, 20, 20}},
                                                                                                {{2, 1, 10, 20}}})),
                                            ::testing::Values(std::vector<size_t>{2, 4}),
                                            ::testing::Values(2),
                                            ::testing::Values(2),
                                            ::testing::ValuesIn(std::vector<float>{1, 0.625}),
                                            ::testing::Values(2),
                                            ::testing::Values("max"),
                                            ::testing::ValuesIn(netPRCs),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ROIAlignLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_avg_asym,
                         ROIAlignV9LayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                            std::vector<std::vector<ov::Shape>>{{{2, 1, 8, 8}},
                                                                                                {{2, 8, 20, 20}},
                                                                                                {{2, 1, 20, 20}},
                                                                                                {{2, 1, 10, 20}}})),
                                            ::testing::Values(ov::Shape{2, 4}),
                                            ::testing::Values(2),
                                            ::testing::Values(2),
                                            ::testing::ValuesIn(std::vector<float>{1, 0.625}),
                                            ::testing::Values(2),
                                            ::testing::Values("avg"),
                                            ::testing::Values("asymmetric"),
                                            ::testing::ValuesIn(netPRCs),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ROIAlignV9LayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_avg_hpfn,
                         ROIAlignV9LayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                            std::vector<std::vector<ov::Shape>>{{{2, 1, 8, 8}},
                                                                                                {{2, 8, 20, 20}},
                                                                                                {{2, 1, 20, 20}},
                                                                                                {{2, 1, 10, 20}}})),
                                            ::testing::Values(ov::Shape{2, 4}),
                                            ::testing::Values(2),
                                            ::testing::Values(2),
                                            ::testing::ValuesIn(std::vector<float>{1, 0.625}),
                                            ::testing::Values(2),
                                            ::testing::Values("avg"),
                                            ::testing::Values("half_pixel_for_nn"),
                                            ::testing::ValuesIn(netPRCs),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ROIAlignV9LayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_max_hp,
                         ROIAlignV9LayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                            std::vector<std::vector<ov::Shape>>{{{2, 1, 8, 8}},
                                                                                                {{2, 8, 20, 20}},
                                                                                                {{2, 1, 20, 20}},
                                                                                                {{2, 1, 10, 20}}})),
                                            ::testing::Values(ov::Shape{2, 4}),
                                            ::testing::Values(2),
                                            ::testing::Values(2),
                                            ::testing::ValuesIn(std::vector<float>{1, 0.625}),
                                            ::testing::Values(2),
                                            ::testing::Values("max"),
                                            ::testing::Values("half_pixel"),
                                            ::testing::ValuesIn(netPRCs),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ROIAlignV9LayerGPUTest::getTestCaseName);
}  // namespace
