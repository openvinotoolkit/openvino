// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_op_tests/roi_align.hpp"
#include "common_test_utils/test_constants.hpp"

#include <random>

namespace {
using ov::test::ROIAlignLayerTest;
using ov::test::ROIAlignV9LayerTest;

class ROIAlignLayerTestGPU : public ROIAlignLayerTest {
protected:
    void SetUp() override {
        ROIAlignLayerTest::SetUp();
        auto param = this->GetParam();
        ov::element::Type model_type = std::get<7>(param);
        if (model_type == ov::element::f32) {
            // ROIAlign involves calculations such as interpolation, pooling, and clamping,
            // so for small number, it can introduce more error than other ops.
            // Therefore, it needs to relax threshold for GPU device to avoid false-positive results.
            // Please see how to set abs_threshold in other frameworks , e.g, TensorFlow (1e-5), PyTorch (1e-5).
            abs_threshold = 1e-5;
        }
    }
};

class ROIAlignV9LayerTestGPU : public ROIAlignV9LayerTest {
protected:
    void SetUp() override {
        ROIAlignV9LayerTest::SetUp();
        auto param = this->GetParam();
        ov::element::Type model_type = std::get<8>(param);
        if (model_type == ov::element::f32) {
            // ROIAlign involves calculations such as interpolation, pooling, and clamping,
            // so for small number, it can introduce more error than other ops.
            // Therefore, it needs to relax threshold for GPU device to avoid false-positive results.
            // Please see how to set abs_threshold in other frameworks , e.g, TensorFlow (1e-5), PyTorch (1e-5).
            abs_threshold = 1e-5;
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

TEST_P(ROIAlignLayerTestGPU, Inference) {
    run();
}

TEST_P(ROIAlignV9LayerTestGPU, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_average,
                         ROIAlignLayerTestGPU,
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
                         ROIAlignLayerTestGPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_max,
                         ROIAlignLayerTestGPU,
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
                         ROIAlignLayerTestGPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_avg_asym,
                         ROIAlignV9LayerTestGPU,
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
                         ROIAlignV9LayerTestGPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_avg_hpfn,
                         ROIAlignV9LayerTestGPU,
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
                         ROIAlignV9LayerTestGPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_max_hp,
                         ROIAlignV9LayerTestGPU,
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
                         ROIAlignV9LayerTestGPU::getTestCaseName);
}  // namespace
