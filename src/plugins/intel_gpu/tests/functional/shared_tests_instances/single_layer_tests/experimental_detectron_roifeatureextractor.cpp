// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/experimental_detectron_roifeatureextractor.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace {
using ov::test::ExperimentalDetectronROIFeatureExtractorLayerTest;

class ExperimentalDetectronROIFeatureExtractorDegenerateTestGPU : public ExperimentalDetectronROIFeatureExtractorLayerTest {
protected:
        void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
                inputs.clear();
                const auto& funcInputs = function->inputs();

                for (size_t i = 0; i < funcInputs.size(); ++i) {
                        const auto& shape = targetInputStaticShapes[i];
                        ov::Tensor tensor;

                        if (i == 0) {
                                tensor = ov::Tensor(funcInputs[i].get_element_type(), shape);
                                auto* rois = tensor.data<float>();
                                const size_t roisNum = shape[0];

                                for (size_t r = 0; r < roisNum; ++r) {
                                        // Deliberately generate non-positive ROI areas to exercise
                                        // pyramid-level edge handling in the kernel.
                                        // NOTE: x1<x0 and y1>y0 => negative area, x1==x0 and y1==y0 => zero area.
                                        rois[4 * r + 0] = 10.0f;
                                        rois[4 * r + 1] = 10.0f;
                                        if ((r & 1) == 0) {
                                                rois[4 * r + 2] = 5.0f;
                                                rois[4 * r + 3] = 15.0f;
                                        } else {
                                                rois[4 * r + 2] = 10.0f;
                                                rois[4 * r + 3] = 10.0f;
                                        }
                                }
                        } else {
                                const float levelValue = static_cast<float>(i);
                                tensor = ov::Tensor(funcInputs[i].get_element_type(), shape);
                                auto* data = tensor.data<float>();
                                std::fill(data, data + tensor.get_size(), levelValue);
                        }

                        inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
                }
        }

        void run() override {
                SKIP_IF_CURRENT_TEST_IS_DISABLED();

                ASSERT_FALSE(targetStaticShapes.empty() && !function->get_parameters().empty())
                        << "Target Static Shape is empty!!!";

                compile_model();
                for (const auto& targetStaticShapeVec : targetStaticShapes) {
                        generate_inputs(targetStaticShapeVec);
                        ASSERT_NO_THROW(infer());

                        const auto output = inferRequest.get_output_tensor(0);
                        ASSERT_EQ(output.get_element_type(), ov::element::f32);
                        const auto* actual = output.data<float>();
                        for (size_t idx = 0; idx < output.get_size(); ++idx) {
                                ASSERT_NEAR(actual[idx], 1.0f, 1e-4f);
                        }
                }
        }
};

class ExperimentalDetectronROIFeatureExtractorLevelSelectionTestGPU : public ExperimentalDetectronROIFeatureExtractorLayerTest {
protected:
        void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
                inputs.clear();
                const auto& funcInputs = function->inputs();

                for (size_t i = 0; i < funcInputs.size(); ++i) {
                        const auto& shape = targetInputStaticShapes[i];
                        ov::Tensor tensor(funcInputs[i].get_element_type(), shape);

                        if (i == 0) {
                                auto* rois = tensor.data<float>();
                                const size_t roisNum = shape[0];
                                for (size_t r = 0; r < roisNum; ++r) {
                                        // Area=170*170. This sits near a pyramid boundary where
                                        // floor and round produce different levels.
                                        rois[4 * r + 0] = 0.0f;
                                        rois[4 * r + 1] = 0.0f;
                                        rois[4 * r + 2] = 170.0f;
                                        rois[4 * r + 3] = 170.0f;
                                }
                        } else {
                                // Fill each level input with a distinct constant to make level
                                // selection observable in the output tensor values.
                                const float levelValue = static_cast<float>(i);
                                auto* data = tensor.data<float>();
                                std::fill(data, data + tensor.get_size(), levelValue);
                        }

                        inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
                }
        }
};

TEST_P(ExperimentalDetectronROIFeatureExtractorDegenerateTestGPU, Inference) {
        run();
}

TEST_P(ExperimentalDetectronROIFeatureExtractorLevelSelectionTestGPU, Inference) {
        run();
}

const std::vector<int64_t> outputSize = {7, 14};
const std::vector<int64_t> samplingRatio = {1, 2, 3};

const std::vector<std::vector<int64_t>> pyramidScales = {
        {8, 16, 32, 64},
        {4, 8, 16, 32},
        {2, 4, 8, 16}
};

const std::vector<std::vector<ov::test::InputShape>> staticInputShape = {
        ov::test::static_shapes_to_test_representation({{1000, 4}, {1, 8, 200, 336}, {1, 8, 100, 168}, {1, 8, 50, 84}, {1, 8, 25, 42}}),
        ov::test::static_shapes_to_test_representation({{1000, 4}, {1, 16, 200, 336}, {1, 16, 100, 168}, {1, 16, 50, 84}, {1, 16, 25, 42}}),
        ov::test::static_shapes_to_test_representation({{1200, 4}, {1, 8, 200, 42}, {1, 8, 100, 336}, {1, 8, 50, 168}, {1, 8, 25, 84}})
};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalROI_static, ExperimentalDetectronROIFeatureExtractorLayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShape),
                                 ::testing::ValuesIn(outputSize),
                                 ::testing::ValuesIn(samplingRatio),
                                 ::testing::ValuesIn(pyramidScales),
                                 ::testing::Values(false),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ExperimentalDetectronROIFeatureExtractorLayerTest::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> static_input_shape_degenerate = {
        // Mirrors the crashing workload dimensions (gws ~= 49 x 80 x 500).
        ov::test::static_shapes_to_test_representation({{500, 4}, {1, 80, 200, 336}, {1, 80, 100, 168}, {1, 80, 50, 84}, {1, 80, 25, 42}})
};

INSTANTIATE_TEST_SUITE_P(smoke_DegenerateRoiArea,
                         ExperimentalDetectronROIFeatureExtractorDegenerateTestGPU,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_input_shape_degenerate),
                                 ::testing::Values(7),
                                 ::testing::Values(2),
                                 ::testing::Values(std::vector<int64_t>{8, 16, 32, 64}),
                                 ::testing::Values(false, true),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ExperimentalDetectronROIFeatureExtractorLayerTest::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> static_input_shape_level_selection = {
        ov::test::static_shapes_to_test_representation({{8, 4}, {1, 80, 200, 336}, {1, 80, 100, 168}, {1, 80, 50, 84}, {1, 80, 25, 42}})
};

INSTANTIATE_TEST_SUITE_P(smoke_LevelSelectionRegression,
                         ExperimentalDetectronROIFeatureExtractorLevelSelectionTestGPU,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_input_shape_level_selection),
                                 ::testing::Values(7),
                                 ::testing::Values(2),
                                 ::testing::Values(std::vector<int64_t>{8, 16, 32, 64}),
                                 ::testing::Values(false, true),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ExperimentalDetectronROIFeatureExtractorLayerTest::getTestCaseName);
} // namespace
