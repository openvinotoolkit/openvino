// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_op_tests/experimental_detectron_detection_output.hpp"

namespace {
using ov::test::ExperimentalDetectronDetectionOutputLayerTest;

const std::vector<float> score_threshold = { 0.01000000074505806f };

const std::vector<float> nms_threshold = { 0.2f };

//// specifies maximal delta of logarithms for width and height
const std::vector<float> max_delta_log_wh = { 2.0f };

// specifies number of detected classes
const std::vector<int64_t> num_classes = { 2 };

// specifies maximal number of detections per class
const std::vector<int64_t> post_nms_count = { 500 };

// specifies maximual number of detections per image
const std::vector<size_t> max_detections_per_image = { 5 };

// a flag specifies whether to delete background classes or not
// `true`  means background classes should be deleted,
// `false` means background classes shouldn't be deleted.
const std::vector<bool> class_agnostic_box_regression = { true };

// specifies deltas of weights
const std::vector<std::vector<float>> deltas_weights = { {10.0f, 10.0f, 5.0f, 5.0f} };

const std::vector<std::vector<ov::test::InputShape>> inputShapes = {
        // inputRois / inputDeltas / inputScores / inputImInfos
        ov::test::static_shapes_to_test_representation({{16, 4}, {16, 8}, {16, 2}, {1, 3}}),
        {
            {{-1, -1}, {{16, 4}, {16, 4}}},
            {{-1, -1}, {{16, 8}, {16, 8}}},
            {{-1, -1}, {{16, 2}, {16, 2}}},
            {{-1, -1}, {{1, 3}, {1, 3}}}
        },
        {
            {{{16, 32}, {4, 8}}, {{16, 4}, {16, 4}}},
            {{{16, 32}, {8, 16}}, {{16, 8}, {16, 8}}},
            {{{16, 32}, {2, 4}}, {{16, 2}, {16, 2}}},
            {{{1, 2}, {3, 6}}, {{1, 3}, {1, 3}}}
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronDetectionOutput, ExperimentalDetectronDetectionOutputLayerTest,
         ::testing::Combine(
                 ::testing::ValuesIn(inputShapes),
                 ::testing::ValuesIn(score_threshold),
                 ::testing::ValuesIn(nms_threshold),
                 ::testing::ValuesIn(max_delta_log_wh),
                 ::testing::ValuesIn(num_classes),
                 ::testing::ValuesIn(post_nms_count),
                 ::testing::ValuesIn(max_detections_per_image),
                 ::testing::ValuesIn(class_agnostic_box_regression),
                 ::testing::ValuesIn(deltas_weights),
                 ::testing::Values(ov::element::Type_t::f32),
                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
         ExperimentalDetectronDetectionOutputLayerTest::getTestCaseName);

} // namespace
