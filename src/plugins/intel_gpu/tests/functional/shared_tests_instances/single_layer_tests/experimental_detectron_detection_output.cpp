// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/experimental_detectron_detection_output.hpp"

namespace {
using ov::test::ExperimentalDetectronDetectionOutputLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f16,
    ov::element::f32,
};

const std::vector<float> score_threshold = {0.01f, 0.8f};

const std::vector<float> nms_threshold = {0.2f, 0.5f};

// specifies maximal delta of logarithms for width and height
const std::vector<float> max_delta_log_wh = {2.0f, 5.0f};

// specifies number of detected classes
const std::vector<int64_t> num_classes = {2};

// specifies maximal number of detections per class
const std::vector<int64_t> post_nms_count = {5, 25};

// specifies maximual number of detections per image
// there is assigning size_t rois_num = attrs.max_detections_per_image at docs/template_plugin/backend/evaluates_map.cpp:2117,
// as a result we have to set max_detections_per_image equal to rois_num
const std::vector<size_t> max_detections_per_image16 = {16};
const std::vector<size_t> max_detections_per_image = {5, 25};

// a flag specifies whether to delete background classes or not
// `true`  means background classes should be deleted,
// `false` means background classes shouldn't be deleted.
const bool class_agnostic_box_regression_true = true;
const bool class_agnostic_box_regression_false = false;

// specifies deltas of weights
const std::vector<std::vector<float>> deltas_weights = {{10.0f, 10.0f, 5.0f, 5.0f}};

const std::vector<ov::test::InputShape> inputShapes = {
    // inputRois / inputDeltas / inputScores / inputImInfos
    ov::test::static_shapes_to_test_representation({{16, 4}, {16, 8}, {16, 2}, {1, 3}}),
};


INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronDetectionOutput,
                         ExperimentalDetectronDetectionOutputLayerTest,
                         ::testing::Combine(::testing::Values(inputShapes),
                                            ::testing::ValuesIn(score_threshold),
                                            ::testing::ValuesIn(nms_threshold),
                                            ::testing::ValuesIn(max_delta_log_wh),
                                            ::testing::ValuesIn(num_classes),
                                            ::testing::ValuesIn(post_nms_count),
                                            ::testing::ValuesIn(max_detections_per_image16),
                                            ::testing::Values(class_agnostic_box_regression_true),
                                            ::testing::ValuesIn(deltas_weights),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ExperimentalDetectronDetectionOutputLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_ExperimentalDetectronDetectionOutputMaxDetectionsPerImage,
                         ExperimentalDetectronDetectionOutputLayerTest,
                         ::testing::Combine(::testing::Values(inputShapes),
                                            ::testing::ValuesIn(score_threshold),
                                            ::testing::ValuesIn(nms_threshold),
                                            ::testing::ValuesIn(max_delta_log_wh),
                                            ::testing::ValuesIn(num_classes),
                                            ::testing::ValuesIn(post_nms_count),
                                            ::testing::ValuesIn(max_detections_per_image),
                                            ::testing::Values(class_agnostic_box_regression_true),
                                            ::testing::ValuesIn(deltas_weights),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ExperimentalDetectronDetectionOutputLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_ExperimentalDetectronDetectionOutput,
                         ExperimentalDetectronDetectionOutputLayerTest,
                         ::testing::Combine(::testing::Values(inputShapes),
                                            ::testing::ValuesIn(score_threshold),
                                            ::testing::ValuesIn(nms_threshold),
                                            ::testing::ValuesIn(max_delta_log_wh),
                                            ::testing::ValuesIn(num_classes),
                                            ::testing::ValuesIn(post_nms_count),
                                            ::testing::ValuesIn(max_detections_per_image),
                                            ::testing::Values(class_agnostic_box_regression_false),
                                            ::testing::ValuesIn(deltas_weights),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ExperimentalDetectronDetectionOutputLayerTest::getTestCaseName);


}  // namespace
