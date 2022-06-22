// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/experimental_detectron_detection_output.hpp"

#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"

using namespace ov::test;
using namespace ov::test::subgraph;

namespace {

const std::vector<ov::test::ElementType> netPrecisions = {
//    ov::element::Type_t::f16,
    ov::element::Type_t::f32,
};

const std::vector<float> score_threshold = {0.01f};

const std::vector<float> nms_threshold = {0.2f};

// specifies maximal delta of logarithms for width and height
const std::vector<float> max_delta_log_wh = {2.0f};

// specifies number of detected classes
const std::vector<int64_t> num_classes = {2};

// specifies maximal number of detections per class
const std::vector<int64_t> post_nms_count = {1};

// specifies maximual number of detections per image
const std::vector<size_t> max_detections_per_image = {1};

// a flag specifies whether to delete background classes or not
// `true`  means background classes should be deleted,
// `false` means background classes shouldn't be deleted.
const std::vector<bool> class_agnostic_box_regression = {true};

// specifies deltas of weights
const std::vector<std::vector<float>> deltas_weights = {{10.0f, 10.0f, 5.0f, 5.0f}};

const std::vector<std::vector<InputShape>> inputShapes = {
    // inputRois / inputDeltas / inputScores / inputImInfos
    static_shapes_to_test_representation({{1, 4}, {1, 8}, {1, 2}, {1, 3}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronDetectionOutput,
                         ExperimentalDetectronDetectionOutputLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(score_threshold),
                                            ::testing::ValuesIn(nms_threshold),
                                            ::testing::ValuesIn(max_delta_log_wh),
                                            ::testing::ValuesIn(num_classes),
                                            ::testing::ValuesIn(post_nms_count),
                                            ::testing::ValuesIn(max_detections_per_image),
                                            ::testing::ValuesIn(class_agnostic_box_regression),
                                            ::testing::ValuesIn(deltas_weights),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                            ::testing::Range(0, 1000)),
                         ExperimentalDetectronDetectionOutputLayerTest::getTestCaseName);


}  // namespace
