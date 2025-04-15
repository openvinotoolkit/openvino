// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/detection_output.hpp"

namespace {
using ov::test::DetectionOutputLayerTest;
using ov::test::ParamsWhichSizeDepends;

const int numClasses = 11;
const int backgroundLabelId = 0;
const std::vector<int> topK = {75};
const std::vector<std::vector<int>> keepTopK = { {50}, {100} };
const std::vector<std::string> codeType = {"caffe.PriorBoxParameter.CORNER", "caffe.PriorBoxParameter.CENTER_SIZE"};
const float nmsThreshold = 0.5f;
const float confidenceThreshold = 0.3f;
const std::vector<bool> clipAfterNms = {true, false};
const std::vector<bool> clipBeforeNms = {true, false};
const std::vector<bool> decreaseLabelId = {true, false};
const float objectnessScore = 0.4f;
const std::vector<size_t> numberBatch = {1, 2};

const auto commonAttributes = ::testing::Combine(
        ::testing::Values(numClasses),
        ::testing::Values(backgroundLabelId),
        ::testing::ValuesIn(topK),
        ::testing::ValuesIn(keepTopK),
        ::testing::ValuesIn(codeType),
        ::testing::Values(nmsThreshold),
        ::testing::Values(confidenceThreshold),
        ::testing::ValuesIn(clipAfterNms),
        ::testing::ValuesIn(clipBeforeNms),
        ::testing::ValuesIn(decreaseLabelId)
);

/* =============== 3 inputs cases =============== */

const std::vector<ParamsWhichSizeDepends> specificParams3In = {
    ParamsWhichSizeDepends{true, true, true, 1, 1, {1, 60}, {1, 165}, {1, 1, 60}, {}, {}},
    ParamsWhichSizeDepends{true, false, true, 1, 1, {1, 660}, {1, 165}, {1, 1, 60}, {}, {}},
    ParamsWhichSizeDepends{false, true, true, 1, 1, {1, 60}, {1, 165}, {1, 2, 60}, {}, {}},
    ParamsWhichSizeDepends{false, false, true, 1, 1, {1, 660}, {1, 165}, {1, 2, 60}, {}, {}},

    ParamsWhichSizeDepends{true, true, false, 10, 10, {1, 60}, {1, 165}, {1, 1, 75}, {}, {}},
    ParamsWhichSizeDepends{true, false, false, 10, 10, {1, 660}, {1, 165}, {1, 1, 75}, {}, {}},
    ParamsWhichSizeDepends{false, true, false, 10, 10, {1, 60}, {1, 165}, {1, 2, 75}, {}, {}},
    ParamsWhichSizeDepends{false, false, false, 10, 10, {1, 660}, {1, 165}, {1, 2, 75}, {}, {}}
};

const auto params3Inputs = ::testing::Combine(
        commonAttributes,
        ::testing::ValuesIn(specificParams3In),
        ::testing::ValuesIn(numberBatch),
        ::testing::Values(0.0f),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput3In, DetectionOutputLayerTest, params3Inputs, DetectionOutputLayerTest::getTestCaseName);

/* =============== 5 inputs cases =============== */

const std::vector<ParamsWhichSizeDepends> specificParams5In = {
    ParamsWhichSizeDepends{true, true, true, 1, 1, {1, 60}, {1, 165}, {1, 1, 60}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{true, false, true, 1, 1, {1, 660}, {1, 165}, {1, 1, 60}, {1, 30}, {1, 660}},
    ParamsWhichSizeDepends{false, true, true, 1, 1, {1, 60}, {1, 165}, {1, 2, 60}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{false, false, true, 1, 1, {1, 660}, {1, 165}, {1, 2, 60}, {1, 30}, {1, 660}},

    ParamsWhichSizeDepends{true, true, false, 10, 10, {1, 60}, {1, 165}, {1, 1, 75}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{true, false, false, 10, 10, {1, 660}, {1, 165}, {1, 1, 75}, {1, 30}, {1, 660}},
    ParamsWhichSizeDepends{false, true, false, 10, 10, {1, 60}, {1, 165}, {1, 2, 75}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{false, false, false, 10, 10, {1, 660}, {1, 165}, {1, 2, 75}, {1, 30}, {1, 660}}
};

const auto params5Inputs = ::testing::Combine(
        commonAttributes,
        ::testing::ValuesIn(specificParams5In),
        ::testing::ValuesIn(numberBatch),
        ::testing::Values(objectnessScore),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput5In, DetectionOutputLayerTest, params5Inputs, DetectionOutputLayerTest::getTestCaseName);

}  // namespace
