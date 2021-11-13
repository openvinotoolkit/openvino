// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/detection_output.hpp"

using namespace LayerTestsDefinitions;

namespace {

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

/* =============== 3 inputs cases dynamic =============== */

const std::vector<ParamsWhichSizeDepends> specificParams3InDynamic = {
    // dynamic input shapes
    ParamsWhichSizeDepends {
        true, true, true, 1, 1,
        {
            // input model dynamic shapes
            {ov::Dimension::dynamic(), ov::Dimension::dynamic()},
            // input tensor shapes
            {{1, 60}, {1, 120}}
        },
        {
            // input model dynamic shapes
            {ov::Dimension::dynamic(), ov::Dimension::dynamic()},
            // input tensor shapes
            {{1, 165}, {1, 330}}},
        {
            // input model dynamic shapes
            {ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
            // input tensor shapes
            {{1, 1, 60}, {1, 1, 120}}
        },
        {},
        {}
    },
    ParamsWhichSizeDepends {
        true, false, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 60}, {1, 1, 120}}},
        {},
        {}
    },
    ParamsWhichSizeDepends {
        false, true, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {},
        {}
    },
    ParamsWhichSizeDepends {
        false, false, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {},
        {}
    },
    ParamsWhichSizeDepends {
        true, true, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {},
        {}},
    ParamsWhichSizeDepends {
        true, false, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {},
        {}},
    ParamsWhichSizeDepends {
        false, true, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 75}, {1, 2, 150}}},
        {},
        {}},
    ParamsWhichSizeDepends {
        false, false, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 75}, {1, 2, 150}}},
        {},
        {}},
};

const auto params3InputsDynamic = ::testing::Combine(
        commonAttributes,
        ::testing::ValuesIn(specificParams3InDynamic),
        ::testing::ValuesIn(numberBatch),
        ::testing::Values(0.0f),
        ::testing::Values(false, true),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput3InDynamic, DetectionOutputLayerTest, params3InputsDynamic, DetectionOutputLayerTest::getTestCaseName);

/* =============== 3 inputs cases static =============== */
const std::vector<ParamsWhichSizeDepends> specificParams3InStatic = {
    // static shapes
    DetectionOutputLayerTest::fromStatic(true, true, true, 1, 1, {1, 60}, {1, 165}, {1, 1, 60}, {}, {}),
    DetectionOutputLayerTest::fromStatic(true, false, true, 1, 1, {1, 660}, {1, 165}, {1, 1, 60}, {}, {}),
    DetectionOutputLayerTest::fromStatic(false, true, true, 1, 1, {1, 60}, {1, 165}, {1, 2, 60}, {}, {}),
    DetectionOutputLayerTest::fromStatic(false, false, true, 1, 1, {1, 660}, {1, 165}, {1, 2, 60}, {}, {}),

    DetectionOutputLayerTest::fromStatic(true, true, false, 10, 10, {1, 60}, {1, 165}, {1, 1, 75}, {}, {}),
    DetectionOutputLayerTest::fromStatic(true, false, false, 10, 10, {1, 660}, {1, 165}, {1, 1, 75}, {}, {}),
    DetectionOutputLayerTest::fromStatic(false, true, false, 10, 10, {1, 60}, {1, 165}, {1, 2, 75}, {}, {}),
    DetectionOutputLayerTest::fromStatic(false, false, false, 10, 10, {1, 660}, {1, 165}, {1, 2, 75}, {}, {})
};

const auto params3InputsStatic = ::testing::Combine(
        commonAttributes,
        ::testing::ValuesIn(specificParams3InStatic),
        ::testing::ValuesIn(numberBatch),
        ::testing::Values(0.0f),
        ::testing::Values(false),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput3InStatic, DetectionOutputLayerTest, params3InputsStatic, DetectionOutputLayerTest::getTestCaseName);

/* =============== 5 inputs cases =============== */

const std::vector<ParamsWhichSizeDepends> specificParams5InDynamic = {
    // dynamic input shapes
    ParamsWhichSizeDepends {
        true, true, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 60}, {1, 1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
    },
    ParamsWhichSizeDepends {
        true, false, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 60}, {1, 1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
    },
    ParamsWhichSizeDepends {
        false, true, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}}
    },
    ParamsWhichSizeDepends {
        false, false, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}}
    },

    ParamsWhichSizeDepends {
        true, true, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}}
    },
    ParamsWhichSizeDepends {
        true, false, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}}
    },
    ParamsWhichSizeDepends {
        false, true, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 75}, {1, 2, 150}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}}
    },
    ParamsWhichSizeDepends {
        false, false, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 75}, {1, 2, 150}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}}
    },
};

const auto params5InputsDynamic = ::testing::Combine(
        commonAttributes,
        ::testing::ValuesIn(specificParams5InDynamic),
        ::testing::ValuesIn(numberBatch),
        ::testing::Values(objectnessScore),
        ::testing::Values(false, true),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput5InDynamic, DetectionOutputLayerTest, params5InputsDynamic, DetectionOutputLayerTest::getTestCaseName);

const std::vector<ParamsWhichSizeDepends> specificParams5InStatic = {
    // static shapes
    DetectionOutputLayerTest::fromStatic(true, true, true, 1, 1, {1, 60}, {1, 165}, {1, 1, 60}, {1, 30}, {1, 60}),
    DetectionOutputLayerTest::fromStatic(true, false, true, 1, 1, {1, 660}, {1, 165}, {1, 1, 60}, {1, 30}, {1, 660}),
    DetectionOutputLayerTest::fromStatic(false, true, true, 1, 1, {1, 60}, {1, 165}, {1, 2, 60}, {1, 30}, {1, 60}),
    DetectionOutputLayerTest::fromStatic(false, false, true, 1, 1, {1, 660}, {1, 165}, {1, 2, 60}, {1, 30}, {1, 660}),

    DetectionOutputLayerTest::fromStatic(true, true, false, 10, 10, {1, 60}, {1, 165}, {1, 1, 75}, {1, 30}, {1, 60}),
    DetectionOutputLayerTest::fromStatic(true, false, false, 10, 10, {1, 660}, {1, 165}, {1, 1, 75}, {1, 30}, {1, 660}),
    DetectionOutputLayerTest::fromStatic(false, true, false, 10, 10, {1, 60}, {1, 165}, {1, 2, 75}, {1, 30}, {1, 60}),
    DetectionOutputLayerTest::fromStatic(false, false, false, 10, 10, {1, 660}, {1, 165}, {1, 2, 75}, {1, 30}, {1, 660})
};

const auto params5InputsStatic = ::testing::Combine(
        commonAttributes,
        ::testing::ValuesIn(specificParams5InStatic),
        ::testing::ValuesIn(numberBatch),
        ::testing::Values(objectnessScore),
        ::testing::Values(false),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput5InStatic, DetectionOutputLayerTest, params5InputsStatic, DetectionOutputLayerTest::getTestCaseName);

}  // namespace
