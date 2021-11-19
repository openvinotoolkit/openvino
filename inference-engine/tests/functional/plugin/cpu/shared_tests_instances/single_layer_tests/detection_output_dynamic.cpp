// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/detection_output_dynamic.hpp"
#include "single_layer_tests/detection_output_attributes.hpp"

using namespace LayerTestsDefinitions;

namespace {

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

const std::vector<ParamsWhichSizeDependsDynamic> specificParams3InDynamic = {
    // dynamic input shapes
    ParamsWhichSizeDependsDynamic {
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
    ParamsWhichSizeDependsDynamic {
        true, false, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 60}, {1, 1, 120}}},
        {},
        {}
    },
    ParamsWhichSizeDependsDynamic {
        false, true, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {},
        {}
    },
    ParamsWhichSizeDependsDynamic {
        false, false, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {},
        {}
    },
    ParamsWhichSizeDependsDynamic {
        true, true, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {},
        {}},
    ParamsWhichSizeDependsDynamic {
        true, false, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {},
        {}},
    ParamsWhichSizeDependsDynamic {
        false, true, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 75}, {1, 2, 150}}},
        {},
        {}},
    ParamsWhichSizeDependsDynamic {
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

INSTANTIATE_TEST_SUITE_P(
        smoke_DetectionOutput3InDynamic,
        DetectionOutputDynamicLayerTest,
        params3InputsDynamic,
        DetectionOutputDynamicLayerTest::getTestCaseName);

/* =============== 5 inputs cases =============== */

const std::vector<ParamsWhichSizeDependsDynamic> specificParams5InDynamic = {
    // dynamic input shapes
    ParamsWhichSizeDependsDynamic {
        true, true, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 60}, {1, 1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
    },
    ParamsWhichSizeDependsDynamic {
        true, false, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 60}, {1, 1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
    },
    ParamsWhichSizeDependsDynamic {
        false, true, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}}
    },
    ParamsWhichSizeDependsDynamic {
        false, false, true, 1, 1,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 60}, {1, 2, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}}
    },

    ParamsWhichSizeDependsDynamic {
        true, true, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}}
    },
    ParamsWhichSizeDependsDynamic {
        true, false, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 75}, {1, 1, 150}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 660}, {1, 1320}}}
    },
    ParamsWhichSizeDependsDynamic {
        false, true, false, 10, 10,
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 165}, {1, 330}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 2, 75}, {1, 2, 150}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 30}, {1, 60}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 60}, {1, 120}}}
    },
    ParamsWhichSizeDependsDynamic {
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

INSTANTIATE_TEST_SUITE_P(
        smoke_DetectionOutputDynamic5InDynamic,
        DetectionOutputDynamicLayerTest,
        params5InputsDynamic,
        DetectionOutputDynamicLayerTest::getTestCaseName);

}  // namespace
