// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "shared_test_classes/single_layer/is_inf.hpp"

using namespace ov::test;
using namespace ov::test::subgraph;

namespace {
std::vector<std::vector<InputShape>> inShapesStatic = {
        { {{}, {{2}}} },
        { {{}, {{2, 200}}} },
        { {{}, {{10, 200}}} },
        { {{}, {{1, 10, 100}}} },
        { {{}, {{4, 4, 16}}} },
        { {{}, {{1, 1, 1, 3}}} },
        { {{}, {{2, 17, 5, 4}}} },
        { {{}, {{2, 17, 5, 1}}} },
        { {{}, {{1, 2, 4}}} },
        { {{}, {{1, 4, 4}}} },
        { {{}, {{1, 4, 4, 1}}} },
        { {{}, {{16, 16, 16, 16, 16}}} },
        { {{}, {{16, 16, 16, 16, 1}}} },
        { {{}, {{16, 16, 16, 1, 16}}} },
        { {{}, {{16, 32, 1, 1, 1}}} },
        { {{}, {{1, 1, 1, 1, 1, 1, 3}}} },
        { {{}, {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}} }
};

std::vector<std::vector<InputShape>> inShapesDynamic = {
        {{{ngraph::Dimension(1, 10), 200}, {{2, 200}, {1, 200}}}}
};

std::vector<ElementType> netPrecisions = {
        ov::element::f32
};

std::vector<bool> detectNegative = {
    true, false
};

std::vector<bool> detectPositive = {
    true, false
};

std::map<std::string, std::string> additional_config = {};

const auto isInfParams = ::testing::Combine(
        ::testing::ValuesIn(inShapesStatic),
        ::testing::ValuesIn(detectNegative),
        ::testing::ValuesIn(detectPositive),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

const auto isInfParamsDyn = ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic),
        ::testing::ValuesIn(detectNegative),
        ::testing::ValuesIn(detectPositive),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));


TEST_P(IsInfLayerTest, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_static, IsInfLayerTest, isInfParams, IsInfLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_dynamic, IsInfLayerTest, isInfParamsDyn, IsInfLayerTest::getTestCaseName);
} // namespace
