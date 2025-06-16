// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/simple_if.hpp"

using namespace ov::test;

namespace {
std::vector<std::vector<ov::test::InputShape>> inputShapes = {
    {
        {{}, {{5, 7}}},
        {{}, {{5, 7}}},
    },
    {{{}, {{30, 20, 10}}}, {{}, {{30, 20, 10}}}},
    {{{-1, -1, -1}, {{10, 20, 5}, {10, 0, 5}, {1, 5, 5}}}, {{-1, -1, -1}, {{10, 20, 5}, {10, 0, 5}, {1, 1, 5}}}},
    {{{-1, 5, -1}, {{10, 5, 10}, {2, 5, 5}, {1, 5, 5}}}, {{-1, 5, -1}, {{1, 5, 1}, {2, 5, 5}, {5, 5, 5}}}},
    {{{{1, 10}, {1, 10}, {1, 10}}, {{2, 5, 10}, {2, 5, 1}, {1, 5, 5}}},
     {{{1, 10}, {1, 10}, {1, 10}}, {{2, 5, 10}, {2, 1, 5}, {5, 5, 5}}}},
};

const std::vector<ov::test::ElementType> inTypes = {ov::test::ElementType::f32,
                                                    ov::test::ElementType::bf16,
                                                    ov::test::ElementType::i8};

std::vector<bool> conditions = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_If,
                         SimpleIfTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(inTypes),
                                            ::testing::ValuesIn(conditions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         SimpleIfTest::getTestCaseName);

TEST_P(SimpleIfTest, CompareWithRefs) {
    run();
};

INSTANTIATE_TEST_SUITE_P(smoke_If,
                         SimpleIf2OutTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(inTypes),
                                            ::testing::ValuesIn(conditions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         SimpleIf2OutTest::getTestCaseName);

TEST_P(SimpleIf2OutTest, CompareWithRefs) {
    run();
};

INSTANTIATE_TEST_SUITE_P(smoke_If,
                         SimpleIfNotConstConditionTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(inTypes),
                                            ::testing::ValuesIn(conditions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         SimpleIfNotConstConditionTest::getTestCaseName);

TEST_P(SimpleIfNotConstConditionTest, CompareWithRefs) {
    run();
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_2 = {
    {
        {{-1, -1, -1}, {{10, 20, 5}, {10, 20, 5}, {1, 5, 5}}},
    },
    {
        {{-1, 5, -1}, {{10, 5, 10}, {2, 5, 5}, {1, 5, 5}}},
    },
    {
        {{{0, 10}, {0, 10}, {0, 10}}, {{2, 5, 10}, {2, 5, 1}, {2, 5, 0}, {1, 5, 5}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_If,
                         SimpleIfNotConstConditionAndInternalDynamismTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes_2),
                                            ::testing::ValuesIn(inTypes),
                                            ::testing::ValuesIn(conditions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         SimpleIfNotConstConditionTest::getTestCaseName);

TEST_P(SimpleIfNotConstConditionAndInternalDynamismTest, CompareWithRefs) {
    run();
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_3 = {
    {
        {{-1, 2, -1}, {{1, 2, 0}, {2, 2, 5}}},
    },
    {
        {{{0, 10}, {0, 10}, {0, 10}}, {{2, 5, 10}, {2, 0, 0}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_If,
                         SimpleIfNotConstConditionAndDimsIncreaseTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes_3),
                                            ::testing::ValuesIn(inTypes),
                                            ::testing::ValuesIn(conditions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         SimpleIfNotConstConditionTest::getTestCaseName);

TEST_P(SimpleIfNotConstConditionAndDimsIncreaseTest, CompareWithRefs) {
    run();
};

// the axis of split in test suit "SimpleIfNotConstConditionUnusedOutputPortsTest" is hardcoded as 1, so shape[axis]
// should be static
std::vector<std::vector<ov::test::InputShape>> inputShapes_4 = {
    {
        {{}, {{5, 7}}},
    },
    {
        {{-1, 5, -1}, {{10, 5, 10}, {2, 5, 5}, {1, 5, 5}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_If,
                         SimpleIfNotConstConditionUnusedOutputPortsTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes_4),
                                            ::testing::ValuesIn(inTypes),
                                            ::testing::ValuesIn(conditions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         SimpleIfNotConstConditionUnusedOutputPortsTest::getTestCaseName);

TEST_P(SimpleIfNotConstConditionUnusedOutputPortsTest, CompareWithRefs) {
    run();
};

}  // namespace
