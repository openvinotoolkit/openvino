// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/simple_if.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
std::vector<std::vector<ov::test::InputShape>> inputShapes = {
        {
            {{}, {{5, 7}}},
            {{}, {{5, 7}}},
        },
        {
            {{}, {{30, 20, 10}}},
            {{}, {{30, 20, 10}}}
        },
        {
            {{-1, -1, -1}, {{10, 20, 5}, {10, 20, 5}, {1, 5, 5}}},
            {{-1, -1, -1}, {{10, 20, 5}, {10, 20, 5}, {1, 1, 5}}}
        },
        {
            {{-1, 5, -1}, {{10, 5, 10}, {2, 5, 5}, {1, 5, 5}}},
            {{-1, 5, -1}, {{1, 5, 1}, {2, 5, 5}, {5, 5, 5}}}
        },
        {
            {{{1, 10}, {1, 10}, {1, 10}}, {{2, 5, 10}, {2, 5, 1}, {1, 5, 5}}},
            {{{1, 10}, {1, 10}, {1, 10}}, {{2, 5, 10}, {2, 1, 5}, {5, 5, 5}}}
        },
};

const std::vector<ov::test::ElementType> inTypes = {
        ov::test::ElementType::f32,
        ov::test::ElementType::bf16,
        ov::test::ElementType::i8
};

std::vector<bool> conditions = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_If, SimpleIfTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes),
                                ::testing::ValuesIn(inTypes),
                                ::testing::ValuesIn(conditions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        SimpleIfTest::getTestCaseName);

TEST_P(SimpleIfTest, CompareWithRefs) {
    run();
};

INSTANTIATE_TEST_SUITE_P(smoke_If, SimpleIf2OutTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes),
                                ::testing::ValuesIn(inTypes),
                                ::testing::ValuesIn(conditions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        SimpleIf2OutTest::getTestCaseName);

TEST_P(SimpleIf2OutTest, CompareWithRefs) {
    run();
};

INSTANTIATE_TEST_SUITE_P(smoke_If, SimpleIfNotConstConditionTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes),
                                ::testing::ValuesIn(inTypes),
                                ::testing::ValuesIn(conditions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
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
            {{{1, 10}, {1, 10}, {1, 10}}, {{2, 5, 10}, {2, 5, 1}, {1, 5, 5}}},
        },
};

const std::vector<ov::test::ElementType> inTypes_2 = {
        ov::test::ElementType::f32,
        ov::test::ElementType::bf16
};

INSTANTIATE_TEST_SUITE_P(smoke_If, SimpleIfNotConstConditionAndInternalDynamismTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_2),
                                 ::testing::ValuesIn(inTypes_2),
                                 ::testing::ValuesIn(conditions),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         SimpleIfNotConstConditionTest::getTestCaseName);

TEST_P(SimpleIfNotConstConditionAndInternalDynamismTest, CompareWithRefs) {
    run();
};

}  // namespace
