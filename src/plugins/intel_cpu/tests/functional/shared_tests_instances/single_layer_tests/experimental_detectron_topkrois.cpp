// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_op_tests/experimental_detectron_topkrois.hpp"

namespace {
using ov::test::ExperimentalDetectronTopKROIsLayerTest;

std::vector<int64_t> maxRois {
        1000,
        1500,
        2000,
        2500
};

const std::vector<std::vector<ov::test::InputShape>> staticInputShape = {
        ov::test::static_shapes_to_test_representation({{3000, 4}, {3000}}),
        ov::test::static_shapes_to_test_representation({{4200, 4}, {4200}}),
        ov::test::static_shapes_to_test_representation({{4500, 4}, {4500}})
};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronTopKROIs_static, ExperimentalDetectronTopKROIsLayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShape),
                                 ::testing::ValuesIn(maxRois),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ExperimentalDetectronTopKROIsLayerTest::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynamicInputShape = {
        {
                {
                        {{-1, 4}, {{5000, 4}, {4000, 4}, {3500, 4}}},
                        {{-1}, {{5000}, {4000}, {3500}}}
                }
        },
        {
                {
                        {{{1000, 5000}, 4}, {{5000, 4}, {3000, 4}, {2500, 4}}},
                        {{{1000, 5000}}, {{5000}, {3000}, {2500}}}
                }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalROI_dynamic, ExperimentalDetectronTopKROIsLayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInputShape),
                                 ::testing::ValuesIn(maxRois),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ExperimentalDetectronTopKROIsLayerTest::getTestCaseName);
} // namespace
