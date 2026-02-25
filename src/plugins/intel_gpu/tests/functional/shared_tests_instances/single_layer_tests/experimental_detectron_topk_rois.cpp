// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/experimental_detectron_topkrois.hpp"

namespace {
using ov::test::ExperimentalDetectronTopKROIsLayerTest;

std::vector<int64_t> maxRois {
        1000,
        1500,
        2000
};

std::vector<ov::element::Type_t> elementTypes {
    ov::element::f16,
    ov::element::f32
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
                                 ::testing::ValuesIn(elementTypes),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ExperimentalDetectronTopKROIsLayerTest::getTestCaseName);

} // namespace
