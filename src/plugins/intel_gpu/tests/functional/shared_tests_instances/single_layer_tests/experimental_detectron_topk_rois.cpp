// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/experimental_detectron_topkrois.hpp"

using namespace ov::test;
using namespace ov::test::subgraph;

namespace {
std::vector<int64_t> maxRois {
        1000,
        1500,
        2000
};

std::vector<ElementType> elementTypes {
    ElementType::f16,
    ElementType::f32
};

const std::vector<std::vector<InputShape>> staticInputShape = {
        static_shapes_to_test_representation({{3000, 4}, {3000}}),
        static_shapes_to_test_representation({{4200, 4}, {4200}}),
        static_shapes_to_test_representation({{4500, 4}, {4500}})
};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronTopKROIs_static, ExperimentalDetectronTopKROIsLayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShape),
                                 ::testing::ValuesIn(maxRois),
                                 ::testing::ValuesIn(elementTypes),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ExperimentalDetectronTopKROIsLayerTest::getTestCaseName);

} // namespace
