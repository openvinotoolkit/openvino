// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/batch_norm.hpp"

namespace {
using ov::test::BatchNormLayerTest;
const std::vector<ov::element::Type> types = {
        ov::element::f16,
        ov::element::f32
};

const std::vector<double> epsilon = {
    1e-6,
    1e-5,
    1e-4
};

const std::vector<std::vector<ov::Shape>> inputShapes = {
        {{1, 3}},
        {{2, 5}},
        {{1, 3, 10}},
        {{1, 3, 1, 1}},
        {{2, 5, 4, 4}},
};

const auto batchNormParams = testing::Combine(
        testing::ValuesIn(epsilon),
        testing::ValuesIn(types),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
        testing::Values(ov::test::utils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_BatchNorm,
        BatchNormLayerTest,
        batchNormParams,
        BatchNormLayerTest::getTestCaseName
);

}  // namespace
