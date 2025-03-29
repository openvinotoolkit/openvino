// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/batch_norm.hpp"

namespace {
using ov::test::BatchNormLayerTest;

const std::vector<ov::element::Type> model_type = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<double> epsilon = {
    0.0,
    1e-6,
    1e-5,
    1e-4
};
const std::vector<std::vector<ov::Shape>> input_shapes_static = {
        {{1, 3}},
        {{2, 5}},
        {{1, 3, 10}},
        {{1, 3, 1, 1}},
        {{2, 5, 4, 4}},
};


const auto batch_norm_params = testing::Combine(
        testing::ValuesIn(epsilon),
        testing::ValuesIn(model_type),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_BatchNorm,
        BatchNormLayerTest,
        batch_norm_params,
        BatchNormLayerTest::getTestCaseName
);

}  // namespace
