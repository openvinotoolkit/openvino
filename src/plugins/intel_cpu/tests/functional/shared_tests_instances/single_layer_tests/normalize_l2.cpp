// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/normalize_l2.hpp"

namespace {
using ov::test::NormalizeL2LayerTest;

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<float> eps = {1e-12f, 1e-6f, 1e-3f, 0.1f, 100};

const std::vector<ov::op::EpsMode> eps_modes = {
        ov::op::EpsMode::ADD,
        ov::op::EpsMode::MAX,
};

/* ============= 1D ============= */
// [SKIPPED][CPU] Unsupported rank, Issue: 35627
const std::vector<std::vector<int64_t>> axes_1d = {
        {},
        {0}
};

std::vector<ov::Shape> input_shape_1d_static = {{5}};

const auto normL2params_1D = testing::Combine(
        testing::ValuesIn(axes_1d),
        testing::ValuesIn(eps),
        testing::ValuesIn(eps_modes),
        testing::Values(ov::test::static_shapes_to_test_representation(input_shape_1d_static)),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_NormalizeL2_1D,
        NormalizeL2LayerTest,
        normL2params_1D,
        NormalizeL2LayerTest::getTestCaseName
);

/* ============= 2D ============= */
const std::vector<std::vector<int64_t>> axes_2D = {
        {},
        {1},

        // [CPU] Unsupported axes, Issue: 59791
        // {0},
        // {0, 1},
};

std::vector<ov::Shape> input_shape_2d_static = {{5, 3}};

const auto normL2params_2D = testing::Combine(
        testing::ValuesIn(axes_2D),
        testing::ValuesIn(eps),
        testing::ValuesIn(eps_modes),
        testing::Values(ov::test::static_shapes_to_test_representation(input_shape_2d_static)),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_NormalizeL2_2D,
        NormalizeL2LayerTest,
        normL2params_2D,
        NormalizeL2LayerTest::getTestCaseName
);

/* ============= 3D ============= */
const std::vector<std::vector<int64_t>> axes_3D = {
        {},
        {1},
        {1, 2},
        {2, 1},

        // [CPU] Unsupported axes, Issue: 59791
        // {0},
        // {2},
        // {0, 1},
        // {0, 1, 2}
};

std::vector<ov::Shape> input_shape_3d_static = {{2, 5, 3}};

const auto normL2params_3D = testing::Combine(
        testing::ValuesIn(axes_3D),
        testing::ValuesIn(eps),
        testing::ValuesIn(eps_modes),
        testing::Values(ov::test::static_shapes_to_test_representation(input_shape_3d_static)),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_NormalizeL2_3D,
        NormalizeL2LayerTest,
        normL2params_3D,
        NormalizeL2LayerTest::getTestCaseName
);

/* ============= 4D ============= */
const std::vector<std::vector<int64_t>> axes_4D = {
        {},
        {1},
        {1, 2, 3},
        {3, 1, 2},

        // [CPU] Unsupported axes, Issue: 59791
        // {0},
        // {2},
        // {3},
        // {0, 1},
        // {1, 2},
        // {2, 3},
        // {0, 1, 2, 3}
};

std::vector<ov::Shape> input_shape_4d_static = {{2, 3, 10, 5}};

const auto normL2params_4D = testing::Combine(
        testing::ValuesIn(axes_4D),
        testing::ValuesIn(eps),
        testing::ValuesIn(eps_modes),
        testing::Values(ov::test::static_shapes_to_test_representation(input_shape_4d_static)),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_NormalizeL2_4D,
        NormalizeL2LayerTest,
        normL2params_4D,
        NormalizeL2LayerTest::getTestCaseName
);

/* ============= 5D ============= */
// [SKIPPED][CPU] Unsupported rank, Issue: 35627
const std::vector<std::vector<int64_t>> axes_5D = {
        {},
        {0},
        {1},
        {2},
        {3},
        {4},
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 4},
        {1, 2, 3},
        {2, 3, 4},
        {4, 3, 2},
        {1, 2, 3, 4},
        {0, 1, 2, 3}
};

std::vector<ov::Shape> input_shape_5d_static = {{2, 2, 3, 10, 5}};

const auto normL2params_5D = testing::Combine(
        testing::ValuesIn(axes_5D),
        testing::ValuesIn(eps),
        testing::ValuesIn(eps_modes),
        testing::Values(ov::test::static_shapes_to_test_representation(input_shape_5d_static)),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_NormalizeL2_5D,
        NormalizeL2LayerTest,
        normL2params_5D,
        NormalizeL2LayerTest::getTestCaseName
);
}  // namespace
