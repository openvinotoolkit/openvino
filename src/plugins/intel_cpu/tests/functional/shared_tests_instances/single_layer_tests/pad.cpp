// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/pad.hpp"

namespace {
using ov::test::PadLayerTest;
using ov::op::PadMode;

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::i32,
        ov::element::f16,
        ov::element::i16,
        ov::element::u16,
        ov::element::i8,
        ov::element::u8,
};

const std::vector<float> arg_pad_values = {0.f, 1.f, -1.f, 2.5f};

const std::vector<PadMode> pad_modes = {
        PadMode::EDGE,
        PadMode::REFLECT,
        PadMode::SYMMETRIC
};


// 1D

const std::vector<std::vector<int64_t>> pads_begin_1d = {{0}, {1}, {2}, {-2}};
const std::vector<std::vector<int64_t>> pads_end_1d   = {{0}, {1}, {2}, {-2}};

const std::vector<ov::Shape> input_shape_1d_static = {{5}};

const auto pad1DConstparams = testing::Combine(
        testing::ValuesIn(pads_begin_1d),
        testing::ValuesIn(pads_end_1d),
        testing::ValuesIn(arg_pad_values),
        testing::Values(PadMode::CONSTANT),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::static_shapes_to_test_representation(input_shape_1d_static)),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad1DConst,
        PadLayerTest,
        pad1DConstparams,
        PadLayerTest::getTestCaseName
);

const auto pad1Dparams = testing::Combine(
        testing::ValuesIn(pads_begin_1d),
        testing::ValuesIn(pads_end_1d),
        testing::Values(0),
        testing::ValuesIn(pad_modes),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::static_shapes_to_test_representation(input_shape_1d_static)),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad1D,
        PadLayerTest,
        pad1Dparams,
        PadLayerTest::getTestCaseName
);


// 2D

const std::vector<std::vector<int64_t>> pads_begin_2d = {{0, 0}, {1, 1}, {-2, 0}, {0, 3}};
const std::vector<std::vector<int64_t>> pads_end_2d   = {{0, 0}, {1, 1}, {0, 1}, {-3, -2}};

const std::vector<ov::Shape> input_shape_2d_static = {{13, 5}};

const auto pad2DConstparams = testing::Combine(
        testing::ValuesIn(pads_begin_2d),
        testing::ValuesIn(pads_end_2d),
        testing::ValuesIn(arg_pad_values),
        testing::Values(PadMode::CONSTANT),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::static_shapes_to_test_representation(input_shape_2d_static)),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad2DConst,
        PadLayerTest,
        pad2DConstparams,
        PadLayerTest::getTestCaseName
);

const auto pad2Dparams = testing::Combine(
        testing::ValuesIn(pads_begin_2d),
        testing::ValuesIn(pads_end_2d),
        testing::Values(0),
        testing::ValuesIn(pad_modes),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::static_shapes_to_test_representation(input_shape_2d_static)),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad2D,
        PadLayerTest,
        pad2Dparams,
        PadLayerTest::getTestCaseName
);


// 4D

const std::vector<std::vector<int64_t>> pads_begin_4d = {{0, 0, 0, 0}, {0, 3, 0, 0}, {0, 0, 0, 1}, {0, 0, -1, 1}, {2, 0, 0, 0}, {0, 3, 0, -1}};
const std::vector<std::vector<int64_t>> pads_end_4d   = {{0, 0, 0, 0}, {0, 3, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 2}, {1, -3, 0, 0}, {0, 3, 0, -1}};

const std::vector<ov::Shape> input_shape_4d_static = {{3, 5, 10, 11}};

const auto pad4DConstparams = testing::Combine(
        testing::ValuesIn(pads_begin_4d),
        testing::ValuesIn(pads_end_4d),
        testing::ValuesIn(arg_pad_values),
        testing::Values(PadMode::CONSTANT),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::static_shapes_to_test_representation(input_shape_4d_static)),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad4DConst,
        PadLayerTest,
        pad4DConstparams,
        PadLayerTest::getTestCaseName
);

const auto pad4Dparams = testing::Combine(
        testing::ValuesIn(pads_begin_4d),
        testing::ValuesIn(pads_end_4d),
        testing::Values(0),
        testing::ValuesIn(pad_modes),
        testing::ValuesIn(model_types),
        testing::Values(ov::test::static_shapes_to_test_representation(input_shape_4d_static)),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad4D,
        PadLayerTest,
        pad4Dparams,
        PadLayerTest::getTestCaseName
);
}  // namespace
