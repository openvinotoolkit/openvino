// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/fake_quantize.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::FakeQuantizeLayerTest;

const ov::op::AutoBroadcastSpec numpy_broadcast = ov::op::AutoBroadcastType::NUMPY;

const ov::op::AutoBroadcastSpec none_broadcast = ov::op::AutoBroadcastType::NONE;

const std::vector<ov::op::AutoBroadcastSpec> broadcasts = {
    {ov::op::AutoBroadcastType::NUMPY},
    {ov::op::AutoBroadcastType::PDPD, -1},
};

const std::vector<ov::element::Type> model_types =
     {ov::element::f32,
      ov::element::f16};

const std::vector<std::vector<ov::Shape>> shapes_static =
     {{{1, 1}},
      {{2, 6}},
      {{1, 1, 1}},
      {{2, 6, 13}},
      {{1, 1, 1, 1}},
      {{3, 10, 5, 6}},
      {{2, 8, 5, 18}},
      {{2, 16, 3, 18}},
      {{3, 49, 5, 6}},
      {{1, 1, 1, 1, 1}},
      {{3, 10, 2, 5, 6}},
      {{2, 8, 1, 5, 18}},
      {{2, 16, 4, 3, 18}},
      {{3, 49, 7, 5, 6}}};
const std::vector<std::vector<size_t>> const_shapes = {{1}};
const std::vector<size_t> levels = {16, 255, 256};

const std::vector<float> fq_args = {};

const auto fq_params = ::testing::Combine(
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(const_shapes),
        ::testing::Values(fq_args),
        ::testing::ValuesIn(broadcasts)
);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize, FakeQuantizeLayerTest,
                        ::testing::Combine(
                                fq_params,
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_static)),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        FakeQuantizeLayerTest::getTestCaseName);


const std::vector<size_t> single_shape = {3, 4, 2, 5};
const auto none_broadcast_fq_params = ::testing::Combine(
        ::testing::ValuesIn(levels),
        ::testing::Values(single_shape),
        ::testing::Values(fq_args),
        ::testing::Values(none_broadcast)
);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizenone_broadcast, FakeQuantizeLayerTest,
                        ::testing::Combine(
                                none_broadcast_fq_params,
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::static_shapes_to_test_representation({ov::Shape(single_shape)})),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        FakeQuantizeLayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> shapes_static_per_channel = {{{11, 10, 22, 19}}, {{11, 10, 5, 6}}};
const std::vector<std::vector<size_t>> const_shapes_per_channel_axis0 = {{11, 1, 1, 1}};
const std::vector<std::vector<size_t>> const_shapes_per_channel_axis1 = {{1, 10, 1, 1}, {10, 1, 1}};

const auto fq_params_per_channel_axis0 = ::testing::Combine(
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(const_shapes_per_channel_axis0),
        ::testing::Values(fq_args),
        ::testing::Values(numpy_broadcast)
);

const auto fq_params_per_channel_axis1 = ::testing::Combine(
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(const_shapes_per_channel_axis1),
        ::testing::Values(fq_args),
        ::testing::Values(numpy_broadcast)
);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizePerChannelAxis0, FakeQuantizeLayerTest,
                        ::testing::Combine(
                                fq_params_per_channel_axis0,
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_static_per_channel)),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        FakeQuantizeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizePerChannelAxis1, FakeQuantizeLayerTest,
                        ::testing::Combine(
                                fq_params_per_channel_axis1,
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_static_per_channel)),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        FakeQuantizeLayerTest::getTestCaseName);

const std::vector<ov::Shape> shapes_static_per_channel_2d = {{1, 10}};
const std::vector<std::vector<size_t>> const_shapes_per_channel_2d = { {10}, {1, 10}, {1} };
const auto fq_params_const_shapes_per_channel_2d = ::testing::Combine(
    ::testing::ValuesIn(levels),
    ::testing::ValuesIn(const_shapes_per_channel_2d),
    ::testing::Values(fq_args),
    ::testing::Values(numpy_broadcast)
);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizePerChannel2D, FakeQuantizeLayerTest,
    ::testing::Combine(
        fq_params_const_shapes_per_channel_2d,
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_static_per_channel_2d)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    FakeQuantizeLayerTest::getTestCaseName);

}  // namespace
