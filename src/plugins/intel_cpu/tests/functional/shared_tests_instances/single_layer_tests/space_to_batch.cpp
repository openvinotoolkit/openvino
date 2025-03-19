// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/space_to_batch.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::SpaceToBatchLayerTest;

namespace {

const std::vector<std::vector<int64_t>> block_shapes_4D {
        {1, 1, 2, 2}
};
const std::vector<std::vector<int64_t>> pads_begins_4D {
        {0, 0, 0, 0}, {0, 0, 0, 2}
};
const std::vector<std::vector<int64_t>> pads_ends_4D {
        {0, 0, 0, 0}, {0, 0, 0, 2}
};
const auto data_shapes_4D = ov::test::static_shapes_to_test_representation(
        std::vector<std::vector<ov::Shape>>{
        {{1, 1, 2, 2}}, {{1, 3, 2, 2}}, {{1, 1, 4, 4}}, {{2, 1, 2, 4}}
});

const auto space_to_batch_4D = ::testing::Combine(
        ::testing::ValuesIn(block_shapes_4D),
        ::testing::ValuesIn(pads_begins_4D),
        ::testing::ValuesIn(pads_ends_4D),
        ::testing::ValuesIn(data_shapes_4D),
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_spacetobatch4D, SpaceToBatchLayerTest, space_to_batch_4D,
        SpaceToBatchLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> block_shapes_5D {
        {1, 1, 3, 2, 2}
};
const std::vector<std::vector<int64_t>> pads_begins_5D {
        {0, 0, 1, 0, 3}
};
const std::vector<std::vector<int64_t>> pads_ends_5D {
        {0, 0, 2, 0, 0}
};
const auto data_shapes_5D = ov::test::static_shapes_to_test_representation(
        std::vector<std::vector<ov::Shape>>{
        {{1, 1, 3, 2, 1}}
});

const auto space_to_batch_5D = ::testing::Combine(
        ::testing::ValuesIn(block_shapes_5D),
        ::testing::ValuesIn(pads_begins_5D),
        ::testing::ValuesIn(pads_ends_5D),
        ::testing::ValuesIn(data_shapes_5D),
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_spacetobatch5D, SpaceToBatchLayerTest, space_to_batch_5D,
        SpaceToBatchLayerTest::getTestCaseName);

}  // namespace
