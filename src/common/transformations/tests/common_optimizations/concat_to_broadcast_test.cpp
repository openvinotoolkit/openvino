// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/concat_to_broadcast.hpp"

#include <gtest/gtest.h>

#include "openvino/core/node_vector.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;

enum class ExpectedType {
    Broadcast,
    Tile,
    Concat,
};

using ConcatToBroadcastParams = std::tuple<ov::PartialShape, size_t, size_t, ExpectedType>;

class ConcatToBroadcastTest : public WithParamInterface<ConcatToBroadcastParams>, public testing::Test {
protected:
    void SetUp() override {
        std::tie(data_shape, concat_num_inputs, concat_axis, expected_type) = GetParam();
    }

    ov::PartialShape data_shape;
    size_t concat_num_inputs;
    ExpectedType expected_type;
    size_t concat_axis;
};

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    ConcatToBroadcastTest,
    Values(ConcatToBroadcastParams({2, 1}, 10, 1, ExpectedType::Broadcast),
           ConcatToBroadcastParams({1, 2, 1, 2}, 2604, 0, ExpectedType::Broadcast),
           ConcatToBroadcastParams(ov::PartialShape::dynamic(), 2604, 0, ExpectedType::Concat),
           // Common case (converting to Tile) causes an issue in e2e test with unknown root cause (ticket: 142246)
           // The following tests cover Tile cases, but temporary Concat remains in the graph.
           ConcatToBroadcastParams({1, 2, 1, 2}, 2604, 1, ExpectedType::Concat),
           ConcatToBroadcastParams({-1, 2, 1, 2}, 2604, 0, ExpectedType::Concat),
           ConcatToBroadcastParams({-1, -1, -1, -1}, 2604, 0, ExpectedType::Concat)));

TEST_P(ConcatToBroadcastTest, TestTransfromationExecuted) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, data_shape);
    std::vector<ov::Output<ov::Node>> concat_inputs(concat_num_inputs, param->output(0));

    auto concat = std::make_shared<ov::op::v0::Concat>(concat_inputs, concat_axis);
    auto result = std::make_shared<ov::op::v0::Result>(concat->output(0));

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConcatToBroadcast>();
    manager.run_passes(model);

    const auto& ops = model->get_ordered_ops();

    size_t tile_count = 0;
    size_t broadcast_count = 0;
    size_t concat_count = 0;

    for (auto& op : ops) {
        std::cout << op << std::endl;
        if (ov::as_type_ptr<ov::op::v3::Broadcast>(op)) {
            ++broadcast_count;
        } else if (ov::as_type_ptr<ov::op::v0::Tile>(op)) {
            ++tile_count;
        } else if (ov::as_type_ptr<ov::op::v0::Concat>(op)) {
            ++concat_count;
        }
    }

    if (expected_type == ExpectedType::Broadcast) {
        ASSERT_EQ(broadcast_count, 1);
        ASSERT_EQ(tile_count, 0);
        ASSERT_EQ(concat_count, 0);
    } else if (expected_type == ExpectedType::Tile) {
        ASSERT_EQ(broadcast_count, 0);
        ASSERT_EQ(tile_count, 1);
        ASSERT_EQ(concat_count, 0);
    } else if (expected_type == ExpectedType::Concat) {
        ASSERT_EQ(broadcast_count, 0);
        ASSERT_EQ(tile_count, 0);
        ASSERT_EQ(concat_count, 1);
    }
}