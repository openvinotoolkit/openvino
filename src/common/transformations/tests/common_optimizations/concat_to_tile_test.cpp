// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/concat_to_tile.hpp"

#include <gtest/gtest.h>

#include "openvino/core/node_vector.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;

enum class ExpectedType {
    Broadcast,
    Tile,
    Concat,
};

using ConcatToTileParams = std::tuple<ov::PartialShape, size_t, size_t, ExpectedType>;

class ConcatToTileTest : public WithParamInterface<ConcatToTileParams>, public testing::Test {
protected:
    void SetUp() override {
        std::tie(data_shape, concat_num_inputs, concat_axis, expected_type) = GetParam();
    }

    ov::PartialShape data_shape;
    size_t concat_num_inputs;
    ExpectedType expected_type;
    size_t concat_axis;
};

INSTANTIATE_TEST_SUITE_P(type_prop,
                         ConcatToTileTest,
                         Values(ConcatToTileParams({1, 2, 1, 2}, 2604, 0, ExpectedType::Broadcast),
                                ConcatToTileParams({1, 2, 1, 2}, 2604, 1, ExpectedType::Tile),
                                ConcatToTileParams({-1, 2, 1, 2}, 2604, 0, ExpectedType::Tile),
                                ConcatToTileParams({-1, -1, -1, -1}, 2604, 0, ExpectedType::Tile),
                                ConcatToTileParams(ov::PartialShape::dynamic(), 2604, 0, ExpectedType::Concat)));

TEST_P(ConcatToTileTest, TestTransfromationExecuted) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, data_shape);
    std::vector<ov::Output<ov::Node>> concat_inputs(concat_num_inputs, param->output(0));

    auto concat = std::make_shared<ov::op::v0::Concat>(concat_inputs, concat_axis);
    auto result = std::make_shared<ov::op::v0::Result>(concat->output(0));

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConcatToTile>();
    manager.run_passes(model);

    const auto& ops = model->get_ordered_ops();

    size_t tile_count = 0;
    size_t broadcast_count = 0;
    size_t concat_count = 0;

    for (auto& op : ops) {
        std::cout << op << std::endl;
        if (std::dynamic_pointer_cast<ov::op::v3::Broadcast>(op)) {
            ++broadcast_count;
        } else if (std::dynamic_pointer_cast<ov::op::v0::Tile>(op)) {
            ++tile_count;
        } else if (std::dynamic_pointer_cast<ov::op::v0::Concat>(op)) {
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