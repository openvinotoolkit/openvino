// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/concat_to_broadcast.hpp"

#include <gtest/gtest.h>

#include "openvino/core/node_vector.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;

TEST(TransformationTests, TestTransfromationExecuted) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto constant = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(param, constant, false);
    std::vector<ov::Output<ov::Node>> concat_input = {};

    // I'm using 2604 because this is the number of outputs there was in the targeted model
    for (int i = 0; i < 2604; ++i) {
        concat_input.push_back(reshape->get_default_output());
    }
    auto concat = std::make_shared<ov::op::v0::Concat>(concat_input, 0);
    auto result = std::make_shared<ov::op::v0::Result>(concat->get_default_output());

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConcatToBroadcast>();
    manager.run_passes(model);

    const auto& ops = model->get_ordered_ops();

    bool braodcast_present = std::any_of(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& node) {
        return std::dynamic_pointer_cast<ov::op::v3::Broadcast>(node) != nullptr;
    });

    EXPECT_TRUE(braodcast_present);
}