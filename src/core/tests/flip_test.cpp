// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reverse.hpp"

using namespace ov;
using namespace ov::preprocess;

namespace {
std::shared_ptr<Model> create_simple_model(const PartialShape& shape) {
    auto param = std::make_shared<op::v0::Parameter>(element::f32, shape);
    auto result = std::make_shared<op::v0::Result>(param);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{param});
}
}  // namespace

TEST(pre_post_process, flip_horizontal_nhwc) {
    auto model = create_simple_model(PartialShape{1, 480, 640, 3});
    auto p = PrePostProcessor(model);

    // Set Layout (Required for Flip)
    p.input().tensor().set_layout("NHWC");

    // Apply Horizontal Flip
    p.input().preprocess().flip(FlipMode::HORIZONTAL);

    // Build Model
    model = p.build();

    // Verify Graph
    auto ops = model->get_ordered_ops();
    bool found_reverse = false;
    for (const auto& op : ops) {
        if (std::string(op->get_type_name()) == "Reverse") {
            found_reverse = true;
            // Verify axis is correct (Axis 2 = Width in NHWC)
            auto axis_const = as_type_ptr<op::v0::Constant>(op->get_input_node_shared_ptr(1));
            ASSERT_TRUE(axis_const);
            auto axis_val = axis_const->cast_vector<int32_t>()[0];
            EXPECT_EQ(axis_val, 2);
        }
    }
    EXPECT_TRUE(found_reverse) << "Horizontal Flip (Reverse op) was not found in the graph!";
}

TEST(pre_post_process, flip_vertical_nchw) {
    auto model = create_simple_model(PartialShape{1, 3, 480, 640});
    auto p = PrePostProcessor(model);

    // Set Layout
    p.input().tensor().set_layout("NCHW");

    // Apply Vertical Flip
    p.input().preprocess().flip(FlipMode::VERTICAL);

    // Build
    model = p.build();

    // Verify Graph
    auto ops = model->get_ordered_ops();
    bool found_reverse = false;
    for (const auto& op : ops) {
        if (std::string(op->get_type_name()) == "Reverse") {
            found_reverse = true;
            auto axis_const = as_type_ptr<op::v0::Constant>(op->get_input_node_shared_ptr(1));
            ASSERT_TRUE(axis_const);
            auto axis_val = axis_const->cast_vector<int32_t>()[0];
            EXPECT_EQ(axis_val, 2);
        }
    }
    EXPECT_TRUE(found_reverse) << "Vertical Flip (Reverse op) was not found in the graph!";
}

TEST(pre_post_process, flip_no_layout) {
    auto model = create_simple_model(PartialShape{1, 3, 480, 640});
    auto p = PrePostProcessor(model);

    // Apply Flip without setting layout (should throw)
    p.input().preprocess().flip(FlipMode::HORIZONTAL);

    EXPECT_THROW(p.build(), ov::AssertFailure);
}

TEST(pre_post_process, flip_horizontal_no_width) {
    auto model = create_simple_model(PartialShape{1, 3, 480, 640});
    auto p = PrePostProcessor(model);

    // Set layout without Width (e.g., NCH?)
    p.input().tensor().set_layout("NCH?");

    p.input().preprocess().flip(FlipMode::HORIZONTAL);

    // Expect AssertFailure because Width is missing, but don't check the specific message string
    // to avoid fragility if the helper function's error message changes.
    EXPECT_THROW(p.build(), ov::AssertFailure);
}

TEST(pre_post_process, flip_vertical_no_height) {
    auto model = create_simple_model(PartialShape{1, 3, 480, 640});
    auto p = PrePostProcessor(model);

    // Set layout without Height (e.g., NCW?)
    p.input().tensor().set_layout("NCW?");

    p.input().preprocess().flip(FlipMode::VERTICAL);

    // Expect AssertFailure because Height is missing
    EXPECT_THROW(p.build(), ov::AssertFailure);
}