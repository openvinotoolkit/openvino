// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/sdpa_to_paged_attention/total_sequence_length_pattern.hpp"

using namespace ov;

TEST(SDPATOPATest, SDPANotPresent) {
    const auto p0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 32, 32});
    const auto p1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 32, 32});
    const auto add = std::make_shared<op::v1::Add>(p0, p1);
    const auto result = std::make_shared<op::v0::Result>(add);

    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{p0, p1});

    ov::pass::Manager manager;
    manager.register_pass<pass::SDPAToPagedAttention>();
    EXPECT_THROW(manager.run_passes(model), ov::Exception);
}

TEST(SDPATOPATest, GatherIdx_ConcatAxis_EQ) {
    // Almost replicating the pattern from the TotalSequenceLengthPattern transformation.
    const int CONCAT_AXIS = 1;
    const int GATHER_IDX = 1;

    const auto input = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{PartialShape::dynamic(), element::i32, "variable"});
    const auto read_value = std::make_shared<op::v6::ReadValue>(input, variable);

    const auto beam_idx = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    const auto gather_axis = op::v0::Constant::create(element::i64, Shape{}, {0});
    const auto gather = std::make_shared<op::v8::Gather>(read_value, beam_idx, gather_axis);

    const auto concat_input = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 2, 3});
    const auto concat = std::make_shared<op::v0::Concat>(NodeVector{gather, concat_input}, CONCAT_AXIS);

    const auto shape_of = std::make_shared<op::v3::ShapeOf>(concat, element::i64);

    const auto gather_indices = op::v0::Constant::create(element::i64, Shape{}, {GATHER_IDX});
    const auto gather_axis2 = op::v0::Constant::create(element::i64, Shape{}, {0});
    const auto gather1 = std::make_shared<op::v8::Gather>(shape_of, gather_indices, gather_axis2);

    const auto result = std::make_shared<op::v0::Result>(gather1);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input, beam_idx, concat_input});

    const auto max_context_len = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::TotalSequenceLengthPattern>(max_context_len);
    bool transformation_run = manager.run_passes(model);

    EXPECT_TRUE(transformation_run);
    const auto new_convert =
        ov::as_type_ptr<op::v0::Convert>(result->input(0).get_source_output().get_node_shared_ptr());
    EXPECT_TRUE(new_convert);
    const auto new_max_context_len =
        ov::as_type_ptr<op::v0::Parameter>(new_convert->input(0).get_source_output().get_node_shared_ptr());
    EXPECT_TRUE(new_max_context_len);
    EXPECT_TRUE(new_max_context_len == max_context_len);
}

TEST(SDPATOPATest, GatherIdx_ConcatAxis_NOTEQ_STATIC) {
    // Almost replicating the pattern from the TotalSequenceLengthPattern transformation.
    const int CONCAT_AXIS = 1;
    const int GATHER_IDX = 0;

    const auto input = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{PartialShape::dynamic(), element::i32, "variable"});
    const auto read_value = std::make_shared<op::v6::ReadValue>(input, variable);

    const auto beam_idx = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    const auto gather_axis = op::v0::Constant::create(element::i64, Shape{}, {0});
    const auto gather = std::make_shared<op::v8::Gather>(read_value, beam_idx, gather_axis);

    const auto concat_input = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 2, 3});
    const auto concat = std::make_shared<op::v0::Concat>(NodeVector{gather, concat_input}, CONCAT_AXIS);

    const auto shape_of = std::make_shared<op::v3::ShapeOf>(concat, element::i64);

    const auto gather_indices = op::v0::Constant::create(element::i64, Shape{}, {GATHER_IDX});
    const auto gather_axis2 = op::v0::Constant::create(element::i64, Shape{}, {0});
    const auto gather1 = std::make_shared<op::v8::Gather>(shape_of, gather_indices, gather_axis2);

    const auto result = std::make_shared<op::v0::Result>(gather1);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input, beam_idx, concat_input});

    const auto max_context_len = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::TotalSequenceLengthPattern>(max_context_len);
    bool transformation_run = manager.run_passes(model);

    EXPECT_TRUE(transformation_run);
    const auto new_constant =
        ov::as_type_ptr<op::v0::Constant>(result->input(0).get_source_output().get_node_shared_ptr());
    EXPECT_TRUE(new_constant);
}

TEST(SDPATOPATest, GatherIdx_ConcatAxis_NOTEQ_DYNAMIC) {
    // Almost replicating the pattern from the TotalSequenceLengthPattern transformation.
    const int CONCAT_AXIS = 1;
    const int GATHER_IDX = 0;

    const auto input = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{PartialShape::dynamic(), element::i32, "variable"});
    const auto read_value = std::make_shared<op::v6::ReadValue>(input, variable);

    const auto beam_idx = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    const auto gather_axis = op::v0::Constant::create(element::i64, Shape{}, {0});
    const auto gather = std::make_shared<op::v8::Gather>(read_value, beam_idx, gather_axis);

    const auto concat_input =
        std::make_shared<op::v0::Parameter>(element::i32,
                                            PartialShape{Dimension(1, 2), Dimension(1, 3), Dimension(1, 4)});
    const auto concat = std::make_shared<op::v0::Concat>(NodeVector{gather, concat_input}, CONCAT_AXIS);

    const auto shape_of = std::make_shared<op::v3::ShapeOf>(concat, element::i64);

    const auto gather_indices = op::v0::Constant::create(element::i64, Shape{}, {GATHER_IDX});
    const auto gather_axis2 = op::v0::Constant::create(element::i64, Shape{}, {0});
    const auto gather1 = std::make_shared<op::v8::Gather>(shape_of, gather_indices, gather_axis2);

    const auto result = std::make_shared<op::v0::Result>(gather1);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input, beam_idx, concat_input});

    const auto max_context_len = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::TotalSequenceLengthPattern>(max_context_len);
    EXPECT_THROW(manager.run_passes(model), ov::Exception);
}