// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/tp_gather.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"

namespace ov::test {

using op::internal::TPGather;
using op::util::TPCollectiveBase;
using op::v0::Parameter;

namespace {
std::shared_ptr<TPGather> make_gather(const PartialShape& shape, uint32_t rank, int64_t axis = -1) {
    return std::make_shared<TPGather>(std::make_shared<Parameter>(element::f16, shape), rank, 4, 0, 0, axis);
}
}  // namespace

TEST(type_prop, tp_gather_widens_axis_on_root_only) {
    EXPECT_EQ(make_gather({1, 1, 32000}, 0)->get_output_partial_shape(0), (PartialShape{1, 1, 128000}));
    EXPECT_EQ(make_gather({1, 1, 32000}, 3)->get_output_partial_shape(0), (PartialShape{1, 1, 32000}));
    EXPECT_EQ(make_gather({1, 1, 32000}, 0)->get_output_element_type(0), element::f16);
}

TEST(type_prop, tp_gather_dynamic_dimensions) {
    EXPECT_EQ(make_gather({1, 1, -1}, 0)->get_output_partial_shape(0), (PartialShape{1, 1, -1}));
    EXPECT_EQ(make_gather({1, 1, {2, 8}}, 0)->get_output_partial_shape(0), (PartialShape{1, 1, {8, 32}}));
    EXPECT_EQ(make_gather(PartialShape::dynamic(), 0)->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, tp_gather_negative_axis) {
    EXPECT_EQ(make_gather({2, 8}, 0, -2)->get_output_partial_shape(0), (PartialShape{8, 8}));
}

TEST(type_prop, tp_gather_clone_keeps_attributes) {
    const auto input = std::make_shared<Parameter>(element::f16, PartialShape{2, 8});
    const auto op = std::make_shared<TPGather>(input, 0, 2, 1, 43, 0);
    const auto clone = as_type_ptr<TPGather>(op->clone_with_new_inputs({input}));
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->get_rank(), 0u);
    EXPECT_EQ(clone->get_world_size(), 2u);
    EXPECT_EQ(clone->get_group_id(), 1u);
    EXPECT_EQ(clone->get_collective_id(), 43u);
    EXPECT_EQ(clone->get_axis(), 0);
    EXPECT_EQ(clone->get_output_partial_shape(0), (PartialShape{4, 8}));
}

TEST(type_prop, tp_gather_is_a_tp_collective) {
    EXPECT_TRUE(is_type<TPCollectiveBase>(make_gather({2, 8}, 0)));
}

TEST(type_prop, tp_gather_rejects_rank_outside_group) {
    EXPECT_THROW(make_gather({2}, 5), NodeValidationFailure);
}

TEST(type_prop, tp_gather_rejects_an_axis_its_input_does_not_have) {
    EXPECT_THROW(make_gather({2, 8}, 0, 2), NodeValidationFailure);
    EXPECT_THROW(make_gather({2, 8}, 0, -3), NodeValidationFailure);
    EXPECT_THROW(make_gather(PartialShape{}, 0), NodeValidationFailure);
}

}  // namespace ov::test
