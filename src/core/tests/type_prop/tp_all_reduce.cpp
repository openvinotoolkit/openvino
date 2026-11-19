// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/tp_all_reduce.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"

namespace ov::test {

using op::internal::ReduceOp;
using op::internal::TPAllReduce;
using op::util::TPCollectiveBase;
using op::v0::Parameter;

TEST(type_prop, tp_all_reduce_keeps_shape_and_type) {
    const auto op = std::make_shared<TPAllReduce>(std::make_shared<Parameter>(element::f16, PartialShape{1, -1, 4096}),
                                                  1,
                                                  4,
                                                  0,
                                                  7,
                                                  ReduceOp::SUM);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, -1, 4096}));
    EXPECT_EQ(op->get_rank(), 1u);
    EXPECT_EQ(op->get_world_size(), 4u);
    EXPECT_EQ(op->get_group_id(), 0u);
    EXPECT_EQ(op->get_collective_id(), 7u);
    EXPECT_EQ(op->get_reduce_op(), ReduceOp::SUM);
}

TEST(type_prop, tp_all_reduce_dynamic_rank_and_type) {
    const auto op = std::make_shared<TPAllReduce>(std::make_shared<Parameter>(element::dynamic, PartialShape::dynamic()),
                                                  0,
                                                  2,
                                                  0,
                                                  0,
                                                  ReduceOp::SUM);
    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, tp_all_reduce_clone_keeps_attributes) {
    const auto input = std::make_shared<Parameter>(element::f16, PartialShape{2, 8});
    const auto op = std::make_shared<TPAllReduce>(input, 2, 3, 1, 42, ReduceOp::MAX);
    const auto clone = as_type_ptr<TPAllReduce>(op->clone_with_new_inputs({input}));
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->get_rank(), 2u);
    EXPECT_EQ(clone->get_world_size(), 3u);
    EXPECT_EQ(clone->get_group_id(), 1u);
    EXPECT_EQ(clone->get_collective_id(), 42u);
    EXPECT_EQ(clone->get_reduce_op(), ReduceOp::MAX);
}

TEST(type_prop, tp_all_reduce_is_a_tp_collective) {
    const auto input = std::make_shared<Parameter>(element::f16, PartialShape{2, 8});
    EXPECT_TRUE(is_type<TPCollectiveBase>(std::make_shared<TPAllReduce>(input, 0, 2, 0, 0, ReduceOp::SUM)));
    EXPECT_FALSE(is_type<TPCollectiveBase>(std::make_shared<op::v0::Relu>(input)));
}

TEST(type_prop, tp_all_reduce_rejects_rank_outside_group) {
    const auto input = std::make_shared<Parameter>(element::f16, PartialShape{2});
    EXPECT_THROW(std::make_shared<TPAllReduce>(input, 2, 2, 0, 0, ReduceOp::SUM), NodeValidationFailure);
}

}  // namespace ov::test
