// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/tp_collective.hpp"

#include <gtest/gtest.h>

#include <map>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"

using ov::op::internal::ReduceOp;
using ov::op::internal::TPAllReduce;
using ov::op::internal::TPCollectiveBase;
using ov::op::internal::TPGather;

namespace {
std::shared_ptr<ov::op::v0::Parameter> make_input(const ov::PartialShape& shape) {
    return std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape);
}

// The integer attributes an op visits, read out of one op and written into another, as the IR does.
class IntegerAttributes : public ov::AttributeVisitor {
public:
    explicit IntegerAttributes(bool write) : m_write(write) {}

    void on_adapter(const std::string&, ov::ValueAccessor<void>&) override {}
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        if (m_write)
            adapter.set(values.at(name));
        else
            values[name] = adapter.get();
    }

    std::map<std::string, int64_t> values;

private:
    bool m_write;
};
}  // namespace

TEST(TPCollectiveTest, AllReduceKeepsShapeAndType) {
    const auto op = std::make_shared<TPAllReduce>(make_input({1, -1, 4096}), 1, 4, 0, 7, ReduceOp::SUM);
    EXPECT_EQ(op->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{1, -1, 4096}));
    EXPECT_EQ(op->get_rank(), 1);
    EXPECT_EQ(op->get_world_size(), 4);
    EXPECT_EQ(op->get_group_id(), 0);
    EXPECT_EQ(op->get_collective_id(), 7u);
    EXPECT_EQ(op->get_reduce_op(), ReduceOp::SUM);
}

TEST(TPCollectiveTest, GatherWidensAxisOnRootOnly) {
    const auto root = std::make_shared<TPGather>(make_input({1, 1, 32000}), 0, 4, 0, 0);
    EXPECT_EQ(root->get_output_partial_shape(0), (ov::PartialShape{1, 1, 128000}));

    const auto other = std::make_shared<TPGather>(make_input({1, 1, 32000}), 3, 4, 0, 0);
    EXPECT_EQ(other->get_output_partial_shape(0), (ov::PartialShape{1, 1, 32000}));

    const auto dynamic = std::make_shared<TPGather>(make_input({1, 1, -1}), 0, 4, 0, 0);
    EXPECT_EQ(dynamic->get_output_partial_shape(0), (ov::PartialShape{1, 1, -1}));
}

TEST(TPCollectiveTest, CloneKeepsAttributes) {
    const auto input = make_input({2, 8});
    const auto reduce = std::make_shared<TPAllReduce>(input, 2, 3, 1, 42, ReduceOp::MAX);
    const auto reduce_clone = ov::as_type_ptr<TPAllReduce>(reduce->clone_with_new_inputs({input}));
    ASSERT_NE(reduce_clone, nullptr);
    EXPECT_EQ(reduce_clone->get_rank(), 2);
    EXPECT_EQ(reduce_clone->get_world_size(), 3);
    EXPECT_EQ(reduce_clone->get_group_id(), 1);
    EXPECT_EQ(reduce_clone->get_collective_id(), 42u);
    EXPECT_EQ(reduce_clone->get_reduce_op(), ReduceOp::MAX);

    const auto gather = std::make_shared<TPGather>(input, 0, 2, 0, 43, 0);
    const auto gather_clone = ov::as_type_ptr<TPGather>(gather->clone_with_new_inputs({input}));
    ASSERT_NE(gather_clone, nullptr);
    EXPECT_EQ(gather_clone->get_collective_id(), 43u);
    EXPECT_EQ(gather_clone->get_axis(), 0);
    EXPECT_EQ(gather_clone->get_output_partial_shape(0), (ov::PartialShape{4, 8}));
}

// An orchestrator may hand out any 64-bit value, a hash for instance; the whole of it survives the IR.
TEST(TPCollectiveTest, CollectiveIdSurvivesAttributeVisiting) {
    constexpr uint64_t id = 0xFEDCBA9876543210ULL;
    const auto input = make_input({2, 8});
    const auto reduce = std::make_shared<TPAllReduce>(input, 1, 2, 0, id, ReduceOp::SUM);

    IntegerAttributes read(false);
    reduce->visit_attributes(read);
    ASSERT_EQ(read.values.count("collective_id"), 1u);

    IntegerAttributes write(true);
    write.values = read.values;
    TPAllReduce restored;
    restored.visit_attributes(write);
    EXPECT_EQ(restored.get_collective_id(), id);
    EXPECT_EQ(restored.get_rank(), 1);
}

TEST(TPCollectiveTest, BaseTypeMatchesCollectivesOnly) {
    const auto input = make_input({2, 8});
    EXPECT_TRUE(ov::is_type<TPCollectiveBase>(std::make_shared<TPAllReduce>(input, 0, 2, 0, 0, ReduceOp::SUM)));
    EXPECT_TRUE(ov::is_type<TPCollectiveBase>(std::make_shared<TPGather>(input, 0, 2, 0, 1)));
    EXPECT_FALSE(ov::is_type<TPCollectiveBase>(std::make_shared<ov::op::v0::Relu>(input)));
}

TEST(TPCollectiveTest, RejectsRankOutsideGroup) {
    EXPECT_THROW(std::make_shared<TPAllReduce>(make_input({2}), 2, 2, 0, 0, ReduceOp::SUM), ov::NodeValidationFailure);
    EXPECT_THROW(std::make_shared<TPGather>(make_input({2}), -1, 2, 0, 0), ov::NodeValidationFailure);
}

TEST(TPCollectiveTest, ReduceOpNames) {
    EXPECT_EQ(ov::as_string(ReduceOp::SUM), "sum");
    EXPECT_EQ(ov::as_enum<ReduceOp>("max"), ReduceOp::MAX);
}
