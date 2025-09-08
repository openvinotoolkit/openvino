// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

template <typename T, ov::element::Type_t ELEMENT_TYPE>
class ReduceOperatorType {
public:
    using op_type = T;
    static constexpr ov::element::Type_t element_type = ELEMENT_TYPE;
};

template <typename T>
class ReduceOperatorVisitor : public ::testing::Test {};

class ReduceOperatorTypeName {
public:
    template <typename T>
    static std::string GetName(int) {
        using OP_Type = typename T::op_type;
        constexpr ov::element::Type precision(T::element_type);
        const ov::Node::type_info_t typeinfo = OP_Type::get_type_info_static();
        return std::string{typeinfo.name} + "_" + precision.get_type_name();
    }
};

TYPED_TEST_SUITE_P(ReduceOperatorVisitor);

TYPED_TEST_P(ReduceOperatorVisitor, keep_dims_3D) {
    using OP_Type = typename TypeParam::op_type;

    Shape in_shape{3, 4, 5};
    const ov::element::Type_t in_et = TypeParam::element_type;

    Shape axes_shape{2};
    element::Type axes_et = element::i64;

    bool keep_dims = true;

    ov::test::NodeBuilder::opset().insert<OP_Type>();
    const auto data = make_shared<ov::op::v0::Parameter>(in_et, in_shape);
    const auto reduction_axes = make_shared<ov::op::v0::Parameter>(axes_et, axes_shape);
    const auto reduce_op = make_shared<OP_Type>(data, reduction_axes, keep_dims);

    NodeBuilder builder(reduce_op, {data, reduction_axes});
    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    auto g_reduce_op = as_type_ptr<OP_Type>(builder.create());
    EXPECT_EQ(g_reduce_op->get_keep_dims(), reduce_op->get_keep_dims());
}

TYPED_TEST_P(ReduceOperatorVisitor, do_not_keep_dims_3D) {
    using OP_Type = typename TypeParam::op_type;

    Shape in_shape{3, 4, 5};
    const ov::element::Type_t in_et = TypeParam::element_type;

    Shape axes_shape{2};
    element::Type axes_et = element::i64;

    bool keep_dims = false;

    ov::test::NodeBuilder::opset().insert<OP_Type>();
    const auto data = make_shared<ov::op::v0::Parameter>(in_et, in_shape);
    const auto reduction_axes = make_shared<ov::op::v0::Parameter>(axes_et, axes_shape);
    const auto reduce_op = make_shared<OP_Type>(data, reduction_axes, keep_dims);

    NodeBuilder builder(reduce_op, {data, reduction_axes});
    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    auto g_reduce_op = ov::as_type_ptr<OP_Type>(builder.create());
    EXPECT_EQ(g_reduce_op->get_keep_dims(), reduce_op->get_keep_dims());
}

REGISTER_TYPED_TEST_SUITE_P(ReduceOperatorVisitor, keep_dims_3D, do_not_keep_dims_3D);
