// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/util/attr_types.hpp"

using namespace ov;
using op::v0::Parameter;
using namespace testing;

template <class TOp>
class BitwiseOperator : public TypePropOpTest<TOp> {};

TYPED_TEST_SUITE_P(BitwiseOperator);

TYPED_TEST_P(BitwiseOperator, default_constructor_integer) {
    auto lhs = std::make_shared<Parameter>(element::i32, PartialShape{-1, 4, 1, 6, {1, 6}, {2, 6}});
    auto rhs = std::make_shared<Parameter>(element::i32, PartialShape{-1, 1, 5, 6, {5, 8}, {5, 8}});

    const auto op = this->make_op();

    op->set_argument(0, lhs);
    op->set_argument(1, rhs);

    auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NONE);
    op->set_autob(autob);
    EXPECT_EQ(op->get_autob(), op::AutoBroadcastType::NONE);
    ASSERT_THROW(op->validate_and_infer_types(), NodeValidationFailure);

    autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY);
    op->set_autob(autob);
    EXPECT_EQ(op->get_autob(), op::AutoBroadcastType::NUMPY);

    op->validate_and_infer_types();

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 4, 5, 6, {5, 8}, {5, 6}}));
}

TYPED_TEST_P(BitwiseOperator, default_constructor_boolean) {
    auto lhs = std::make_shared<Parameter>(element::boolean, PartialShape{-1, 4, 1, 6, {1, 6}, {2, 6}});
    auto rhs = std::make_shared<Parameter>(element::boolean, PartialShape{-1, 1, 5, 6, {5, 8}, {5, 8}});

    const auto op = this->make_op();

    op->set_argument(0, lhs);
    op->set_argument(1, rhs);

    auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NONE);
    op->set_autob(autob);
    EXPECT_EQ(op->get_autob(), op::AutoBroadcastType::NONE);
    ASSERT_THROW(op->validate_and_infer_types(), NodeValidationFailure);

    autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY);
    op->set_autob(autob);
    EXPECT_EQ(op->get_autob(), op::AutoBroadcastType::NUMPY);

    op->validate_and_infer_types();

    EXPECT_EQ(op->get_element_type(), element::boolean);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 4, 5, 6, {5, 8}, {5, 6}}));
}

TYPED_TEST_P(BitwiseOperator, shape_inference_2D) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 2});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{2, 2});

    const auto op = this->make_op(lhs, rhs);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 2}));
}

TYPED_TEST_P(BitwiseOperator, shape_inference_4D) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 2, 3, 3});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{2, 2, 3, 3});

    const auto op = this->make_op(lhs, rhs);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 2, 3, 3}));
}

TYPED_TEST_P(BitwiseOperator, default_autobroadcast) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 2});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{2, 2});

    const auto op = this->make_op(lhs, rhs);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 2}));
    EXPECT_EQ(op->get_autob(), op::AutoBroadcastType::NUMPY);
}

TYPED_TEST_P(BitwiseOperator, no_autobroadcast) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 2});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{2, 2});

    const auto op = this->make_op(lhs, rhs, op::AutoBroadcastType::NONE);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 2}));
    EXPECT_EQ(op->get_autob(), op::AutoBroadcastType::NONE);
}

TYPED_TEST_P(BitwiseOperator, shape_inference_4D_x_scalar_numpy_broadcast) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{1});

    const auto op = this->make_op(lhs, rhs);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(BitwiseOperator, shape_inference_4D_x_1D_numpy_broadcast) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{5});

    const auto op = this->make_op(lhs, rhs);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(BitwiseOperator, shape_inference_2D_x_4D_numpy_broadcast) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{4, 5});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{2, 3, 4, 5});

    const auto op = this->make_op(lhs, rhs);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(BitwiseOperator, shape_inference_3D_x_4D_numpy_broadcast) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{1, 4, 5});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{2, 3, 1, 1});

    const auto op = this->make_op(lhs, rhs);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(BitwiseOperator, shape_inference_4D_x_3D_numpy_broadcast) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{8, 1, 6, 1});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{7, 1, 5});

    const auto op = this->make_op(lhs, rhs);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_shape(), (Shape{8, 7, 6, 5}));
    EXPECT_EQ(op->get_autob(), op::AutoBroadcastType::NUMPY);
}

TYPED_TEST_P(BitwiseOperator, static_shape_pdpd_doc_examples) {
    {
        auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 3, 4, 5});
        auto rhs = std::make_shared<Parameter>(element::i32, Shape{3, 4});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1);
        const auto op = this->make_op(lhs, rhs, autob);

        EXPECT_EQ(op->get_element_type(), element::i32);
        EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
    }
    {
        auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 3, 4, 5});
        auto rhs = std::make_shared<Parameter>(element::i32, Shape{3, 1});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1);
        const auto op = this->make_op(lhs, rhs, autob);

        EXPECT_EQ(op->get_element_type(), element::i32);
        EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
    }
    {
        auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 3, 4, 5});
        auto rhs = std::make_shared<Parameter>(element::i32, Shape{});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD);
        const auto op = this->make_op(lhs, rhs, autob);

        EXPECT_EQ(op->get_element_type(), element::i32);
        EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
    }
    {
        auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 3, 4, 5});
        auto rhs = std::make_shared<Parameter>(element::i32, Shape{5});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 3);
        const auto op = this->make_op(lhs, rhs, autob);

        EXPECT_EQ(op->get_element_type(), element::i32);
        EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
    }
    {
        auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 3, 4, 5});
        auto rhs = std::make_shared<Parameter>(element::i32, Shape{1, 3});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 0);
        const auto op = this->make_op(lhs, rhs, autob);

        EXPECT_EQ(op->get_element_type(), element::i32);
        EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
    }
    {
        auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 3, 4, 5});
        auto rhs = std::make_shared<Parameter>(element::i32, Shape{3, 1, 5});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1);
        const auto op = this->make_op(lhs, rhs, autob);

        EXPECT_EQ(op->get_element_type(), element::i32);
        EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
    }
}

TYPED_TEST_P(BitwiseOperator, static_shape_inference_4D_x_4D_pdpd_broadcast) {
    {
        auto lhs = std::make_shared<Parameter>(element::i32, Shape{8, 1, 6, 5});
        auto rhs = std::make_shared<Parameter>(element::i32, Shape{8, 1, 6, 5});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD);
        const auto op = this->make_op(lhs, rhs, autob);

        EXPECT_EQ(op->get_element_type(), element::i32);
        EXPECT_EQ(op->get_shape(), (Shape{8, 1, 6, 5}));
        EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
    }
    {
        auto lhs = std::make_shared<Parameter>(element::i32, Shape{8, 7, 6, 5});
        auto rhs = std::make_shared<Parameter>(element::i32, Shape{8, 1, 6, 5});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD);
        const auto op = this->make_op(lhs, rhs, autob);

        EXPECT_EQ(op->get_element_type(), element::i32);
        EXPECT_EQ(op->get_shape(), (Shape{8, 7, 6, 5}));
        EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
    }
}

TYPED_TEST_P(BitwiseOperator, static_shape_inference_4D_x_3D_ax_default_pdpd_broadcast) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{8, 7, 6, 5});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{7, 1, 5});

    const auto op = this->make_op(lhs, rhs, op::AutoBroadcastType::PDPD);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_shape(), (Shape{8, 7, 6, 5}));
    EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
}

TYPED_TEST_P(BitwiseOperator, incompatible_element_types_f32) {
    auto lhs = std::make_shared<Parameter>(element::f32, Shape{2, 2, 3, 3});
    auto rhs = std::make_shared<Parameter>(element::f32, Shape{2, 2, 3, 3});

    OV_EXPECT_THROW(std::ignore = this->make_op(lhs, rhs),
                    NodeValidationFailure,
                    HasSubstr("The element type of the input tensor must be integer or boolean."));
}

TYPED_TEST_P(BitwiseOperator, shape_inference_1D_x_1D_incompatible) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{3});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{4});

    ASSERT_THROW(const auto unused = this->make_op(lhs, rhs), NodeValidationFailure);
}

TYPED_TEST_P(BitwiseOperator, shape_inference_3D_x_3D_incompatible) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{3, 5, 6});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{4, 10, 12});

    ASSERT_THROW(const auto unused = this->make_op(lhs, rhs), NodeValidationFailure);
}

TYPED_TEST_P(BitwiseOperator, shape_inference_5D_x_5D_incompatible) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{389, 112, 12});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{389, 112, 19});

    ASSERT_THROW(const auto unused = this->make_op(lhs, rhs), NodeValidationFailure);
}

TYPED_TEST_P(BitwiseOperator, shape_inference_axis_less_than_negative_1_pdpd_incompatible) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{3, 1});

    const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, -2);

    ASSERT_THROW(const auto unused = this->make_op(lhs, rhs, autob), NodeValidationFailure);
}

TYPED_TEST_P(BitwiseOperator, shape_inference_dst_smaller_than_src_pdpd_broadcast) {
    auto lhs = std::make_shared<Parameter>(element::i32, Shape{2, 3, 4, 1});
    auto rhs = std::make_shared<Parameter>(element::i32, Shape{2, 3, 4, 5});

    const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD);

    ASSERT_THROW(const auto unused = this->make_op(lhs, rhs, autob), NodeValidationFailure);
}

TYPED_TEST_P(BitwiseOperator, fully_dynamic_shape_broadcast_numpy) {
    auto param = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY);

    const auto op = this->make_op(param, param, autob);
    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST_P(BitwiseOperator, fully_dynamic_shape_broadcast_none) {
    auto param = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NONE);

    const auto op = this->make_op(param, param, autob);
    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST_P(BitwiseOperator, fully_dynamic_shape_broadcast_pdpd) {
    auto param = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD);

    const auto op = this->make_op(param, param, autob);
    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST_P(BitwiseOperator, dynamic_shape_3D) {
    Dimension dynamic = Dimension::dynamic();
    auto lhs = std::make_shared<Parameter>(element::i32, PartialShape{dynamic, dynamic, 6});
    auto rhs = std::make_shared<Parameter>(element::i32, PartialShape{dynamic, dynamic, 6});

    const auto op = this->make_op(lhs, rhs);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{dynamic, dynamic, 6}));
}

TYPED_TEST_P(BitwiseOperator, dynamic_shape_5D) {
    Dimension dynamic = Dimension::dynamic();
    auto lhs = std::make_shared<Parameter>(element::i32, PartialShape{dynamic, 4, dynamic, dynamic, 6});
    auto rhs = std::make_shared<Parameter>(element::i32, PartialShape{dynamic, 4, dynamic, dynamic, 6});

    const auto op = this->make_op(lhs, rhs);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{dynamic, 4, dynamic, dynamic, 6}));
}

TYPED_TEST_P(BitwiseOperator, dynamic_shape_intervals_broadcast_none) {
    auto lhs = std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {2, 7}, {6, -1}, {-1, 6}, -1, 8});
    auto rhs = std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {2, 7}, {6, -1}, {-1, 6}, -1, 8});

    const auto op = this->make_op(lhs, rhs, op::AutoBroadcastType::NONE);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 3}, {2, 7}, {6, -1}, {-1, 6}, -1, 8}));
}

TYPED_TEST_P(BitwiseOperator, dynamic_shape_intervals_equal_rank_broadcast_numpy) {
    // Equal rank
    auto lhs = std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {1, 3}, {1, 3}, {4, 8}, -1, 1, -1, 1, 3});
    auto rhs = std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {2, 7}, -1, 1, {1, 3}, {4, 8}, -1, 1, 3});

    const auto op = this->make_op(lhs, rhs);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 3}, {2, 7}, -1, {4, 8}, -1, {4, 8}, -1, 1, 3}));
}

TYPED_TEST_P(BitwiseOperator, dynamic_shape_intervals_a_rank_smaller_broadcast_numpy) {
    // `lhs` rank smaller
    auto lhs = std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {4, 8}, -1, 1, -1, 1, 3});
    auto rhs = std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {2, 7}, -1, 1, {1, 3}, {4, 8}, -1, 1, 3});

    const auto op = this->make_op(lhs, rhs);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 3}, {2, 7}, -1, {4, 8}, -1, {4, 8}, -1, 1, 3}));
}

TYPED_TEST_P(BitwiseOperator, dynamic_shape_intervals_b_rank_smaller_broadcast_numpy) {
    // `rhs` rank smaller
    auto lhs = std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {2, 7}, -1, 1, {1, 3}, {4, 8}, -1, 1, 3});
    auto rhs = std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {4, 8}, -1, 1, -1, 1, 3});

    const auto op = this->make_op(lhs, rhs);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 3}, {2, 7}, -1, {4, 8}, -1, {4, 8}, -1, 1, 3}));
}

TYPED_TEST_P(BitwiseOperator, dynamic_shape_intervals_broadcast_pdpd) {
    {  // Equal rank
        auto lhs = std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {2, 7}, {1, 6}, {6, -1}, -1, 8});
        auto rhs = std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {2, 7}, 1, 1, -1, 8});

        const auto op = this->make_op(lhs, rhs, op::AutoBroadcastType::PDPD);

        EXPECT_EQ(op->get_element_type(), element::i32);
        EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 3}, {2, 7}, {1, 6}, {6, -1}, -1, 8}));
    }
    {  // `lhs` rank smaller
        auto lhs =
            std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {1, 3}, {1, 3}, {4, 8}, -1, 1, -1, 1, 3});
        auto rhs =
            std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {2, 7}, -1, 1, {1, 3}, {4, 8}, -1, 1, 3});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 0);
        const auto op = this->make_op(lhs, rhs, autob);

        EXPECT_EQ(op->get_element_type(), element::i32);
        EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 3}, {2, 7}, -1, {4, 8}, -1, {4, 8}, -1, 1, 3}));
    }
    {  // `rhs` rank smaller
        auto lhs =
            std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {2, 7}, -1, 1, {1, 3}, {4, 8}, -1, 1, 3});
        auto rhs = std::make_shared<Parameter>(element::i32, PartialShape{{1, 3}, {4, 8}, -1, 1, -1, 1, 3});

        const auto op = this->make_op(lhs, rhs);

        EXPECT_EQ(op->get_element_type(), element::i32);
        EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 3}, {2, 7}, -1, {4, 8}, -1, {4, 8}, -1, 1, 3}));
    }
}

TYPED_TEST_P(BitwiseOperator, labels_a_dynamic_mixed_dims_broadcast_numpy) {
    // All dimensions of lhs have labels, rhs without labels
    PartialShape pshape_lhs{{-1}, {3}, {1}, {2, 128}};
    PartialShape pshape_rhs{{-1}, {3}, {2, 224}, {1}};

    PartialShape expected_shape = {-1, 3, {2, 224}, {2, 128}};

    set_shape_labels(pshape_lhs, {10, 11, 12, 13});
    set_shape_labels(expected_shape, {10, 11, 0, 13});

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(BitwiseOperator, labels_b_dynamic_mixed_dims_broadcast_numpy) {
    // All dimensions of rhs have labels, lhs without labels
    PartialShape pshape_lhs{{-1}, {3}, {1}, {2, 128}};
    PartialShape pshape_rhs{{-1}, {3}, {2, 224}, {1}};

    PartialShape expected_shape = {-1, 3, {2, 224}, {2, 128}};

    set_shape_labels(pshape_rhs, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {20, 21, 22, 0});

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(BitwiseOperator, labels_different_interval_mixed_dims_broadcast_numpy) {
    // Both params have dimensions with different labels
    PartialShape pshape_lhs{{-1}, {3}, {1}, {2, 128}};
    PartialShape pshape_rhs{{-1}, {3}, {2, 224}, {1}};

    PartialShape expected_shape = {-1, 3, {2, 224}, {2, 128}};

    set_shape_labels(pshape_lhs, {10, 11, 12, 13});
    set_shape_labels(pshape_rhs, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {0, 21, 22, 13});

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(BitwiseOperator, labels_different_interval_b_and_fully_dyn_a_broadcast_numpy) {
    // Both params have dimension labels, output has label rhs
    Dimension dim_0_lhs = {-1};
    Dimension dim_0_rhs = {2, 4};

    DimensionTracker::set_label(dim_0_lhs, 10);
    DimensionTracker::set_label(dim_0_rhs, 20);

    PartialShape pshape_lhs = {dim_0_lhs, 3, 224, 1}, pshape_rhs = {dim_0_rhs, 3, 1, 224};
    PartialShape expected_shape = {{2, 4}, 3, 224, 224};
    TensorLabel expected_labels{20, 0, 0, 0};

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(BitwiseOperator, labels_different_interval_a_and_fully_dyn_b_broadcast_numpy) {
    // Both params have dimension labels, output has label lhs
    Dimension dim_0_lhs = {2, 4};
    Dimension dim_0_rhs = {-1};

    DimensionTracker::set_label(dim_0_lhs, 10);
    DimensionTracker::set_label(dim_0_rhs, 20);

    PartialShape pshape_lhs = {dim_0_lhs, 3, 224, 1}, pshape_rhs = {dim_0_rhs, 3, 1, 224};
    PartialShape expected_shape = {{2, 4}, 3, 224, 224};
    TensorLabel expected_labels{10, 0, 0, 0};

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(BitwiseOperator, labels_equal_interval_dims_without_one_broadcast_numpy) {
    // Both params have dynamic interval dimension the same labels
    PartialShape pshape_lhs{{2, 4}, {8, 16}, {8, 16}, {8, 16}};
    PartialShape pshape_rhs{{2, 4}, {4, 12}, {10, 12}, {16, 24}};

    PartialShape expected_shape = {{2, 4}, {8, 12}, {10, 12}, 16};

    set_shape_labels(pshape_lhs, {10, 11, 12, 13});
    set_shape_labels(pshape_rhs, {10, 11, 12, 13});
    set_shape_labels(expected_shape, {10, 11, 12, 13});

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(BitwiseOperator, labels_different_interval_dims_without_one_broadcast_numpy) {
    // Both params have dynamic interval dimension different labels
    PartialShape pshape_lhs{{2, 4}, {8, 16}, {8, 16}, {8, 16}};
    PartialShape pshape_rhs{{2, 4}, {4, 12}, {10, 12}, {16, 24}};

    PartialShape expected_shape = {{2, 4}, {8, 12}, {10, 12}, 16};
    TensorLabel expected_labels{20, 21, 22, 23};

    set_shape_labels(pshape_lhs, {10, 11, 12, 13});
    set_shape_labels(pshape_rhs, {20, 21, 22, 23});

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(BitwiseOperator, labels_different_interval_batch_without_one_equivalence_table_broadcast_numpy) {
    // Both params have dynamic interval dimension different labels, use table of equivalence
    auto table_of_equivalence = std::make_shared<TableOfEquivalence>();
    DimensionTracker dim_tracker(table_of_equivalence);

    Dimension dim_0_lhs = {2, 4};
    Dimension dim_0_rhs = {2, 4};

    dim_tracker.set_up_for_tracking(dim_0_lhs, 10);
    dim_tracker.set_up_for_tracking(dim_0_rhs, 20);

    PartialShape pshape_lhs = {dim_0_lhs, 3, 224, 1}, pshape_rhs = {dim_0_rhs, 3, 1, 224};

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    PartialShape expected_shape = {{2, 4}, 3, 224, 224};
    TensorLabel expected_labels{20, 0, 0, 0};

    auto eq_table = table_of_equivalence->get_equivalence_table();
    EXPECT_EQ(*eq_table[DimensionTracker::get_label(dim_0_lhs)], std::set<label_t>({10, 20}));
    EXPECT_EQ(*eq_table[DimensionTracker::get_label(dim_0_rhs)], std::set<label_t>({10, 20}));

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(BitwiseOperator, labels_different_fully_dynamic_batch_broadcast_numpy) {
    // Both params have fully dynamic dimension and different labels
    Dimension dim_0_lhs = {-1};
    Dimension dim_0_rhs = {-1};

    DimensionTracker::set_label(dim_0_lhs, 10);
    DimensionTracker::set_label(dim_0_rhs, 20);

    PartialShape pshape_lhs = {dim_0_lhs, 3, 224, 1}, pshape_rhs = {dim_0_rhs, 3, 1, 224};
    PartialShape expected_shape = {-1, 3, 224, 224};
    TensorLabel expected_labels{0, 0, 0, 0};

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(BitwiseOperator, labels_equal_fully_dynamic_batch_broadcast_numpy) {
    // Both params have fully dynamic dimension and the same labels
    Dimension dim_0_lhs = {-1};
    Dimension dim_0_rhs = {-1};

    DimensionTracker::set_label(dim_0_lhs, 10);
    DimensionTracker::set_label(dim_0_rhs, 10);

    PartialShape pshape_lhs = {dim_0_lhs, 3, 224, 1}, pshape_rhs = {dim_0_rhs, 3, 1, 224};
    PartialShape expected_shape = {-1, 3, 224, 224};
    TensorLabel expected_labels{10, 0, 0, 0};

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(BitwiseOperator, labels_dyn_batch_a_broadcast_numpy) {
    Dimension dim_0_lhs = -1;
    DimensionTracker::set_label(dim_0_lhs, 10);
    PartialShape pshape_lhs = {dim_0_lhs, 3, 224, 224}, pshape_rhs = {1, 3, 1, 1};
    PartialShape expected_shape{dim_0_lhs, 3, 224, 224};

    TensorLabel expected_labels{10, 0, 0, 0};

    auto lhs = std::make_shared<Parameter>(element::i64, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i64, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(BitwiseOperator, labels_dyn_batch_b_broadcast_numpy) {
    Dimension dim_0_rhs = -1;
    DimensionTracker::set_label(dim_0_rhs, 10);
    PartialShape pshape_rhs = {dim_0_rhs, 3, 224, 224}, pshape_lhs = {1, 3, 1, 1};
    PartialShape expected_shape{dim_0_rhs, 3, 224, 224};

    TensorLabel expected_labels{10, 0, 0, 0};

    auto lhs = std::make_shared<Parameter>(element::i64, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i64, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(BitwiseOperator, labels_dyn_batch_and_higher_rank_a_broadcast_numpy) {
    Dimension dim_0_lhs = -1;
    DimensionTracker::set_label(dim_0_lhs, 10);

    PartialShape pshape_lhs{dim_0_lhs, -1, -1, -1};
    PartialShape pshape_rhs{3, 1, 1};
    PartialShape expected_shape{dim_0_lhs, 3, -1, -1};

    TensorLabel expected_labels{10, 0, 0, 0};

    auto lhs = std::make_shared<Parameter>(element::i64, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i64, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(BitwiseOperator, labels_dyn_batch_and_higher_rank_b_broadcast_numpy) {
    Dimension dim_0_rhs = -1;
    DimensionTracker::set_label(dim_0_rhs, 10);

    PartialShape pshape_lhs{3, 1, 1};
    PartialShape pshape_rhs{dim_0_rhs, -1, -1, -1};
    PartialShape expected_shape{dim_0_rhs, 3, -1, -1};

    TensorLabel expected_labels{10, 0, 0, 0};

    auto lhs = std::make_shared<Parameter>(element::i64, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i64, pshape_rhs);

    const auto op = this->make_op(lhs, rhs);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(BitwiseOperator, labels_different_static_shape_broadcast_numpy) {
    // Static shape, different labels
    PartialShape pshape_lhs{{2}, {1}, {224}, {1}};
    PartialShape pshape_rhs{{2}, {1}, {1}, {128}};
    PartialShape expected_shape{2, 1, 224, 128};

    // Different labels
    set_shape_labels(pshape_lhs, {10, 11, 12, 13});
    set_shape_labels(pshape_rhs, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {20, 21, 12, 23});

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);
    const auto op = this->make_op(lhs, rhs, op::AutoBroadcastType::NUMPY);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(BitwiseOperator, labels_equal_static_shape_broadcast_numpy) {
    // Static shape, the same labels
    PartialShape pshape_lhs{2, 1, 224, 1};
    PartialShape pshape_rhs{2, 1, 1, 128};
    PartialShape expected_shape{2, 1, 224, 128};

    // Equal labels
    set_shape_labels(pshape_lhs, {30, 31, 32, 33});
    set_shape_labels(pshape_rhs, {30, 31, 32, 33});
    set_shape_labels(expected_shape, {30, 31, 32, 33});

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);
    const auto op = this->make_op(lhs, rhs, op::AutoBroadcastType::NUMPY);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(BitwiseOperator, labels_different_static_shape_broadcast_none) {
    // Static shape
    PartialShape pshape_lhs{2, 3, 224, 128};
    PartialShape pshape_rhs{2, 3, 224, 128};
    PartialShape expected_shape{2, 3, 224, 128};

    // Different labels
    set_shape_labels(pshape_lhs, {10, 11, 12, 13});
    set_shape_labels(pshape_rhs, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {20, 21, 22, 23});

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);
    const auto op = this->make_op(lhs, rhs, op::AutoBroadcastType::NONE);

    auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(BitwiseOperator, labels_equal_static_shape_broadcast_none) {
    // Static shape
    PartialShape pshape_lhs{2, 3, 224, 128};
    PartialShape pshape_rhs{2, 3, 224, 128};
    PartialShape expected_shape{2, 3, 224, 128};

    // Equal labels
    set_shape_labels(pshape_lhs, {30, 31, 32, 33});
    set_shape_labels(pshape_rhs, {30, 31, 32, 33});
    set_shape_labels(expected_shape, {30, 31, 32, 33});

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);
    const auto op = this->make_op(lhs, rhs, op::AutoBroadcastType::NONE);

    auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(BitwiseOperator, labels_different_dynamic_shape_broadcast_none) {
    // Dynamic shape
    PartialShape pshape_lhs{{-1}, {3}, {2, 224}, {1, 128}};
    PartialShape pshape_rhs{{-1}, {3}, {2, 224}, {1, 128}};
    PartialShape expected_shape{-1, 3, {2, 224}, {1, 128}};

    // Different labels
    set_shape_labels(pshape_lhs, {10, 11, 12, 13});
    set_shape_labels(pshape_rhs, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {20, 21, 22, 23});

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);
    const auto op = this->make_op(lhs, rhs, op::AutoBroadcastType::NONE);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(BitwiseOperator, labels_equal_dynamic_shape_broadcast_none) {
    // Dynamic shape
    PartialShape pshape_lhs{{-1}, {3}, {2, 224}, {1, 128}};
    PartialShape pshape_rhs{{-1}, {3}, {2, 224}, {1, 128}};
    PartialShape expected_shape{-1, 3, {2, 224}, {1, 128}};

    // Equal labels
    set_shape_labels(pshape_lhs, {30, 31, 32, 33});
    set_shape_labels(pshape_rhs, {30, 31, 32, 33});
    set_shape_labels(expected_shape, {30, 31, 32, 33});

    auto lhs = std::make_shared<Parameter>(element::i32, pshape_lhs);
    auto rhs = std::make_shared<Parameter>(element::i32, pshape_rhs);
    const auto op = this->make_op(lhs, rhs, op::AutoBroadcastType::NONE);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

REGISTER_TYPED_TEST_SUITE_P(BitwiseOperator,
                            default_constructor_integer,
                            default_constructor_boolean,

                            // Static shapes
                            shape_inference_2D,
                            shape_inference_4D,
                            default_autobroadcast,
                            no_autobroadcast,
                            shape_inference_4D_x_scalar_numpy_broadcast,
                            shape_inference_4D_x_1D_numpy_broadcast,
                            shape_inference_2D_x_4D_numpy_broadcast,
                            shape_inference_3D_x_4D_numpy_broadcast,
                            shape_inference_4D_x_3D_numpy_broadcast,
                            static_shape_pdpd_doc_examples,
                            static_shape_inference_4D_x_4D_pdpd_broadcast,
                            static_shape_inference_4D_x_3D_ax_default_pdpd_broadcast,
                            incompatible_element_types_f32,
                            shape_inference_1D_x_1D_incompatible,
                            shape_inference_3D_x_3D_incompatible,
                            shape_inference_5D_x_5D_incompatible,
                            shape_inference_axis_less_than_negative_1_pdpd_incompatible,
                            shape_inference_dst_smaller_than_src_pdpd_broadcast,

                            // Dynamic shapes
                            fully_dynamic_shape_broadcast_numpy,
                            fully_dynamic_shape_broadcast_none,
                            fully_dynamic_shape_broadcast_pdpd,
                            dynamic_shape_3D,
                            dynamic_shape_5D,
                            dynamic_shape_intervals_broadcast_none,
                            dynamic_shape_intervals_equal_rank_broadcast_numpy,
                            dynamic_shape_intervals_a_rank_smaller_broadcast_numpy,
                            dynamic_shape_intervals_b_rank_smaller_broadcast_numpy,
                            dynamic_shape_intervals_broadcast_pdpd,

                            // Dimension labels (static and dynamic)
                            labels_a_dynamic_mixed_dims_broadcast_numpy,
                            labels_b_dynamic_mixed_dims_broadcast_numpy,
                            labels_different_interval_mixed_dims_broadcast_numpy,
                            labels_different_interval_b_and_fully_dyn_a_broadcast_numpy,
                            labels_different_interval_a_and_fully_dyn_b_broadcast_numpy,
                            labels_equal_interval_dims_without_one_broadcast_numpy,
                            labels_different_interval_dims_without_one_broadcast_numpy,
                            labels_different_interval_batch_without_one_equivalence_table_broadcast_numpy,
                            labels_different_fully_dynamic_batch_broadcast_numpy,
                            labels_equal_fully_dynamic_batch_broadcast_numpy,
                            labels_dyn_batch_a_broadcast_numpy,
                            labels_dyn_batch_b_broadcast_numpy,
                            labels_dyn_batch_and_higher_rank_a_broadcast_numpy,
                            labels_dyn_batch_and_higher_rank_b_broadcast_numpy,
                            labels_different_static_shape_broadcast_numpy,
                            labels_equal_static_shape_broadcast_numpy,
                            labels_different_static_shape_broadcast_none,
                            labels_equal_static_shape_broadcast_none,
                            labels_different_dynamic_shape_broadcast_none,
                            labels_equal_dynamic_shape_broadcast_none);
