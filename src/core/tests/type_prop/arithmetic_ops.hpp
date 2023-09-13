// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "common_test_utils/type_prop.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/util/attr_types.hpp"

using namespace ov;
using namespace testing;

template <class T>
class ArithmeticOperator : public testing::Test {};

TYPED_TEST_SUITE_P(ArithmeticOperator);

TYPED_TEST_P(ArithmeticOperator, default_constructor) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32,
                                                     PartialShape{-1, 4, 1, 6, Dimension(1, 6), Dimension(2, 6)});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32,
                                                     PartialShape{-1, 1, 5, 6, Dimension(5, 8), Dimension(5, 8)});

    const auto op = std::make_shared<TypeParam>();

    op->set_argument(0, A);
    op->set_argument(1, B);

    auto autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NONE);
    op->set_autob(autob);
    EXPECT_EQ(op->get_autob(), ov::op::AutoBroadcastType::NONE);
    ASSERT_THROW(op->validate_and_infer_types(), ov::NodeValidationFailure);

    autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY);
    op->set_autob(autob);
    EXPECT_EQ(op->get_autob(), ov::op::AutoBroadcastType::NUMPY);

    op->validate_and_infer_types();

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (ov::PartialShape{-1, 4, 5, 6, ov::Dimension(5, 8), ov::Dimension(5, 6)}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_2D) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_shape(), (ov::Shape{2, 2}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_4D) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 3, 3});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 3, 3});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_shape(), (ov::Shape{2, 2, 3, 3}));
}

TYPED_TEST_P(ArithmeticOperator, default_autobroadcast) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_shape(), (ov::Shape{2, 2}));
    EXPECT_EQ(op->get_autob(), ov::op::AutoBroadcastType::NUMPY);
}

TYPED_TEST_P(ArithmeticOperator, no_autobroadcast) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});

    const auto op = std::make_shared<TypeParam>(A, B, ov::op::AutoBroadcastType::NONE);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_shape(), (ov::Shape{2, 2}));
    EXPECT_EQ(op->get_autob(), ov::op::AutoBroadcastType::NONE);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_4D_x_scalar_numpy_broadcast) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_shape(), (ov::Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_4D_x_1D_numpy_broadcast) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_shape(), (ov::Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_2D_x_4D_numpy_broadcast) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 5});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_shape(), (ov::Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_3D_x_4D_numpy_broadcast) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 5});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 1, 1});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_shape(), (ov::Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_4D_x_3D_numpy_broadcast) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{8, 1, 6, 1});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{7, 1, 5});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_shape(), (ov::Shape{8, 7, 6, 5}));
    EXPECT_EQ(op->get_autob(), ov::op::AutoBroadcastType::NUMPY);
}

TYPED_TEST_P(ArithmeticOperator, static_shape_pdpd_doc_examples) {
    {
        auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

        const auto autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 1);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), ov::element::f32);
        EXPECT_EQ(op->get_shape(), (ov::Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, ov::op::AutoBroadcastType::PDPD);
    }
    {
        auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 1});

        const auto autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 1);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), ov::element::f32);
        EXPECT_EQ(op->get_shape(), (ov::Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, ov::op::AutoBroadcastType::PDPD);
    }
    {
        auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});

        const auto autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), ov::element::f32);
        EXPECT_EQ(op->get_shape(), (ov::Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, ov::op::AutoBroadcastType::PDPD);
    }
    {
        auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{5});

        const auto autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 3);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), ov::element::f32);
        EXPECT_EQ(op->get_shape(), (ov::Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, ov::op::AutoBroadcastType::PDPD);
    }
    {
        auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3});

        const auto autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 0);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), ov::element::f32);
        EXPECT_EQ(op->get_shape(), (ov::Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, ov::op::AutoBroadcastType::PDPD);
    }
    {
        auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 1, 5});

        const auto autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 1);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), ov::element::f32);
        EXPECT_EQ(op->get_shape(), (ov::Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, ov::op::AutoBroadcastType::PDPD);
    }
}

TYPED_TEST_P(ArithmeticOperator, static_shape_inference_4D_x_4D_pdpd_broadcast) {
    {
        auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{8, 1, 6, 5});
        auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{8, 1, 6, 5});

        const auto autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), ov::element::f32);
        EXPECT_EQ(op->get_shape(), (ov::Shape{8, 1, 6, 5}));
        EXPECT_EQ(op->get_autob().m_type, ov::op::AutoBroadcastType::PDPD);
    }
    {
        auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{8, 7, 6, 5});
        auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{8, 1, 6, 5});

        const auto autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), ov::element::f32);
        EXPECT_EQ(op->get_shape(), (ov::Shape{8, 7, 6, 5}));
        EXPECT_EQ(op->get_autob().m_type, ov::op::AutoBroadcastType::PDPD);
    }
}

TYPED_TEST_P(ArithmeticOperator, static_shape_inference_4D_x_3D_ax_default_pdpd_broadcast) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{8, 7, 6, 5});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{7, 1, 5});

    const auto op = std::make_shared<TypeParam>(A, B, ov::op::AutoBroadcastType::PDPD);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_shape(), (ov::Shape{8, 7, 6, 5}));
    EXPECT_EQ(op->get_autob().m_type, ov::op::AutoBroadcastType::PDPD);
}

TYPED_TEST_P(ArithmeticOperator, incompatible_element_types) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 3, 3});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 2, 3, 3});

    ASSERT_THROW(const auto unused = std::make_shared<TypeParam>(A, B), ov::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, incompatible_boolean_type) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::boolean, Shape{2, 2, 3, 3});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::boolean, Shape{2, 2, 3, 3});

    ASSERT_THROW(const auto unused = std::make_shared<TypeParam>(A, B), ov::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_1D_x_1D_incompatible) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});

    ASSERT_THROW(const auto unused = std::make_shared<TypeParam>(A, B), ov::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_3D_x_3D_incompatible) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 5, 6});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 10, 12});

    ASSERT_THROW(const auto unused = std::make_shared<TypeParam>(A, B), ov::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_5D_x_5D_incompatible) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{389, 112, 12});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{389, 112, 19});

    ASSERT_THROW(const auto unused = std::make_shared<TypeParam>(A, B), ov::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_axis_less_than_negative_1_pdpd_incompatible) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 1});

    const auto autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, -2);

    ASSERT_THROW(const auto unused = std::make_shared<TypeParam>(A, B, autob), ov::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_dst_smaller_than_src_pdpd_broadcast) {
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 1});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});

    const auto autob = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD);

    ASSERT_THROW(const auto unused = std::make_shared<TypeParam>(A, B, autob), ov::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, fully_dynamic_shape_broadcast_numpy) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY);

    const auto op = std::make_shared<TypeParam>(param, param, autob);
    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape::dynamic());
}

TYPED_TEST_P(ArithmeticOperator, fully_dynamic_shape_broadcast_none) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NONE);

    const auto op = std::make_shared<TypeParam>(param, param, autob);
    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape::dynamic());
}

TYPED_TEST_P(ArithmeticOperator, fully_dynamic_shape_broadcast_pdpd) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD);

    const auto op = std::make_shared<TypeParam>(param, param, autob);
    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape::dynamic());
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_3D) {
    Dimension dynamic = Dimension::dynamic();
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{dynamic, dynamic, 6});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{dynamic, dynamic, 6});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{dynamic, dynamic, 6}));
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_5D) {
    Dimension dynamic = Dimension::dynamic();
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{dynamic, 4, dynamic, dynamic, 6});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{dynamic, 4, dynamic, dynamic, 6});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{dynamic, 4, dynamic, dynamic, 6}));
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_intervals_broadcast_none) {
    auto A = std::make_shared<ov::op::v0::Parameter>(
        element::f32,
        PartialShape{Dimension(1, 3), Dimension(2, 7), Dimension(6, -1), Dimension(-1, 6), -1, 8});
    auto B = std::make_shared<ov::op::v0::Parameter>(
        element::f32,
        PartialShape{Dimension(1, 3), Dimension(2, 7), Dimension(6, -1), Dimension(-1, 6), -1, 8});

    const auto op = std::make_shared<TypeParam>(A, B, ov::op::AutoBroadcastType::NONE);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (ov::PartialShape{ov::Dimension(1, 3),
                                ov::Dimension(2, 7),
                                ov::Dimension(6, -1),
                                ov::Dimension(-1, 6),
                                -1,
                                8}));
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_intervals_equal_rank_broadcast_numpy) {
    // Equal rank
    auto A = std::make_shared<ov::op::v0::Parameter>(
        element::f32,
        PartialShape{Dimension(1, 3), Dimension(1, 3), Dimension(1, 3), Dimension(4, 8), -1, 1, -1, 1, 3});
    auto B = std::make_shared<ov::op::v0::Parameter>(
        element::f32,
        PartialShape{Dimension(1, 3), Dimension(2, 7), -1, 1, Dimension(1, 3), Dimension(4, 8), -1, 1, 3});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (ov::PartialShape{ov::Dimension(1, 3),
                                ov::Dimension(2, 7),
                                -1,
                                ov::Dimension(4, 8),
                                -1,
                                ov::Dimension(4, 8),
                                -1,
                                1,
                                3}));
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_intervals_a_rank_smaller_broadcast_numpy) {
    // `A` rank smaller
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32,
                                                     PartialShape{Dimension(1, 3), Dimension(4, 8), -1, 1, -1, 1, 3});
    auto B = std::make_shared<ov::op::v0::Parameter>(
        element::f32,
        PartialShape{Dimension(1, 3), Dimension(2, 7), -1, 1, Dimension(1, 3), Dimension(4, 8), -1, 1, 3});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (ov::PartialShape{ov::Dimension(1, 3),
                                ov::Dimension(2, 7),
                                -1,
                                ov::Dimension(4, 8),
                                -1,
                                ov::Dimension(4, 8),
                                -1,
                                1,
                                3}));
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_intervals_b_rank_smaller_broadcast_numpy) {
    // `B` rank smaller
    auto A = std::make_shared<ov::op::v0::Parameter>(
        element::f32,
        PartialShape{Dimension(1, 3), Dimension(2, 7), -1, 1, Dimension(1, 3), Dimension(4, 8), -1, 1, 3});
    auto B = std::make_shared<ov::op::v0::Parameter>(element::f32,
                                                     PartialShape{Dimension(1, 3), Dimension(4, 8), -1, 1, -1, 1, 3});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (ov::PartialShape{ov::Dimension(1, 3),
                                ov::Dimension(2, 7),
                                -1,
                                ov::Dimension(4, 8),
                                -1,
                                ov::Dimension(4, 8),
                                -1,
                                1,
                                3}));
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_intervals_broadcast_pdpd) {
    {  // Equal rank
        auto A = std::make_shared<ov::op::v0::Parameter>(
            element::f32,
            PartialShape{Dimension(1, 3), Dimension(2, 7), Dimension(1, 6), /* Dimension(6, -1), */ -1, 8});
        auto B =
            std::make_shared<ov::op::v0::Parameter>(element::f32,
                                                    PartialShape{Dimension(1, 3), Dimension(2, 7), 1, /* 1, */ -1, 8});

        const auto op = std::make_shared<TypeParam>(A, B, op::AutoBroadcastType::PDPD);

        EXPECT_EQ(op->get_element_type(), element::f32);
        EXPECT_EQ(op->get_output_partial_shape(0),
                  (ov::PartialShape{ov::Dimension(1, 3),
                                    ov::Dimension(2, 7),
                                    ov::Dimension(1, 6),
                                    /* Dimension(6, -1), */ -1,
                                    8}));
    }
    {  // `A` rank smaller
        auto A = std::make_shared<ov::op::v0::Parameter>(
            element::f32,
            PartialShape{Dimension(1, 3), Dimension(1, 3), Dimension(1, 3), Dimension(4, 8), -1, 1, -1, 1, 3});
        auto B = std::make_shared<ov::op::v0::Parameter>(
            element::f32,
            PartialShape{Dimension(1, 3), Dimension(2, 7), -1, 1, Dimension(1, 3), Dimension(4, 8), -1, 1, 3});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 0);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), ov::element::f32);
        EXPECT_EQ(op->get_output_partial_shape(0),
                  (ov::PartialShape{ov::Dimension(1, 3),
                                    ov::Dimension(2, 7),
                                    -1,
                                    ov::Dimension(4, 8),
                                    -1,
                                    ov::Dimension(4, 8),
                                    -1,
                                    1,
                                    3}));
    }
    {  // `B` rank smaller
        auto A = std::make_shared<ov::op::v0::Parameter>(
            element::f32,
            PartialShape{Dimension(1, 3), Dimension(2, 7), -1, 1, Dimension(1, 3), Dimension(4, 8), -1, 1, 3});
        auto B =
            std::make_shared<ov::op::v0::Parameter>(element::f32,
                                                    PartialShape{Dimension(1, 3), Dimension(4, 8), -1, 1, -1, 1, 3});

        const auto op = std::make_shared<TypeParam>(A, B);

        EXPECT_EQ(op->get_element_type(), ov::element::f32);
        EXPECT_EQ(op->get_output_partial_shape(0),
                  (ov::PartialShape{ov::Dimension(1, 3),
                                    ov::Dimension(2, 7),
                                    -1,
                                    ov::Dimension(4, 8),
                                    -1,
                                    ov::Dimension(4, 8),
                                    -1,
                                    1,
                                    3}));
    }
}

TYPED_TEST_P(ArithmeticOperator, labels_a_dynamic_mixed_dims_broadcast_numpy) {
    // All dimensions of A have labels, B without labels
    ov::PartialShape pshape_A{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(1), ov::Dimension(2, 128)};
    ov::PartialShape pshape_B{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1)};

    ov::PartialShape expected_shape = {-1, 3, ov::Dimension(2, 224), ov::Dimension(2, 128)};

    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(expected_shape, {10, 11, 0, 13});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_b_dynamic_mixed_dims_broadcast_numpy) {
    // All dimensions of B have labels, A without labels
    ov::PartialShape pshape_A{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(1), ov::Dimension(2, 128)};
    ov::PartialShape pshape_B{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1)};

    ov::PartialShape expected_shape = {-1, 3, ov::Dimension(2, 224), ov::Dimension(2, 128)};

    set_shape_labels(pshape_B, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {20, 21, 22, 0});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_different_interval_mixed_dims_broadcast_numpy) {
    // Both params have dimensions with different labels
    ov::PartialShape pshape_A{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(1), ov::Dimension(2, 128)};
    ov::PartialShape pshape_B{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1)};

    ov::PartialShape expected_shape = {-1, 3, ov::Dimension(2, 224), ov::Dimension(2, 128)};

    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(pshape_B, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {0, 21, 22, 13});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_different_interval_b_and_fully_dyn_a_broadcast_numpy) {
    // Both params have dimension labels, output has label B
    ov::Dimension dim_0_A = ov::Dimension(-1);
    ov::Dimension dim_0_B = ov::Dimension(2, 4);

    ov::DimensionTracker::set_label(dim_0_A, 10);
    ov::DimensionTracker::set_label(dim_0_B, 20);

    ov::PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};
    ov::PartialShape expected_shape = {ov::Dimension(2, 4), 3, 224, 224};
    ov::TensorLabel expected_labels{20, 0, 0, 0};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_different_interval_a_and_fully_dyn_b_broadcast_numpy) {
    // Both params have dimension labels, output has label A
    ov::Dimension dim_0_A = ov::Dimension(2, 4);
    ov::Dimension dim_0_B = ov::Dimension(-1);

    ov::DimensionTracker::set_label(dim_0_A, 10);
    ov::DimensionTracker::set_label(dim_0_B, 20);

    ov::PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};
    ov::PartialShape expected_shape = {ov::Dimension(2, 4), 3, 224, 224};
    ov::TensorLabel expected_labels{10, 0, 0, 0};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_equal_interval_dims_without_one_broadcast_numpy) {
    // Both params have dynamic interval dimension the same labels
    ov::PartialShape pshape_A{ov::Dimension(2, 4), ov::Dimension(8, 16), ov::Dimension(8, 16), ov::Dimension(8, 16)};
    ov::PartialShape pshape_B{ov::Dimension(2, 4), ov::Dimension(4, 12), ov::Dimension(10, 12), ov::Dimension(16, 24)};

    ov::PartialShape expected_shape = {ov::Dimension(2, 4), ov::Dimension(8, 12), ov::Dimension(10, 12), 16};

    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(pshape_B, {10, 11, 12, 13});
    set_shape_labels(expected_shape, {10, 11, 12, 13});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_different_interval_dims_without_one_broadcast_numpy) {
    // Both params have dynamic interval dimension different labels
    ov::PartialShape pshape_A{ov::Dimension(2, 4), ov::Dimension(8, 16), ov::Dimension(8, 16), ov::Dimension(8, 16)};
    ov::PartialShape pshape_B{ov::Dimension(2, 4), ov::Dimension(4, 12), ov::Dimension(10, 12), ov::Dimension(16, 24)};

    ov::PartialShape expected_shape = {ov::Dimension(2, 4), ov::Dimension(8, 12), ov::Dimension(10, 12), 16};
    ov::TensorLabel expected_labels{20, 21, 22, 23};

    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(pshape_B, {20, 21, 22, 23});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_different_interval_batch_without_one_equivalence_table_broadcast_numpy) {
    // Both params have dynamic interval dimension different labels, use table of equivalence
    auto table_of_equivalence = std::make_shared<ov::TableOfEquivalence>();
    ov::DimensionTracker dim_tracker(table_of_equivalence);

    ov::Dimension dim_0_A = ov::Dimension(2, 4);
    ov::Dimension dim_0_B = ov::Dimension(2, 4);

    dim_tracker.set_up_for_tracking(dim_0_A, 10);
    dim_tracker.set_up_for_tracking(dim_0_B, 20);

    ov::PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    ov::PartialShape expected_shape = {ov::Dimension(2, 4), 3, 224, 224};
    ov::TensorLabel expected_labels{20, 0, 0, 0};

    auto eq_table = table_of_equivalence->get_equivalence_table();
    EXPECT_EQ(*eq_table[ov::DimensionTracker::get_label(dim_0_A)], std::set<ov::label_t>({10, 20}));
    EXPECT_EQ(*eq_table[ov::DimensionTracker::get_label(dim_0_B)], std::set<ov::label_t>({10, 20}));

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_different_fully_dynamic_batch_broadcast_numpy) {
    // Both params have fully dynamic dimension and different labels
    ov::Dimension dim_0_A = ov::Dimension(-1);
    ov::Dimension dim_0_B = ov::Dimension(-1);

    ov::DimensionTracker::set_label(dim_0_A, 10);
    ov::DimensionTracker::set_label(dim_0_B, 20);

    ov::PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};
    ov::PartialShape expected_shape = {-1, 3, 224, 224};
    ov::TensorLabel expected_labels{0, 0, 0, 0};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_equal_fully_dynamic_batch_broadcast_numpy) {
    // Both params have fully dynamic dimension and the same labels
    ov::Dimension dim_0_A = ov::Dimension(-1);
    ov::Dimension dim_0_B = ov::Dimension(-1);

    ov::DimensionTracker::set_label(dim_0_A, 10);
    ov::DimensionTracker::set_label(dim_0_B, 10);

    ov::PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};
    ov::PartialShape expected_shape = {-1, 3, 224, 224};
    ov::TensorLabel expected_labels{10, 0, 0, 0};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_dyn_batch_a_broadcast_numpy) {
    ov::Dimension b = -1;
    ov::DimensionTracker::set_label(b, 10);
    ov::PartialShape A = {b, 3, 224, 224}, B = {1, 3, 1, 1};
    ov::PartialShape expected_shape{b, 3, 224, 224};

    ov::TensorLabel expected_labels{10, 0, 0, 0};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f64, A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f64, B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_dyn_batch_b_broadcast_numpy) {
    ov::Dimension b = -1;
    ov::DimensionTracker::set_label(b, 10);
    ov::PartialShape B = {b, 3, 224, 224}, A = {1, 3, 1, 1};
    ov::PartialShape expected_shape{b, 3, 224, 224};

    ov::TensorLabel expected_labels{10, 0, 0, 0};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f64, A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f64, B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_dyn_batch_and_higher_rank_a_broadcast_numpy) {
    ov::Dimension b = -1;
    ov::DimensionTracker::set_label(b, 10);

    ov::PartialShape pshape_A{b, -1, -1, -1};
    ov::PartialShape pshape_B{3, 1, 1};
    ov::PartialShape expected_shape{b, 3, -1, -1};

    ov::TensorLabel expected_labels{10, 0, 0, 0};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f64, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f64, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_dyn_batch_and_higher_rank_b_broadcast_numpy) {
    ov::Dimension b = -1;
    ov::DimensionTracker::set_label(b, 10);

    ov::PartialShape pshape_A{3, 1, 1};
    ov::PartialShape pshape_B{b, -1, -1, -1};
    ov::PartialShape expected_shape{b, 3, -1, -1};

    ov::TensorLabel expected_labels{10, 0, 0, 0};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f64, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f64, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_different_static_shape_broadcast_numpy) {
    // Static shape, different labels
    ov::PartialShape pshape_A{ov::Dimension(2), ov::Dimension(1), ov::Dimension(224), ov::Dimension(1)};
    ov::PartialShape pshape_B{ov::Dimension(2), ov::Dimension(1), ov::Dimension(1), ov::Dimension(128)};
    ov::PartialShape expected_shape{2, 1, 224, 128};

    // Different labels
    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(pshape_B, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {20, 21, 12, 23});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NUMPY);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_equal_static_shape_broadcast_numpy) {
    // Static shape, the same labels
    ov::PartialShape pshape_A{2, 1, 224, 1};
    ov::PartialShape pshape_B{2, 1, 1, 128};
    ov::PartialShape expected_shape{2, 1, 224, 128};

    // Equal labels
    set_shape_labels(pshape_A, {30, 31, 32, 33});
    set_shape_labels(pshape_B, {30, 31, 32, 33});
    set_shape_labels(expected_shape, {30, 31, 32, 33});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NUMPY);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_different_static_shape_broadcast_none) {
    // Static shape
    ov::PartialShape pshape_A{2, 3, 224, 128};
    ov::PartialShape pshape_B{2, 3, 224, 128};
    ov::PartialShape expected_shape{2, 3, 224, 128};

    // Different labels
    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(pshape_B, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {20, 21, 22, 23});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NONE);

    auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_equal_static_shape_broadcast_none) {
    // Static shape
    ov::PartialShape pshape_A{2, 3, 224, 128};
    ov::PartialShape pshape_B{2, 3, 224, 128};
    ov::PartialShape expected_shape{2, 3, 224, 128};

    // Equal labels
    set_shape_labels(pshape_A, {30, 31, 32, 33});
    set_shape_labels(pshape_B, {30, 31, 32, 33});
    set_shape_labels(expected_shape, {30, 31, 32, 33});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NONE);

    auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_different_dynamic_shape_broadcast_none) {
    // Dynamic shape
    ov::PartialShape pshape_A{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1, 128)};
    ov::PartialShape pshape_B{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1, 128)};
    ov::PartialShape expected_shape{-1, 3, ov::Dimension(2, 224), ov::Dimension(1, 128)};

    // Different labels
    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(pshape_B, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {20, 21, 22, 23});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NONE);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_equal_dynamic_shape_broadcast_none) {
    // Dynamic shape
    ov::PartialShape pshape_A{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1, 128)};
    ov::PartialShape pshape_B{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1, 128)};
    ov::PartialShape expected_shape{-1, 3, ov::Dimension(2, 224), ov::Dimension(1, 128)};

    // Equal labels
    set_shape_labels(pshape_A, {30, 31, 32, 33});
    set_shape_labels(pshape_B, {30, 31, 32, 33});
    set_shape_labels(expected_shape, {30, 31, 32, 33});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NONE);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

REGISTER_TYPED_TEST_SUITE_P(ArithmeticOperator,
                            default_constructor,

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
                            incompatible_element_types,
                            incompatible_boolean_type,
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
