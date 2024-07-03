// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/symbol.hpp"
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

TYPED_TEST_P(ArithmeticOperator, unsupported_element_type) {
    {
        auto A = std::make_shared<ov::op::v0::Parameter>(element::boolean, Shape{2, 2, 3, 3});
        auto B = std::make_shared<ov::op::v0::Parameter>(element::boolean, Shape{2, 2, 3, 3});

        OV_EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(A, B),
                        NodeValidationFailure,
                        HasSubstr("This operation does not support inputs with element type: boolean"));
    }
    {
        auto A = std::make_shared<ov::op::v0::Parameter>(element::string, Shape{2});
        auto B = std::make_shared<ov::op::v0::Parameter>(element::string, Shape{2});

        OV_EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(A, B),
                        NodeValidationFailure,
                        HasSubstr("This operation does not support inputs with element type: string"));
    }
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

TYPED_TEST_P(ArithmeticOperator, symbols_a_dynamic_mixed_dims_broadcast_numpy) {
    // All dimensions of A have symbols, B without symbols
    ov::PartialShape pshape_A{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(1), ov::Dimension(2, 128)};
    ov::PartialShape pshape_B{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1)};

    ov::PartialShape expected_shape = {-1, 3, ov::Dimension(2, 224), ov::Dimension(2, 128)};

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();

    set_shape_symbols(pshape_A, {A, B, C, D});
    set_shape_symbols(expected_shape, {A, B, nullptr, D});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), get_shape_symbols(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, symbols_b_dynamic_mixed_dims_broadcast_numpy) {
    // All dimensions of B have symbols, A without symbols
    ov::PartialShape pshape_A{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(1), ov::Dimension(2, 128)};
    ov::PartialShape pshape_B{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1)};

    ov::PartialShape expected_shape = {-1, 3, ov::Dimension(2, 224), ov::Dimension(2, 128)};

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();

    set_shape_symbols(pshape_B, {A, B, C, D});
    set_shape_symbols(expected_shape, {A, B, C, nullptr});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), get_shape_symbols(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, symbols_different_interval_mixed_dims_broadcast_numpy) {
    // Both params have dimensions with different symbols
    ov::PartialShape pshape_A{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(1), ov::Dimension(2, 128)};
    ov::PartialShape pshape_B{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1)};

    ov::PartialShape expected_shape = {-1, 3, ov::Dimension(2, 224), ov::Dimension(2, 128)};

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();
    auto E = std::make_shared<ov::Symbol>(), F = std::make_shared<ov::Symbol>(), G = std::make_shared<ov::Symbol>(),
         H = std::make_shared<ov::Symbol>();

    set_shape_symbols(pshape_A, {A, B, C, D});
    set_shape_symbols(pshape_B, {E, F, G, H});
    set_shape_symbols(expected_shape, {nullptr, B, G, D});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), get_shape_symbols(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, symbols_different_interval_b_and_fully_dyn_a_broadcast_numpy) {
    // Both params have dimension symbols, output has symbol B
    ov::Dimension dim_0_A = ov::Dimension(-1);
    ov::Dimension dim_0_B = ov::Dimension(2, 4);

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>();
    dim_0_A.set_symbol(A);
    dim_0_B.set_symbol(B);

    ov::PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};
    ov::PartialShape expected_shape = {ov::Dimension(2, 4), 3, 224, 224};
    ov::TensorSymbol expected_symbols{B, nullptr, nullptr, nullptr};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TYPED_TEST_P(ArithmeticOperator, symbols_different_interval_a_and_fully_dyn_b_broadcast_numpy) {
    // Both params have dimension symbols, output has symbol A
    ov::Dimension dim_0_A = ov::Dimension(2, 4);
    ov::Dimension dim_0_B = ov::Dimension(-1);

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>();
    dim_0_A.set_symbol(A);
    dim_0_B.set_symbol(B);

    ov::PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};
    ov::PartialShape expected_shape = {ov::Dimension(2, 4), 3, 224, 224};
    ov::TensorSymbol expected_symbols{A, nullptr, nullptr, nullptr};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TYPED_TEST_P(ArithmeticOperator, symbols_equal_interval_dims_without_one_broadcast_numpy) {
    // Both params have dynamic interval dimension the same symbols
    ov::PartialShape pshape_A{ov::Dimension(2, 4), ov::Dimension(8, 16), ov::Dimension(8, 16), ov::Dimension(8, 16)};
    ov::PartialShape pshape_B{ov::Dimension(2, 4), ov::Dimension(4, 12), ov::Dimension(10, 12), ov::Dimension(16, 24)};

    ov::PartialShape expected_shape = {ov::Dimension(2, 4), ov::Dimension(8, 12), ov::Dimension(10, 12), 16};

    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();

    set_shape_symbols(pshape_A, {A, B, C, D});
    set_shape_symbols(pshape_B, {A, B, C, D});
    set_shape_symbols(expected_shape, {A, B, C, D});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), get_shape_symbols(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, symbols_different_interval_dims_without_one_broadcast_numpy) {
    // Both params have dynamic interval dimension different symbols
    ov::PartialShape pshape_A{ov::Dimension(2, 4), ov::Dimension(8, 16), ov::Dimension(8, 16), ov::Dimension(8, 16)};
    ov::PartialShape pshape_B{ov::Dimension(2, 4), ov::Dimension(4, 12), ov::Dimension(10, 12), ov::Dimension(16, 24)};

    ov::PartialShape expected_shape = {ov::Dimension(2, 4), ov::Dimension(8, 12), ov::Dimension(10, 12), 16};
    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();
    auto E = std::make_shared<ov::Symbol>(), F = std::make_shared<ov::Symbol>(), G = std::make_shared<ov::Symbol>(),
         H = std::make_shared<ov::Symbol>();
    ov::TensorSymbol expected_symbols{A, B, C, D};
    set_shape_symbols(pshape_A, {A, B, C, D});
    set_shape_symbols(pshape_B, {E, F, G, H});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TYPED_TEST_P(ArithmeticOperator, symbols_different_interval_batch_without_one_equivalence_table_broadcast_numpy) {
    // Both params have dynamic interval dimension different symbols, use table of equivalence

    ov::Dimension dim_0_A = ov::Dimension(2, 4);
    auto A = std::make_shared<ov::Symbol>();
    dim_0_A.set_symbol(A);

    ov::Dimension dim_0_B = ov::Dimension(2, 4);
    auto B = std::make_shared<ov::Symbol>();
    dim_0_B.set_symbol(B);

    ov::PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    ov::PartialShape expected_shape = {ov::Dimension(2, 4), 3, 224, 224};

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_TRUE(ov::symbol::are_equal(out_shape[0].get_symbol(), A));
    EXPECT_TRUE(ov::symbol::are_equal(out_shape[0].get_symbol(), B));
    EXPECT_TRUE(ov::symbol::are_equal(A, B));
}

TYPED_TEST_P(ArithmeticOperator, symbols_different_fully_dynamic_batch_broadcast_numpy) {
    // Both params have fully dynamic dimension and different symbols
    ov::Dimension dim_0_A = ov::Dimension(-1);
    auto A = std::make_shared<ov::Symbol>();
    dim_0_A.set_symbol(A);

    ov::Dimension dim_0_B = ov::Dimension(-1);
    auto B = std::make_shared<ov::Symbol>();
    dim_0_B.set_symbol(B);

    ov::PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};
    ov::PartialShape expected_shape = {-1, 3, 224, 224};
    ov::TensorSymbol expected_symbols{nullptr, nullptr, nullptr, nullptr};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TYPED_TEST_P(ArithmeticOperator, symbols_equal_fully_dynamic_batch_broadcast_numpy) {
    // Both params have fully dynamic dimension and the same symbols
    auto A = std::make_shared<ov::Symbol>();

    ov::Dimension dim_0_A = ov::Dimension(-1);
    dim_0_A.set_symbol(A);
    ov::Dimension dim_0_B = ov::Dimension(-1);
    dim_0_B.set_symbol(A);

    ov::PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};
    ov::PartialShape expected_shape = {-1, 3, 224, 224};
    ov::TensorSymbol expected_symbols{A, nullptr, nullptr, nullptr};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TYPED_TEST_P(ArithmeticOperator, symbols_dyn_batch_a_broadcast_numpy) {
    auto S = std::make_shared<ov::Symbol>();
    ov::Dimension b = -1;
    b.set_symbol(S);
    ov::PartialShape A = {b, 3, 224, 224}, B = {1, 3, 1, 1};
    ov::PartialShape expected_shape{b, 3, 224, 224};

    ov::TensorSymbol expected_symbols{S, nullptr, nullptr, nullptr};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f64, A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f64, B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TYPED_TEST_P(ArithmeticOperator, symbols_dyn_batch_b_broadcast_numpy) {
    auto S = std::make_shared<ov::Symbol>();
    ov::Dimension b = -1;
    b.set_symbol(S);
    ov::PartialShape B = {b, 3, 224, 224}, A = {1, 3, 1, 1};
    ov::PartialShape expected_shape{b, 3, 224, 224};

    ov::TensorSymbol expected_symbols{S, nullptr, nullptr, nullptr};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f64, A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f64, B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TYPED_TEST_P(ArithmeticOperator, symbols_dyn_batch_and_higher_rank_a_broadcast_numpy) {
    auto S = std::make_shared<ov::Symbol>();
    ov::Dimension b = -1;
    b.set_symbol(S);

    ov::PartialShape pshape_A{b, -1, -1, -1};
    ov::PartialShape pshape_B{3, 1, 1};
    ov::PartialShape expected_shape{b, 3, -1, -1};

    ov::TensorSymbol expected_symbols{S, nullptr, nullptr, nullptr};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f64, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f64, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TYPED_TEST_P(ArithmeticOperator, symbols_dyn_batch_and_higher_rank_b_broadcast_numpy) {
    auto S = std::make_shared<ov::Symbol>();
    ov::Dimension b = -1;
    b.set_symbol(S);

    ov::PartialShape pshape_A{3, 1, 1};
    ov::PartialShape pshape_B{b, -1, -1, -1};
    ov::PartialShape expected_shape{b, 3, -1, -1};

    ov::TensorSymbol expected_symbols{S, nullptr, nullptr, nullptr};

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f64, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f64, pshape_B);

    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), expected_symbols);
}

TYPED_TEST_P(ArithmeticOperator, symbols_different_static_shape_broadcast_numpy) {
    // Static shape, different symbols
    ov::PartialShape pshape_A{ov::Dimension(2), ov::Dimension(1), ov::Dimension(224), ov::Dimension(1)};
    ov::PartialShape pshape_B{ov::Dimension(2), ov::Dimension(1), ov::Dimension(1), ov::Dimension(128)};
    ov::PartialShape expected_shape{2, 1, 224, 128};

    // Different symbols
    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();
    auto E = std::make_shared<ov::Symbol>(), F = std::make_shared<ov::Symbol>(), G = std::make_shared<ov::Symbol>(),
         H = std::make_shared<ov::Symbol>();

    set_shape_symbols(pshape_A, {A, B, C, D});
    set_shape_symbols(pshape_B, {E, F, G, H});
    set_shape_symbols(expected_shape, {A, F, C, H});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NUMPY);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), get_shape_symbols(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, symbols_equal_static_shape_broadcast_numpy) {
    // Static shape, the same symbols
    ov::PartialShape pshape_A{2, 1, 224, 1};
    ov::PartialShape pshape_B{2, 1, 1, 128};
    ov::PartialShape expected_shape{2, 1, 224, 128};

    // Equal symbols
    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();
    set_shape_symbols(pshape_A, {A, B, C, D});
    set_shape_symbols(pshape_B, {A, B, C, D});
    set_shape_symbols(expected_shape, {A, B, C, D});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NUMPY);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), get_shape_symbols(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, symbols_different_static_shape_broadcast_none) {
    // Static shape
    ov::PartialShape pshape_A{2, 3, 224, 128};
    ov::PartialShape pshape_B{2, 3, 224, 128};
    ov::PartialShape expected_shape{2, 3, 224, 128};

    // Different symbols
    auto symbols = set_shape_symbols(pshape_A);
    set_shape_symbols(pshape_B);
    set_shape_symbols(expected_shape, symbols);

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NONE);

    auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), get_shape_symbols(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, symbols_equal_static_shape_broadcast_none) {
    // Static shape
    ov::PartialShape pshape_A{2, 3, 224, 128};
    ov::PartialShape pshape_B{2, 3, 224, 128};
    ov::PartialShape expected_shape{2, 3, 224, 128};

    // Equal symbols
    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();
    set_shape_symbols(pshape_A, {A, B, C, D});
    set_shape_symbols(pshape_B, {A, B, C, D});
    set_shape_symbols(expected_shape, {A, B, C, D});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NONE);

    auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), get_shape_symbols(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, symbols_different_dynamic_shape_broadcast_none) {
    // Dynamic shape
    ov::PartialShape pshape_A{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1, 128)};
    ov::PartialShape pshape_B{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1, 128)};
    ov::PartialShape expected_shape{-1, 3, ov::Dimension(2, 224), ov::Dimension(1, 128)};

    // Different labels
    auto symbols = set_shape_symbols(pshape_A);
    set_shape_symbols(pshape_B);
    set_shape_symbols(expected_shape, symbols);

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NONE);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), get_shape_symbols(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, symbols_equal_dynamic_shape_broadcast_none) {
    // Dynamic shape
    ov::PartialShape pshape_A{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1, 128)};
    ov::PartialShape pshape_B{ov::Dimension(-1), ov::Dimension(3), ov::Dimension(2, 224), ov::Dimension(1, 128)};
    ov::PartialShape expected_shape{-1, 3, ov::Dimension(2, 224), ov::Dimension(1, 128)};

    // Equal symbols
    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>(),
         D = std::make_shared<ov::Symbol>();
    set_shape_symbols(pshape_A, {A, B, C, D});
    set_shape_symbols(pshape_B, {A, B, C, D});
    set_shape_symbols(expected_shape, {A, B, C, D});

    auto param_A = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<ov::op::v0::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NONE);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_symbols(out_shape), get_shape_symbols(expected_shape));
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
                            unsupported_element_type,
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

                            // Dimension symbols (static and dynamic)
                            symbols_a_dynamic_mixed_dims_broadcast_numpy,
                            symbols_b_dynamic_mixed_dims_broadcast_numpy,
                            symbols_different_interval_mixed_dims_broadcast_numpy,
                            symbols_different_interval_b_and_fully_dyn_a_broadcast_numpy,
                            symbols_different_interval_a_and_fully_dyn_b_broadcast_numpy,
                            symbols_equal_interval_dims_without_one_broadcast_numpy,
                            symbols_different_interval_dims_without_one_broadcast_numpy,
                            symbols_different_interval_batch_without_one_equivalence_table_broadcast_numpy,
                            symbols_different_fully_dynamic_batch_broadcast_numpy,
                            symbols_equal_fully_dynamic_batch_broadcast_numpy,
                            symbols_dyn_batch_a_broadcast_numpy,
                            symbols_dyn_batch_b_broadcast_numpy,
                            symbols_dyn_batch_and_higher_rank_a_broadcast_numpy,
                            symbols_dyn_batch_and_higher_rank_b_broadcast_numpy,
                            symbols_different_static_shape_broadcast_numpy,
                            symbols_equal_static_shape_broadcast_numpy,
                            symbols_different_static_shape_broadcast_none,
                            symbols_equal_static_shape_broadcast_none,
                            symbols_different_dynamic_shape_broadcast_none,
                            symbols_equal_dynamic_shape_broadcast_none);
