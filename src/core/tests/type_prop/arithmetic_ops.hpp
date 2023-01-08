//*****************************************************************************
// Copyright 2017-2022 Intel Corporation
//  Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <vector>

#include "dimension_tracker.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;
using namespace testing;

template <class T>
class ArithmeticOperator : public testing::Test {};

TYPED_TEST_SUITE_P(ArithmeticOperator);

TYPED_TEST_P(ArithmeticOperator, default_constructor) {
    auto A = std::make_shared<op::Parameter>(element::f32, PartialShape{-1, 4, 1, 6, Dimension(1, 6), Dimension(2, 6)});
    auto B = std::make_shared<op::Parameter>(element::f32, PartialShape{-1, 1, 5, 6, Dimension(5, 8), Dimension(5, 8)});

    const auto op = std::make_shared<TypeParam>();

    op->set_argument(0, A);
    op->set_argument(1, B);

    auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NONE);
    op->set_autob(autob);
    EXPECT_EQ(op->get_autob(), op::AutoBroadcastType::NONE);
    ASSERT_THROW(op->validate_and_infer_types(), NodeValidationFailure);

    autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY);
    op->set_autob(autob);
    EXPECT_EQ(op->get_autob(), op::AutoBroadcastType::NUMPY);

    op->validate_and_infer_types();

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 4, 5, 6, Dimension(5, 8), Dimension(5, 6)}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_2D) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 2}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_4D) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 3});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 3});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 2, 3, 3}));
}

TYPED_TEST_P(ArithmeticOperator, default_autobroadcast) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 2}));
    EXPECT_EQ(op->get_autob(), op::AutoBroadcastType::NUMPY);
}

TYPED_TEST_P(ArithmeticOperator, no_autobroadcast) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});

    const auto op = std::make_shared<TypeParam>(A, B, op::AutoBroadcastType::NONE);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 2}));
    EXPECT_EQ(op->get_autob(), op::AutoBroadcastType::NONE);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_4D_x_scalar_numpy_broadcast) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{1});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_4D_x_1D_numpy_broadcast) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{5});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_2D_x_4D_numpy_broadcast) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{4, 5});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_3D_x_4D_numpy_broadcast) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{1, 4, 5});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{2, 3, 1, 1});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_4D_x_3D_numpy_broadcast) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{8, 1, 6, 1});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{7, 1, 5});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), (Shape{8, 7, 6, 5}));
    EXPECT_EQ(op->get_autob(), op::AutoBroadcastType::NUMPY);
}

TYPED_TEST_P(ArithmeticOperator, static_shape_pdpd_doc_examples) {
    // TODO: PDPD broadcast review, ticket: 93618
    {
        auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto B = std::make_shared<op::Parameter>(element::f32, Shape{3, 4});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), element::f32);
        EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
    }
    {
        auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto B = std::make_shared<op::Parameter>(element::f32, Shape{3, 1});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), element::f32);
        EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
    }
    {
        auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto B = std::make_shared<op::Parameter>(element::f32, Shape{});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), element::f32);
        EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
    }
    {
        auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto B = std::make_shared<op::Parameter>(element::f32, Shape{5});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 3);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), element::f32);
        EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
        EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
    }
}

TYPED_TEST_P(ArithmeticOperator, static_shape_4D_x_4D_equal_pdpd_broadcast) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{8, 1, 6, 5});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{8, 1, 6, 5});

    const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD);
    const auto op = std::make_shared<TypeParam>(A, B, autob);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), (Shape{8, 1, 6, 5}));
    EXPECT_EQ(op->get_autob().m_type, op::AutoBroadcastType::PDPD);
}

TYPED_TEST_P(ArithmeticOperator, incompatible_element_types) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 3});
    auto B = std::make_shared<op::Parameter>(element::i32, Shape{2, 2, 3, 3});

    ASSERT_THROW(const auto unused = std::make_shared<TypeParam>(A, B), ngraph::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, incompatible_boolean_type) {
    auto A = std::make_shared<op::Parameter>(element::boolean, Shape{2, 2, 3, 3});
    auto B = std::make_shared<op::Parameter>(element::boolean, Shape{2, 2, 3, 3});

    ASSERT_THROW(const auto unused = std::make_shared<TypeParam>(A, B), ngraph::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_1D_x_1D_incompatible) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{3});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{4});

    ASSERT_THROW(const auto unused = std::make_shared<TypeParam>(A, B), ngraph::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_3D_x_3D_incompatible) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{3, 5, 6});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{4, 10, 12});

    ASSERT_THROW(const auto unused = std::make_shared<TypeParam>(A, B), ngraph::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_5D_x_5D_incompatible) {
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{389, 112, 12});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{389, 112, 19});

    ASSERT_THROW(const auto unused = std::make_shared<TypeParam>(A, B), ngraph::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, fully_dynamic_shape_broadcast_numpy) {
    auto param = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY);
    const auto op = std::make_shared<TypeParam>(param, param, autob);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST_P(ArithmeticOperator, fully_dynamic_shape_broadcast_none) {
    auto param = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::NONE);
    const auto op = std::make_shared<TypeParam>(param, param, autob);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST_P(ArithmeticOperator, fully_dynamic_shape_broadcast_pdpd) {
    auto param = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD);
    const auto op = std::make_shared<TypeParam>(param, param, autob);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_3D) {
    Dimension dynamic = Dimension::dynamic();
    auto A = std::make_shared<op::Parameter>(element::f32, PartialShape{dynamic, dynamic, 6});
    auto B = std::make_shared<op::Parameter>(element::f32, PartialShape{dynamic, dynamic, 6});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{dynamic, dynamic, 6}));
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_5D) {
    Dimension dynamic = Dimension::dynamic();
    auto A = std::make_shared<op::Parameter>(element::f32, PartialShape{dynamic, 4, dynamic, dynamic, 6});
    auto B = std::make_shared<op::Parameter>(element::f32, PartialShape{dynamic, 4, dynamic, dynamic, 6});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{dynamic, 4, dynamic, dynamic, 6}));
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_intervals_broadcast_none) {
    auto A = std::make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension(1, 3), Dimension(2, 7), Dimension(6, -1), Dimension(-1, 6), -1, 8});
    auto B = std::make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension(1, 3), Dimension(2, 7), Dimension(6, -1), Dimension(-1, 6), -1, 8});

    const auto op = std::make_shared<TypeParam>(A, B, op::AutoBroadcastType::NONE);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension(1, 3), Dimension(2, 7), Dimension(6, -1), Dimension(-1, 6), -1, 8}));
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_intervals_equal_rank_broadcast_numpy) {
    // Equal rank
    auto A = std::make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension(1, 3), Dimension(1, 3), Dimension(1, 3), Dimension(4, 8), -1, 1, -1, 1, 3});
    auto B = std::make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension(1, 3), Dimension(2, 7), -1, 1, Dimension(1, 3), Dimension(4, 8), -1, 1, 3});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension(1, 3), Dimension(2, 7), -1, Dimension(4, 8), -1, Dimension(4, 8), -1, 1, 3}));
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_intervals_a_rank_smaller_broadcast_numpy) {
    // `A` rank smaller
    auto A =
        std::make_shared<op::Parameter>(element::f32, PartialShape{Dimension(1, 3), Dimension(4, 8), -1, 1, -1, 1, 3});
    auto B = std::make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension(1, 3), Dimension(2, 7), -1, 1, Dimension(1, 3), Dimension(4, 8), -1, 1, 3});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension(1, 3), Dimension(2, 7), -1, Dimension(4, 8), -1, Dimension(4, 8), -1, 1, 3}));
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_intervals_b_rank_smaller_broadcast_numpy) {
    // `B` rank smaller
    auto A = std::make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension(1, 3), Dimension(2, 7), -1, 1, Dimension(1, 3), Dimension(4, 8), -1, 1, 3});
    auto B =
        std::make_shared<op::Parameter>(element::f32, PartialShape{Dimension(1, 3), Dimension(4, 8), -1, 1, -1, 1, 3});

    const auto op = std::make_shared<TypeParam>(A, B);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension(1, 3), Dimension(2, 7), -1, Dimension(4, 8), -1, Dimension(4, 8), -1, 1, 3}));
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_intervals_broadcast_pdpd) {
    // TODO: PDPD broadcast review, ticket: 93618
    {  // Equal rank
        auto A = std::make_shared<op::Parameter>(
            element::f32,
            PartialShape{Dimension(1, 3), Dimension(2, 7), Dimension(1, 6), /* Dimension(6, -1), */ -1, 8});
        auto B = std::make_shared<op::Parameter>(element::f32,
                                                 PartialShape{Dimension(1, 3), Dimension(2, 7), 1, /* 1, */ -1, 8});

        const auto op = std::make_shared<TypeParam>(A, B, op::AutoBroadcastType::PDPD);

        EXPECT_EQ(op->get_element_type(), element::f32);
        EXPECT_EQ(op->get_output_partial_shape(0),
                  (PartialShape{Dimension(1, 3), Dimension(2, 7), Dimension(1, 6), /* Dimension(6, -1), */ -1, 8}));
    }
    {  // `A` fully dynamic dimension, axis = 0
        auto A = std::make_shared<op::Parameter>(element::f32, PartialShape{-1, -1});
        auto B = std::make_shared<op::Parameter>(element::f32, PartialShape{Dimension(1, 3), Dimension(2, 7)});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 0);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), element::f32);
        EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, -1}));
    }
    {  // `B` fully dynamic dimension, axis = 0
        auto A = std::make_shared<op::Parameter>(element::f32, PartialShape{Dimension(1, 3), Dimension(2, 7)});
        auto B = std::make_shared<op::Parameter>(element::f32, PartialShape{-1, -1});

        const auto autob = op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 0);
        const auto op = std::make_shared<TypeParam>(A, B, autob);

        EXPECT_EQ(op->get_element_type(), element::f32);
        EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension(1, 3), Dimension(2, 7)}));
    }
}

TYPED_TEST_P(ArithmeticOperator, labels_a_dynamic_mixed_dims_broadcast_numpy) {
    // All dimensions of A have labels, B without labels
    PartialShape pshape_A{Dimension(-1), Dimension(3), Dimension(1), Dimension(2, 128)};
    PartialShape pshape_B{Dimension(-1), Dimension(3), Dimension(2, 224), Dimension(1)};

    PartialShape expected_shape = {-1, 3, Dimension(2, 224), Dimension(2, 128)};

    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(expected_shape, {10, 11, 0, 13});

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_b_dynamic_mixed_dims_broadcast_numpy) {
    // All dimensions of B have labels, A without labels
    PartialShape pshape_A{Dimension(-1), Dimension(3), Dimension(1), Dimension(2, 128)};
    PartialShape pshape_B{Dimension(-1), Dimension(3), Dimension(2, 224), Dimension(1)};

    PartialShape expected_shape = {-1, 3, Dimension(2, 224), Dimension(2, 128)};

    set_shape_labels(pshape_B, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {20, 21, 22, 0});

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_different_interval_mixed_dims_broadcast_numpy) {
    // Both params have dimensions with different labels
    PartialShape pshape_A{Dimension(-1), Dimension(3), Dimension(1), Dimension(2, 128)};
    PartialShape pshape_B{Dimension(-1), Dimension(3), Dimension(2, 224), Dimension(1)};

    PartialShape expected_shape = {-1, 3, Dimension(2, 224), Dimension(2, 128)};

    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(pshape_B, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {0, 21, 22, 13});

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_different_interval_b_and_fully_dyn_a_broadcast_numpy) {
    // Both params have dimension labels, output has label B
    Dimension dim_0_A = Dimension(-1);
    Dimension dim_0_B = Dimension(2, 4);

    ov::DimensionTracker::set_label(dim_0_A, 10);
    ov::DimensionTracker::set_label(dim_0_B, 20);

    PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};
    PartialShape expected_shape = {Dimension(2, 4), 3, 224, 224};
    std::vector<size_t> expected_labels{20, 0, 0, 0};

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_different_interval_a_and_fully_dyn_b_broadcast_numpy) {
    // Both params have dimension labels, output has label A
    Dimension dim_0_A = Dimension(2, 4);
    Dimension dim_0_B = Dimension(-1);

    ov::DimensionTracker::set_label(dim_0_A, 10);
    ov::DimensionTracker::set_label(dim_0_B, 20);

    PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};
    PartialShape expected_shape = {Dimension(2, 4), 3, 224, 224};
    std::vector<size_t> expected_labels{10, 0, 0, 0};

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_equal_interval_dims_without_one_broadcast_numpy) {
    // Both params have dynamic interval dimension the same labels
    PartialShape pshape_A{Dimension(2, 4), Dimension(8, 16), Dimension(8, 16), Dimension(8, 16)};
    PartialShape pshape_B{Dimension(2, 4), Dimension(4, 12), Dimension(10, 12), Dimension(16, 24)};

    PartialShape expected_shape = {Dimension(2, 4), Dimension(8, 12), Dimension(10, 12), 16};

    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(pshape_B, {10, 11, 12, 13});
    set_shape_labels(expected_shape, {10, 11, 12, 13});

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_different_interval_dims_without_one_broadcast_numpy) {
    // Both params have dynamic interval dimension different labels
    PartialShape pshape_A{Dimension(2, 4), Dimension(8, 16), Dimension(8, 16), Dimension(8, 16)};
    PartialShape pshape_B{Dimension(2, 4), Dimension(4, 12), Dimension(10, 12), Dimension(16, 24)};

    PartialShape expected_shape = {Dimension(2, 4), Dimension(8, 12), Dimension(10, 12), 16};
    std::vector<size_t> expected_labels{20, 21, 22, 23};

    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(pshape_B, {20, 21, 22, 23});

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_different_interval_batch_without_one_equivalence_table_broadcast_numpy) {
    // Both params have dynamic interval dimension different labels, use table of equivalence
    auto table_of_equivalence = std::make_shared<ov::TableOfEquivalence>();
    ov::DimensionTracker dim_tracker(table_of_equivalence);

    Dimension dim_0_A = Dimension(2, 4);
    Dimension dim_0_B = Dimension(2, 4);

    dim_tracker.set_up_for_tracking(dim_0_A, 10);
    dim_tracker.set_up_for_tracking(dim_0_B, 20);

    PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    PartialShape expected_shape = {Dimension(2, 4), 3, 224, 224};
    std::vector<size_t> expected_labels{20, 0, 0, 0};

    auto eq_table = table_of_equivalence->get_equivalence_table();
    EXPECT_EQ(eq_table[ov::DimensionTracker::get_label(dim_0_A)], std::unordered_set<size_t>{20});
    EXPECT_EQ(eq_table[ov::DimensionTracker::get_label(dim_0_B)], std::unordered_set<size_t>{10});

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_different_fully_dynamic_batch_broadcast_numpy) {
    // Both params have fully dynamic dimension and different abels
    Dimension dim_0_A = Dimension(-1);
    Dimension dim_0_B = Dimension(-1);

    ov::DimensionTracker::set_label(dim_0_A, 10);
    ov::DimensionTracker::set_label(dim_0_B, 20);

    PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};
    PartialShape expected_shape = {-1, 3, 224, 224};
    std::vector<size_t> expected_labels{0, 0, 0, 0};

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_equal_fully_dynamic_batch_broadcast_numpy) {
    // Both params have fully dynamic dimension and the same labels
    Dimension dim_0_A = Dimension(-1);
    Dimension dim_0_B = Dimension(-1);

    ov::DimensionTracker::set_label(dim_0_A, 10);
    ov::DimensionTracker::set_label(dim_0_B, 10);

    PartialShape pshape_A = {dim_0_A, 3, 224, 1}, pshape_B = {dim_0_B, 3, 1, 224};
    PartialShape expected_shape = {-1, 3, 224, 224};
    std::vector<size_t> expected_labels{10, 0, 0, 0};

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_dyn_batch_a_broadcast_numpy) {
    Dimension b = -1;
    ov::DimensionTracker::set_label(b, 10);
    PartialShape A = {b, 3, 224, 224}, B = {1, 3, 1, 1};
    PartialShape expected_shape{b, 3, 224, 224};

    std::vector<size_t> expected_labels{10, 0, 0, 0};

    auto param_A = std::make_shared<op::Parameter>(element::f64, A);
    auto param_B = std::make_shared<op::Parameter>(element::f64, B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_dyn_batch_b_broadcast_numpy) {
    Dimension b = -1;
    ov::DimensionTracker::set_label(b, 10);
    PartialShape B = {b, 3, 224, 224}, A = {1, 3, 1, 1};
    PartialShape expected_shape{b, 3, 224, 224};

    std::vector<size_t> expected_labels{10, 0, 0, 0};

    auto param_A = std::make_shared<op::Parameter>(element::f64, A);
    auto param_B = std::make_shared<op::Parameter>(element::f64, B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_dyn_batch_and_higher_rank_a_broadcast_numpy) {
    Dimension b = -1;
    ov::DimensionTracker::set_label(b, 10);

    PartialShape pshape_A{b, -1, -1, -1};
    PartialShape pshape_B{3, 1, 1};
    PartialShape expected_shape{b, 3, -1, -1};

    std::vector<size_t> expected_labels{10, 0, 0, 0};

    auto param_A = std::make_shared<op::Parameter>(element::f64, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f64, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_dyn_batch_and_higher_rank_b_broadcast_numpy) {
    Dimension b = -1;
    ov::DimensionTracker::set_label(b, 10);

    PartialShape pshape_A{3, 1, 1};
    PartialShape pshape_B{b, -1, -1, -1};
    PartialShape expected_shape{b, 3, -1, -1};

    std::vector<size_t> expected_labels{10, 0, 0, 0};

    auto param_A = std::make_shared<op::Parameter>(element::f64, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f64, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), expected_labels);
}

TYPED_TEST_P(ArithmeticOperator, labels_different_static_shape_broadcast_numpy) {
    // Static shape, different labels
    PartialShape pshape_A{Dimension(2), Dimension(1), Dimension(224), Dimension(1)};
    PartialShape pshape_B{Dimension(2), Dimension(1), Dimension(1), Dimension(128)};
    PartialShape expected_shape{2, 1, 224, 128};

    // Different labels
    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(pshape_B, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {20, 21, 12, 23});

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NUMPY);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_equal_static_shape_broadcast_numpy) {
    // Static shape, the same labels
    PartialShape pshape_A{2, 1, 224, 1};
    PartialShape pshape_B{2, 1, 1, 128};
    PartialShape expected_shape{2, 1, 224, 128};

    // Equal labels
    set_shape_labels(pshape_A, {30, 31, 32, 33});
    set_shape_labels(pshape_B, {30, 31, 32, 33});
    set_shape_labels(expected_shape, {30, 31, 32, 33});

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NUMPY);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_different_static_shape_broadcast_none) {
    // Static shape
    PartialShape pshape_A{2, 3, 224, 128};
    PartialShape pshape_B{2, 3, 224, 128};
    PartialShape expected_shape{2, 3, 224, 128};

    // Different labels
    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(pshape_B, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {20, 21, 22, 23});

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NONE);

    auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_equal_static_shape_broadcast_none) {
    // Static shape
    PartialShape pshape_A{2, 3, 224, 128};
    PartialShape pshape_B{2, 3, 224, 128};
    PartialShape expected_shape{2, 3, 224, 128};

    // Equal labels
    set_shape_labels(pshape_A, {30, 31, 32, 33});
    set_shape_labels(pshape_B, {30, 31, 32, 33});
    set_shape_labels(expected_shape, {30, 31, 32, 33});

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NONE);

    auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_different_dynamic_shape_broadcast_none) {
    // Dynamic shape
    PartialShape pshape_A{Dimension(-1), Dimension(3), Dimension(2, 224), Dimension(1, 128)};
    PartialShape pshape_B{Dimension(-1), Dimension(3), Dimension(2, 224), Dimension(1, 128)};
    PartialShape expected_shape{-1, 3, Dimension(2, 224), Dimension(1, 128)};

    // Different labels
    set_shape_labels(pshape_A, {10, 11, 12, 13});
    set_shape_labels(pshape_B, {20, 21, 22, 23});
    set_shape_labels(expected_shape, {20, 21, 22, 23});

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
    const auto op = std::make_shared<TypeParam>(param_A, param_B, op::AutoBroadcastType::NONE);

    const auto out_shape = op->get_output_partial_shape(0);

    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_EQ(get_shape_labels(out_shape), get_shape_labels(expected_shape));
}

TYPED_TEST_P(ArithmeticOperator, labels_equal_dynamic_shape_broadcast_none) {
    // Dynamic shape
    PartialShape pshape_A{Dimension(-1), Dimension(3), Dimension(2, 224), Dimension(1, 128)};
    PartialShape pshape_B{Dimension(-1), Dimension(3), Dimension(2, 224), Dimension(1, 128)};
    PartialShape expected_shape{-1, 3, Dimension(2, 224), Dimension(1, 128)};

    // Equal labels
    set_shape_labels(pshape_A, {30, 31, 32, 33});
    set_shape_labels(pshape_B, {30, 31, 32, 33});
    set_shape_labels(expected_shape, {30, 31, 32, 33});

    auto param_A = std::make_shared<op::Parameter>(element::f32, pshape_A);
    auto param_B = std::make_shared<op::Parameter>(element::f32, pshape_B);
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
                            static_shape_4D_x_4D_equal_pdpd_broadcast,
                            incompatible_element_types,
                            incompatible_boolean_type,
                            shape_inference_1D_x_1D_incompatible,
                            shape_inference_3D_x_3D_incompatible,
                            shape_inference_5D_x_5D_incompatible,

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
