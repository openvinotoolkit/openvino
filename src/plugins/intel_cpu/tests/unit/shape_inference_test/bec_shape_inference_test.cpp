// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/parameter.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

template <class TOp>
class BECStaticShapeInferenceTest : public OpStaticShapeInferenceTest<TOp> {
protected:
    void SetUp() override {
        this->output_shapes = StaticShapeVector(1);
    }
};

TYPED_TEST_SUITE_P(BECStaticShapeInferenceTest);

TYPED_TEST_P(BECStaticShapeInferenceTest, broadcast_none) {
    const auto a = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto b = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto op = this->make_op(a, b, op::AutoBroadcastType::NONE);

    this->input_shapes = {StaticShape{3, 4, 7, 5}, StaticShape{3, 4, 7, 5}};
    this->output_shapes = shape_inference(op.get(), this->input_shapes);

    ASSERT_EQ(this->output_shapes.front(), StaticShape({3, 4, 7, 5}));
}

TYPED_TEST_P(BECStaticShapeInferenceTest, broadcast_none_incompatible_shapes) {
    const auto a = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto b = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto op = this->make_op(a, b, op::AutoBroadcastType::NONE);

    this->input_shapes = {StaticShape{3, 4, 6, 5}, StaticShape{3, 1, 6, 1}};

    OV_EXPECT_THROW(shape_inference(op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Argument shapes are inconsistent."))
}

TYPED_TEST_P(BECStaticShapeInferenceTest, broadcast_numpy_equal_rank) {
    const auto a = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto b = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto op = this->make_op(a, b);

    this->input_shapes = {StaticShape{3, 1, 1, 5}, StaticShape{3, 1, 6, 1}};

    this->output_shapes = shape_inference(op.get(), this->input_shapes);

    ASSERT_EQ(this->output_shapes.front(), StaticShape({3, 1, 6, 5}));
}

TYPED_TEST_P(BECStaticShapeInferenceTest, broadcast_numpy_a_rank_higher) {
    const auto a = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto b = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    const auto op = this->make_op(a, b);

    this->input_shapes = {StaticShape{6, 5, 1, 8}, StaticShape{5, 6, 1}},

    this->output_shapes = shape_inference(op.get(), this->input_shapes);

    ASSERT_EQ(this->output_shapes.front(), StaticShape({6, 5, 6, 8}));
}

TYPED_TEST_P(BECStaticShapeInferenceTest, broadcast_numpy_b_rank_higher) {
    const auto a = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    const auto b = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto op = this->make_op(a, b);

    this->input_shapes = {StaticShape{5, 6, 1}, StaticShape{6, 5, 1, 8}},

    this->output_shapes = shape_inference(op.get(), this->input_shapes);

    ASSERT_EQ(this->output_shapes.front(), StaticShape({6, 5, 6, 8}));
}

TYPED_TEST_P(BECStaticShapeInferenceTest, broadcast_numpy_incompatible_shapes) {
    const auto a = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto b = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto op = this->make_op(a, b);

    this->input_shapes = {StaticShape{3, 4, 6, 6}, StaticShape{2, 4, 6, 6}};

    OV_EXPECT_THROW(shape_inference(op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Argument shapes are inconsistent."))
}

REGISTER_TYPED_TEST_SUITE_P(BECStaticShapeInferenceTest,
                            broadcast_none,
                            broadcast_none_incompatible_shapes,
                            broadcast_numpy_equal_rank,
                            broadcast_numpy_a_rank_higher,
                            broadcast_numpy_b_rank_higher,
                            broadcast_numpy_incompatible_shapes);

using BinaryOpTypes =
    Types<op::v1::Equal, op::v1::NotEqual, op::v1::Greater, op::v1::GreaterEqual, op::v1::Less, op::v1::LessEqual>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, BECStaticShapeInferenceTest, BinaryOpTypes);
