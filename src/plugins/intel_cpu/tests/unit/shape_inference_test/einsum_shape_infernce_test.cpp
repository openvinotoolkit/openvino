// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gmock/gmock.h>

#include "openvino/op/einsum.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using testing::ElementsAre;

class EinsumStaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v7::Einsum> {};

TEST_F(EinsumStaticShapeInferenceTest, dot_product) {
    auto inputs = OutputVector(2, std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic()));
    auto op = make_op(inputs, "i,i->");

    output_shapes = shape_inference(op.get(), StaticShapeVector{{3}, {3}});
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{}));
}

TEST_F(EinsumStaticShapeInferenceTest, matmul) {
    auto inputs = OutputVector(2, std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic()));
    auto op = make_op(inputs, "ab,bc->ac");

    output_shapes = shape_inference(op.get(), StaticShapeVector{{2, 3}, {3, 4}});
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{2, 4}));
}

TEST_F(EinsumStaticShapeInferenceTest, trace) {
    auto I1 = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    auto op = make_op(OutputVector{I1}, "kii->k");

    output_shapes = shape_inference(op.get(), StaticShapeVector{{2, 3, 3}});
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{2}));
}

TEST_F(EinsumStaticShapeInferenceTest, transpose) {
    auto I1 = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    auto op = make_op(OutputVector{I1}, "ijk->kij");

    output_shapes = shape_inference(op.get(), StaticShapeVector{{1, 2, 3}});
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{3, 1, 2}));
}

TEST_F(EinsumStaticShapeInferenceTest, multi_matmul) {
    auto inputs = OutputVector(3, std::make_shared<op::v0::Parameter>(element::i32, ov::PartialShape::dynamic()));
    auto op = make_op(inputs, "ab,bcd,bc->ca");

    output_shapes = shape_inference(op.get(), StaticShapeVector{{2, 5}, {5, 3, 6}, {5, 3}});
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{3, 2}));
}
