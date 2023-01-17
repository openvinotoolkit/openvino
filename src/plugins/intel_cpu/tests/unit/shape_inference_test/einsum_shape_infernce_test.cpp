// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/einsum.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

class EinsumStaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v7::Einsum> {};

TEST_F(EinsumStaticShapeInferenceTest, dot_product) {
    auto inputs = OutputVector(2, std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic()));
    auto op = make_op(inputs, "i,i->");

    check_static_shape(op.get(), {StaticShape{3}, StaticShape{3}}, {StaticShape{}});
}

TEST_F(EinsumStaticShapeInferenceTest, matmul) {
    auto inputs = OutputVector(2, std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic()));
    auto op = make_op(inputs, "ab,bc->ac");

    check_static_shape(op.get(), {StaticShape{2, 3}, StaticShape{3, 4}}, {StaticShape{2, 4}});
}

TEST_F(EinsumStaticShapeInferenceTest, trace) {
    auto I1 = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    auto op = make_op(OutputVector{I1}, "kii->k");

    check_static_shape(op.get(), {StaticShape{2, 3, 3}}, {StaticShape{2}});
}

TEST_F(EinsumStaticShapeInferenceTest, transpose) {
    auto I1 = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    auto op = make_op(OutputVector{I1}, "ijk->kij");

    check_static_shape(op.get(), {StaticShape{1, 2, 3}}, {StaticShape{3, 1, 2}});
}

TEST_F(EinsumStaticShapeInferenceTest, multi_matmul) {
    auto inputs = OutputVector(3, std::make_shared<op::v0::Parameter>(element::i32, ov::PartialShape::dynamic()));
    auto op = make_op(inputs, "ab,bcd,bc->ca");

    check_static_shape(op.get(), {StaticShape{2, 5}, StaticShape{5, 3, 6}, StaticShape{5, 3}}, {StaticShape{3, 2}});
}
