// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <einsum_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, Einsum1) {
    auto I1 = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    auto I2 = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    auto O = std::make_shared<op::v7::Einsum>(OutputVector{I1, I2}, "i,i->");

    check_static_shape(O.get(), {StaticShape{3}, StaticShape{3}}, {StaticShape{}});
}

TEST(StaticShapeInferenceTest, Einsum2) {
    auto I1 = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    auto I2 = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    auto O = std::make_shared<op::v7::Einsum>(OutputVector{I1, I2}, "ab,bc->ac");

    check_static_shape(O.get(), {StaticShape{2, 3}, StaticShape{3, 4}}, {StaticShape{2, 4}});
}

TEST(StaticShapeInferenceTest, Einsum3) {
    auto I1 = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    auto O = std::make_shared<op::v7::Einsum>(OutputVector{I1}, "kii->k");

    check_static_shape(O.get(), {StaticShape{2, 3, 3}}, {StaticShape{2}});
}

TEST(StaticShapeInferenceTest, Einsum4) {
    auto I1 = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    auto O = std::make_shared<op::v7::Einsum>(OutputVector{I1}, "ijk->kij");

    check_static_shape(O.get(), {StaticShape{1, 2, 3}}, {StaticShape{3, 1, 2}});
}

TEST(StaticShapeInferenceTest, Einsum5) {
    auto I1 = std::make_shared<op::v0::Parameter>(element::i32, ov::PartialShape::dynamic());
    auto I2 = std::make_shared<op::v0::Parameter>(element::i32, ov::PartialShape::dynamic());
    auto I3 = std::make_shared<op::v0::Parameter>(element::i32, ov::PartialShape::dynamic());
    auto O = std::make_shared<op::v7::Einsum>(OutputVector{I1, I2, I3}, "ab,bcd,bc->ca");

    check_static_shape(O.get(),
                       {StaticShape{2, 5}, StaticShape{5, 3, 6}, StaticShape{5, 3}},
                       {StaticShape{3, 2}});
}
