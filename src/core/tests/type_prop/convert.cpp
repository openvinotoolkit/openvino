// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/shape_of.hpp"

using namespace std;

TEST(type_prop, convert_deduce) {
    // Deduce type
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 4});
    auto c = make_shared<ov::op::v0::Convert>(param, ov::element::i32);
    ASSERT_EQ(c->get_element_type(), ov::element::i32);
    ASSERT_EQ(c->get_shape(), (ov::Shape{2, 3, 4}));
}

TEST(type_prop, convert_dynamic_value_and_symbol_propagation) {
    ov::Dimension marked_0 = ov::Dimension(3);
    auto A = std::make_shared<ov::Symbol>();
    marked_0.set_symbol(A);
    ov::PartialShape target_0 = ov::PartialShape{marked_0, 4};

    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto param_0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, target_0);
    auto shape_0 = make_shared<ov::op::v0::ShapeOf>(param_0);

    auto convert_0 = make_shared<ov::op::v0::Convert>(shape_0, ov::element::i8);
    auto convert_1 = make_shared<ov::op::v0::Convert>(convert_0, ov::element::i64);

    auto bc = make_shared<ov::op::v1::Broadcast>(param, convert_1);
    ASSERT_EQ(bc->get_shape(), (ov::Shape{3, 4}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    ASSERT_EQ(output_shape[0].get_symbol(), A);
    ASSERT_EQ(output_shape[1].get_symbol(), nullptr);
}
