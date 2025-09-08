// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/shape_of.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/broadcast.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, shape_of_v0) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto so = make_shared<op::v0::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_et_dynamic_v0) {
    auto a = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto so = make_shared<op::v0::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_rank_static_dynamic_v0) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32,
                                                PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), 4});
    auto so = make_shared<op::v0::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_rank_dynamic_v0) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto so = make_shared<op::v0::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_TRUE(so->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, shape_of_v3) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto so = make_shared<op::v3::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_et_dynamic_v3) {
    auto a = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto so = make_shared<op::v3::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_rank_static_dynamic_v3) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32,
                                                PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), 4});
    auto so = make_shared<op::v3::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_rank_dynamic_v3) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto so = make_shared<op::v3::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_TRUE(so->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, shape_of_output_type_v3) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto so = make_shared<op::v3::ShapeOf>(a, element::i32);
    try {
        auto sx = make_shared<op::v3::ShapeOf>(a, element::i8);
        FAIL() << "Invalid output_type not detected";
    } catch (const NodeValidationFailure&) {
    } catch (...) {
        FAIL() << "Node validation error not thrown";
    }
    try {
        auto sx = make_shared<op::v3::ShapeOf>(a, element::i16);
        FAIL() << "Invalid output_type not detected";
    } catch (const NodeValidationFailure&) {
    } catch (...) {
        FAIL() << "Node validation error not thrown";
    }
    try {
        auto sx = make_shared<op::v3::ShapeOf>(a, element::f32);
        FAIL() << "Invalid output_type not detected";
    } catch (const NodeValidationFailure&) {
    } catch (...) {
        FAIL() << "Node validation error not thrown";
    }

    ASSERT_EQ(so->get_output_element_type(0), element::i32);
    ASSERT_EQ(so->get_shape(), Shape{4});
}

TEST(type_prop, shape_of_1_dynamic_value_and_label_propagation) {
    Dimension marked_0 = Dimension(3);
    auto symbol = std::make_shared<Symbol>();
    marked_0.set_symbol(symbol);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::v0::ShapeOf>(param_0);

    auto bc = std::make_shared<op::v1::Broadcast>(param, shape_0);
    ASSERT_EQ(bc->get_shape(), (Shape{3, 4}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    ASSERT_EQ(output_shape[0].get_symbol(), symbol);
}

TEST(type_prop, shape_of_3_dynamic_value_and_label_propagation) {
    Dimension marked_0 = Dimension(3);
    auto symbol = std::make_shared<Symbol>();
    marked_0.set_symbol(symbol);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::v3::ShapeOf>(param_0);

    auto bc = std::make_shared<op::v1::Broadcast>(param, shape_0);
    ASSERT_EQ(bc->get_shape(), (Shape{3, 4}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    ASSERT_EQ(output_shape[0].get_symbol(), symbol);
}

TEST(type_prop, shape_of_3_dynamic_value_propagation_out_i32) {
    constexpr auto i32_max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    constexpr auto bound = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 10;

    auto param = std::make_shared<ov::op::v0::Parameter>(
        element::f32,
        PartialShape{{2, bound}, {3, -1}, {i32_max + 1, bound}, {i32_max, -1}, {1, 1021}});
    auto op = std::make_shared<op::v3::ShapeOf>(param, element::i32);

    auto bc_param = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto bc = std::make_shared<op::v1::Broadcast>(bc_param, op);

    EXPECT_EQ(bc->get_output_partial_shape(0), PartialShape({{2, -1}, {3, -1}, -1, -1, {1, 1021}}));
}
