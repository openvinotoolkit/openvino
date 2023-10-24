// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/shape_of.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/broadcast.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, shape_of_v0) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto so = make_shared<op::v0::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_output_partial_shape(0).to_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_et_dynamic_v0) {
    auto a = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto so = make_shared<op::v0::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_output_partial_shape(0).to_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_rank_static_dynamic_v0) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32,
                                                PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), 4});
    auto so = make_shared<op::v0::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_output_partial_shape(0).to_shape(), Shape{4});
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
    ASSERT_EQ(so->get_output_partial_shape(0).to_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_et_dynamic_v3) {
    auto a = make_shared<ov::op::v0::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto so = make_shared<op::v3::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_output_partial_shape(0).to_shape(), Shape{4});
}

TEST(type_prop, shape_of_partial_rank_static_dynamic_v3) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32,
                                                PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), 4});
    auto so = make_shared<op::v3::ShapeOf>(a);

    ASSERT_EQ(so->get_output_element_type(0), element::i64);
    ASSERT_EQ(so->get_output_partial_shape(0).to_shape(), Shape{4});
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
    ASSERT_EQ(so->get_output_partial_shape(0).to_shape(), Shape{4});
}

TEST(type_prop, shape_of_1_dynamic_value_and_label_propagation) {
    Dimension marked_0 = Dimension(3);
    ov::DimensionTracker::set_label(marked_0, 10);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::v0::ShapeOf>(param_0);

    auto bc = std::make_shared<op::v1::Broadcast>(param, shape_0);
    ASSERT_EQ(bc->get_output_partial_shape(0).to_shape(), (Shape{3, 4}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    ASSERT_EQ(ov::DimensionTracker::get_label(output_shape[0]), 10);
}

TEST(type_prop, shape_of_3_dynamic_value_and_label_propagation) {
    Dimension marked_0 = Dimension(3);
    ov::DimensionTracker::set_label(marked_0, 10);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::v3::ShapeOf>(param_0);

    auto bc = std::make_shared<op::v1::Broadcast>(param, shape_0);
    ASSERT_EQ(bc->get_output_partial_shape(0).to_shape(), (Shape{3, 4}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    ASSERT_EQ(ov::DimensionTracker::get_label(output_shape[0]), 10);
}
