// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, param_partial_rank_dynamic) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());

    auto& pshape = a->get_output_partial_shape(0);

    ASSERT_TRUE(pshape.is_dynamic());
    ASSERT_TRUE(pshape.rank().is_dynamic());
}

TEST(type_prop, param_partial_rank_static) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3, 4});

    auto& pshape = a->get_output_partial_shape(0);

    ASSERT_TRUE(pshape.is_dynamic());
    ASSERT_EQ(pshape.rank().get_length(), 4);
    ASSERT_TRUE(pshape[0].is_static() && pshape[0].get_length() == 2);
    ASSERT_TRUE(pshape[1].is_dynamic());
    ASSERT_TRUE(pshape[2].is_static() && pshape[2].get_length() == 3);
    ASSERT_TRUE(pshape[3].is_static() && pshape[3].get_length() == 4);
}

TEST(type_prop, param_layout) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    a->set_layout("NHWC");
    ASSERT_EQ(a->get_layout(), "NHWC");
    a->set_layout(ov::Layout());
    EXPECT_TRUE(a->get_layout().empty());
    EXPECT_EQ(a->get_output_tensor(0).get_rt_info().count(ov::LayoutAttribute::get_type_info_static()), 0);
}

TEST(type_prop, param_layout_empty) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    ASSERT_TRUE(a->get_layout().empty());
}

TEST(type_prop, param_layout_invalid) {
    auto a = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    a->get_output_tensor(0).get_rt_info()[ov::LayoutAttribute::get_type_info_static()] = "NCHW";  // incorrect way
    ASSERT_THROW(a->get_layout(), ov::Exception);
}
