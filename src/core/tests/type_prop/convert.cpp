// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dimension_tracker.hpp>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, convert_deduce) {
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto c = make_shared<op::Convert>(param, element::i32);
    ASSERT_EQ(c->get_element_type(), element::i32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 3, 4}));
}

TEST(type_prop, convert_dynamic_value_and_label_propagation) {
    Dimension marked_0 = Dimension(3);
    ov::DimensionTracker::set_label(marked_0, 500);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = make_shared<op::Parameter>(element::f32, Shape{1});
    auto param_0 = make_shared<op::Parameter>(element::f32, target_0);
    auto shape_0 = make_shared<op::ShapeOf>(param_0);

    auto convert_0 = make_shared<op::Convert>(shape_0, element::i8);
    auto convert_1 = make_shared<op::Convert>(convert_0, element::i64);

    auto bc = make_shared<op::v1::Broadcast>(param, convert_1);
    ASSERT_EQ(bc->get_shape(), (Shape{3, 4}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    ASSERT_EQ(ov::DimensionTracker::get_label(output_shape[0]), 500);
    ASSERT_EQ(ov::DimensionTracker::get_label(output_shape[1]), 0);
}
