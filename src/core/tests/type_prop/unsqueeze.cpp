// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dimension_tracker.hpp>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, unsqueeze) {
    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 1, 4, 1, 8});
    auto axes_node = make_shared<ngraph::op::Constant>(element::u64, Shape{2}, vector<int64_t>{1, 2});
    auto unsqueeze = make_shared<op::v0::Unsqueeze>(param, axes_node);

    ASSERT_EQ(unsqueeze->get_element_type(), element::f32);
    ASSERT_EQ(unsqueeze->get_shape(), (Shape{4, 1, 1, 1, 4, 1, 8}));
}

TEST(type_prop, unsqueeze_dynamic) {
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(5));
    auto axes_node = make_shared<ngraph::op::Constant>(element::u64, Shape{2}, vector<int64_t>{1, 2});
    auto unsqueeze = make_shared<op::v0::Unsqueeze>(param, axes_node);

    ASSERT_EQ(unsqueeze->get_element_type(), element::f32);
    EXPECT_TRUE(unsqueeze->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(),
                                                                                1,
                                                                                1,
                                                                                Dimension::dynamic(),
                                                                                Dimension::dynamic(),
                                                                                Dimension::dynamic(),
                                                                                Dimension::dynamic()}));
}

TEST(type_prop, unsqueeze_incorrect_axes_shape) {
    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 1, 4, 1, 8});
    auto axes_node = make_shared<ngraph::op::Constant>(element::u64, Shape{1, 1, 1}, vector<int64_t>{1});

    try {
        auto unsqueeze = make_shared<op::v0::Unsqueeze>(param, axes_node);
        FAIL() << "Unsqueeze axes invalid rank not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Second input (axes) should not be of rank higher than 1");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, unsqueeze_empty_axes) {
    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 1, 4, 1, 8});
    auto axes_node = make_shared<ngraph::op::Constant>(element::u64, Shape{0}, vector<int64_t>{});
    try {
        auto unsqueeze = make_shared<op::v0::Unsqueeze>(param, axes_node);
        FAIL() << "Unsqueeze axes empty not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'axes' input is mandatory");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, unsqueeze_dynamic_axes) {
    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 1, 4, 1, 8});
    auto axes_node = make_shared<ngraph::op::Parameter>(element::u64, PartialShape::dynamic());

    auto unsqueeze = make_shared<op::v0::Unsqueeze>(param, axes_node);
    ASSERT_EQ(unsqueeze->get_element_type(), element::f32);
    ASSERT_EQ(unsqueeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, unsqueeze_dynamic_value_and_label_propagation) {
    Dimension marked_0 = Dimension(3);
    ov::DimensionTracker::set_label(marked_0, 10);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<op::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<op::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::ShapeOf>(param_0);

    const auto& et = element::i64;
    std::vector<int64_t> zero{0};
    const auto indices = std::make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto axis = std::make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto gather = std::make_shared<op::v7::Gather>(shape_0, indices, axis);

    const auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(gather, axis);

    auto bc = std::make_shared<op::v1::Broadcast>(param, unsqueeze);
    ASSERT_EQ(bc->get_shape(), (Shape{3}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    ASSERT_EQ(ov::DimensionTracker::get_label(output_shape[0]), 10);
}
