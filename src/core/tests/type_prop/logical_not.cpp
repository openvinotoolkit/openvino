// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

namespace {
void type_check(const ngraph::element::Type& type) {
    auto input = make_shared<op::Parameter>(type, Shape{1, 3, 6});
    auto logical_not = make_shared<op::v1::LogicalNot>(input);

    ASSERT_EQ(logical_not->get_element_type(), type);
}
}  // namespace

TEST(type_prop, logical_not_i32) {
    type_check(element::i32);
}

TEST(type_prop, logical_not_i64) {
    type_check(element::i64);
}

TEST(type_prop, logical_not_u32) {
    type_check(element::u32);
}

TEST(type_prop, logical_not_u64) {
    type_check(element::u64);
}

TEST(type_prop, logical_not_f16) {
    type_check(element::f16);
}

TEST(type_prop, logical_not_f32) {
    type_check(element::f32);
}

TEST(type_prop, logical_not_shape_inference) {
    auto input = make_shared<op::Parameter>(element::boolean, Shape{1, 3, 6});
    auto logical_not = make_shared<op::v1::LogicalNot>(input);
    ASSERT_EQ(logical_not->get_shape(), (Shape{1, 3, 6}));
}
