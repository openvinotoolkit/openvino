// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, tile) {
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8, 10});
    auto param1 = op::Constant::create(element::i64, Shape{3}, {3, 4, 1});
    auto top = make_shared<op::v0::Tile>(param0, param1);
    ASSERT_EQ(top->get_element_type(), element::f32);
    ASSERT_EQ(top->get_shape(), (Shape{18, 32, 10}));
}

TEST(type_prop, tile_small_data_rank) {
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{8, 10});
    auto param1 = op::Constant::create(element::i64, Shape{3}, {3, 4, 1});
    auto top = make_shared<op::v0::Tile>(param0, param1);
    ASSERT_EQ(top->get_element_type(), element::f32);
    ASSERT_EQ(top->get_shape(), (Shape{3, 32, 10}));
}

TEST(type_prop, tile_few_repeats) {
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8, 10});
    auto param1 = op::Constant::create(element::i64, Shape{2}, {4, 1});
    auto top = make_shared<op::v0::Tile>(param0, param1);
    ASSERT_EQ(top->get_element_type(), element::f32);
    ASSERT_EQ(top->get_shape(), (Shape{6, 32, 10}));
}

TEST(type_prop, tile_few_repeats_dyn_input) {
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{6, Dimension(8, 10), 10});
    auto param1 = op::Constant::create(element::i64, Shape{2}, {4, 1});
    auto top = make_shared<op::v0::Tile>(param0, param1);
    ASSERT_EQ(top->get_element_type(), element::f32);
    ASSERT_EQ(top->get_output_partial_shape(0), (PartialShape{6, Dimension(32, 40), 10}));
}

TEST(type_prop, tile_out_rank_from_repeats) {
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8, 10});
    auto param1 = make_shared<op::Parameter>(element::i32, Shape{5});
    auto top = make_shared<op::v0::Tile>(param0, param1);
    ASSERT_EQ(top->get_element_type(), element::f32);
    ASSERT_EQ(top->get_output_partial_shape(0).size(), 5);
}
