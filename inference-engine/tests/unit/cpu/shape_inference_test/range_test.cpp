// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <range_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace std;

TEST(StaticShapeInferenceTest, Rangev4_i32) {
    auto start = make_shared<op::v0::Parameter>(element::i32, Shape{});
    auto stop = make_shared<op::v0::Parameter>(element::i32, Shape{});
    auto step = make_shared<op::v0::Parameter>(element::i32, Shape{});

    auto range = make_shared<op::v4::Range>(start, stop, step, element::i32);

    check_static_shape(range, {2, 0, -2}, {ov::StaticShape{1}});
    check_static_shape(range, {2, 0, -1}, {ov::StaticShape{2}});
    check_static_shape(range, {-19, 19, 1}, {ov::StaticShape{38}});
    check_static_shape(range, {-19, 19, 3}, {ov::StaticShape{13}});
    check_static_shape(range, {20, -19, 1}, {ov::StaticShape{0}});
}

TEST(StaticShapeInferenceTest, Rangev4_f32) {
    auto start = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto stop = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::v0::Parameter>(element::f32, Shape{});

    auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);

    check_static_shape(range, {0., 1., 0.25}, {ov::StaticShape{4}});
    check_static_shape(range, {-1., 1., 0.25}, {ov::StaticShape{8}});
    check_static_shape(range, {-1., 0.875, 0.25}, {ov::StaticShape{8}});
}
