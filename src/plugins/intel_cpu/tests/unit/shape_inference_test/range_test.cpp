// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <range_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace std;

TEST(StaticShapeInferenceTest, Rangev4_i32) {
    auto start = make_shared<op::v0::Parameter>(element::i32, ov::PartialShape{});
    auto stop = make_shared<op::v0::Parameter>(element::i32, ov::PartialShape{});
    auto step = make_shared<op::v0::Parameter>(element::i32, ov::PartialShape{});

    auto range = make_shared<op::v4::Range>(start, stop, step, element::i32);

    check_static_shape(range.get(), {2, 0, -2}, {StaticShape{1}});
    check_static_shape(range.get(), {2, 0, -1}, {StaticShape{2}});
    check_static_shape(range.get(), {-19, 19, 1}, {StaticShape{38}});
    check_static_shape(range.get(), {-19, 19, 3}, {StaticShape{13}});
    check_static_shape(range.get(), {20, -19, 1}, {StaticShape{0}});
}

TEST(StaticShapeInferenceTest, Rangev4_f32) {
    auto start = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{});
    auto stop = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{});
    auto step = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{});

    auto range = make_shared<op::v4::Range>(start, stop, step, element::f32);

    check_static_shape(range.get(), {0., 1., 0.25}, {StaticShape{4}});
    check_static_shape(range.get(), {-1., 1., 0.25}, {StaticShape{8}});
    check_static_shape(range.get(), {-1., 0.875, 0.25}, {StaticShape{8}});
}
