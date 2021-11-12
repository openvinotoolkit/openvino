
// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <region_yolo_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace std;

TEST(StaticShapeInferenceTest, RegionYoloV0) {
    auto inputs = make_shared<op::v0::Parameter>(element::f32, Shape{1, 125, 13, 13});
    auto op = make_shared<op::v0::RegionYolo>(inputs, 0, 0, 0, true, std::vector<int64_t>{}, 0, 1);

    check_partial_shape(op, {ov::PartialShape{1, 125, 13, 13}}, {ov::PartialShape{1 * 125, 13, 13}});
    check_static_shape(op, {ov::StaticShape{1, 125, 13, 13}}, {ov::StaticShape{1 * 125, 13, 13}});
}