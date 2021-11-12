
// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <reorg_yolo_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace std;

TEST(StaticShapeInferenceTest, ReorgYoloV0) {
    const auto in_shape = Shape{1, 64, 26, 26};
    size_t stride = 2;
    auto data_param = make_shared<op::v0::Parameter>(element::f32, in_shape);
    auto op = make_shared<op::v0::ReorgYolo>(data_param, stride);

    check_partial_shape(op, {ov::PartialShape{1, 64, 26, 26}}, {ov::PartialShape{1, 256, 13, 13}});
    check_static_shape(op, {ov::StaticShape{1, 64, 26, 26}}, {ov::StaticShape{1, 256, 13, 13}});
}