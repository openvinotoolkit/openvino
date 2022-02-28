// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, region_yolo_v2) {
    const size_t num = 5;
    const size_t coords = 4;
    const size_t classes = 20;
    const size_t batch = 1;
    const size_t channels = 125;
    const size_t width = 13;
    const size_t height = 13;
    const std::vector<int64_t> mask{0, 1, 2};
    const int axis = 1;
    const int end_axis = 3;
    const auto in_shape = Shape{batch, channels, width, height};
    auto data_param = make_shared<op::Parameter>(element::f32, in_shape);
    auto region_yolo = make_shared<op::v0::RegionYolo>(data_param, coords, classes, num, true, mask, axis, end_axis);

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{batch, channels * height * width};

    EXPECT_EQ(region_yolo->get_output_shape(0), expected_shape);
}

TEST(type_prop, region_yolo_v3_1) {
    const size_t num = 9;
    const size_t coords = 4;
    const size_t classes = 20;
    const size_t batch = 1;
    const size_t channels = 75;
    const size_t width = 32;
    const size_t height = 32;
    const std::vector<int64_t> mask{0, 1, 2};
    const int axis = 1;
    const int end_axis = 3;
    const auto in_shape = Shape{batch, channels, width, height};
    auto data_param = make_shared<op::Parameter>(element::f32, in_shape);
    auto region_yolo = make_shared<op::v0::RegionYolo>(data_param, coords, classes, num, false, mask, axis, end_axis);

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{batch, channels, height, width};

    EXPECT_EQ(region_yolo->get_output_shape(0), expected_shape);
}

TEST(type_prop, region_yolo_v3_2) {
    const size_t num = 1;
    const size_t coords = 4;
    const size_t classes = 1;
    const size_t batch = 1;
    const size_t channels = 8;
    const size_t width = 2;
    const size_t height = 2;
    const std::vector<int64_t> mask{0};
    const int axis = 1;
    const int end_axis = 3;
    const auto in_shape = Shape{batch, channels, width, height};
    auto data_param = make_shared<op::Parameter>(element::f32, in_shape);
    auto region_yolo = make_shared<op::v0::RegionYolo>(data_param, coords, classes, num, false, mask, axis, end_axis);

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{batch, (classes + coords + 1) * mask.size(), height, width};

    EXPECT_EQ(region_yolo->get_output_shape(0), expected_shape);
}