// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/region_yolo.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, region_yolo_op) {
    NodeBuilder::opset().insert<ov::op::v0::RegionYolo>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 255, 26, 26});

    size_t num_coords = 4;
    size_t num_classes = 1;
    size_t num_regions = 6;
    auto do_softmax = false;
    auto mask = std::vector<int64_t>{0, 1};
    auto axis = 1;
    auto end_axis = 3;
    auto anchors = std::vector<float>{10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};

    auto region_yolo = make_shared<ov::op::v0::RegionYolo>(data,
                                                           num_coords,
                                                           num_classes,
                                                           num_regions,
                                                           do_softmax,
                                                           mask,
                                                           axis,
                                                           end_axis,
                                                           anchors);
    NodeBuilder builder(region_yolo, {data});
    auto g_region_yolo = ov::as_type_ptr<ov::op::v0::RegionYolo>(builder.create());

    EXPECT_EQ(g_region_yolo->get_num_coords(), region_yolo->get_num_coords());
    EXPECT_EQ(g_region_yolo->get_num_classes(), region_yolo->get_num_classes());
    EXPECT_EQ(g_region_yolo->get_num_regions(), region_yolo->get_num_regions());
    EXPECT_EQ(g_region_yolo->get_do_softmax(), region_yolo->get_do_softmax());
    EXPECT_EQ(g_region_yolo->get_mask(), region_yolo->get_mask());
    EXPECT_EQ(g_region_yolo->get_anchors(), region_yolo->get_anchors());
    EXPECT_EQ(g_region_yolo->get_axis(), region_yolo->get_axis());
    EXPECT_EQ(g_region_yolo->get_end_axis(), region_yolo->get_end_axis());
}
