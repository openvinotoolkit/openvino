// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, region_yolo_op) {
    NodeBuilder::get_ops().register_factory<opset1::RegionYolo>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 255, 26, 26});

    size_t num_coords = 4;
    size_t num_classes = 1;
    size_t num_regions = 6;
    auto do_softmax = false;
    auto mask = std::vector<int64_t>{0, 1};
    auto axis = 1;
    auto end_axis = 3;
    auto anchors = std::vector<float>{10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};

    auto region_yolo = make_shared<opset1::RegionYolo>(data,
                                                       num_coords,
                                                       num_classes,
                                                       num_regions,
                                                       do_softmax,
                                                       mask,
                                                       axis,
                                                       end_axis,
                                                       anchors);
    NodeBuilder builder(region_yolo);
    auto g_region_yolo = ov::as_type_ptr<opset1::RegionYolo>(builder.create());

    EXPECT_EQ(g_region_yolo->get_num_coords(), region_yolo->get_num_coords());
    EXPECT_EQ(g_region_yolo->get_num_classes(), region_yolo->get_num_classes());
    EXPECT_EQ(g_region_yolo->get_num_regions(), region_yolo->get_num_regions());
    EXPECT_EQ(g_region_yolo->get_do_softmax(), region_yolo->get_do_softmax());
    EXPECT_EQ(g_region_yolo->get_mask(), region_yolo->get_mask());
    EXPECT_EQ(g_region_yolo->get_anchors(), region_yolo->get_anchors());
    EXPECT_EQ(g_region_yolo->get_axis(), region_yolo->get_axis());
    EXPECT_EQ(g_region_yolo->get_end_axis(), region_yolo->get_end_axis());
}
