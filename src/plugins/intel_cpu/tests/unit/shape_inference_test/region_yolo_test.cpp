// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <region_yolo_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace std;

TEST(StaticShapeInferenceTest, RegionYoloV0) {
    auto inputs = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{-1, -1, -1, -1});
    auto op = make_shared<op::v0::RegionYolo>(inputs, 0, 0, 0, true, std::vector<int64_t>{}, 0, 1);

    check_static_shape(op.get(), {StaticShape{1, 125, 13, 13}}, {StaticShape{1 * 125, 13, 13}});
}

TEST(StaticShapeInferenceTest, RegionYoloV0Dynamic) {
    auto inputs = make_shared<op::v0::Parameter>(element::f32,
                                                 ov::PartialShape{{1, 11}, {2, 12}, ov::Dimension::dynamic(), {4, 14}});
    auto op = make_shared<op::v0::RegionYolo>(inputs, 4, 80, 5, true, std::vector<int64_t>{}, 1, 3);

    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape({{1, 11}, ov::Dimension::dynamic()}));

    check_static_shape(op.get(), {StaticShape{10, 125, 13, 13}}, {StaticShape{10, 125 * 13 * 13}});
}