
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "region_yolo_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class StaticShapeRegionYoloTest : public OpStaticShapeInferenceTest<op::v0::RegionYolo> {};

TEST_F(StaticShapeRegionYoloTest, default_ctor_do_soft_max_no_args) {
    op = make_op();
    op->set_do_softmax(true);
    op->set_axis(-2);
    op->set_end_axis(3);

    input_shapes = ShapeVector{{10, 8, 12, 6}};
    shape_inference(op.get(), input_shapes, output_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({10, 8, 72}));
}

TEST_F(StaticShapeRegionYoloTest, data_input_is_dynamic_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    op = make_op(data, 0, 0, 0, true, std::vector<int64_t>(), 1, 3);

    input_shapes = ShapeVector{{2, 2, 3, 4}};
    shape_inference(op.get(), input_shapes, output_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 24}));
}

TEST_F(StaticShapeRegionYoloTest, data_input_is_static_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic(4));
    op = make_op(data, 5, 4, 20, false, std::vector<int64_t>{0, 1}, 1, 3);

    input_shapes = ShapeVector{{2, 5, 6, 7}};
    shape_inference(op.get(), input_shapes, output_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 20, 6, 7}));
}

TEST_F(StaticShapeRegionYoloTest, data_shape_not_compatible_rank_4) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    op = make_op(data, 5, 4, 20, false, std::vector<int64_t>{0, 1}, 1, 3);

    OV_EXPECT_THROW(shape_inference(op.get(), ShapeVector{{2, 20, 12, 24, 1}}, output_shapes),
                    NodeValidationFailure,
                    HasSubstr("Input must be a tensor of rank 4, but got"));
}

// TEST_F(StaticShapeRegionYoloTest, h_dim_not_div_by_stride) {
//     const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
//     op = make_op(data, 2);

//     OV_EXPECT_THROW(shape_inference(op.get(), ShapeVector{{2, 20, 11, 24}}, output_shapes),
//                     NodeValidationFailure,
//                     HasSubstr("H and W should be divisible by stride"));
// }

// #include <region_yolo_shape_inference.hpp>

// #include "utils.hpp"

// using namespace ov;
// using namespace ov::intel_cpu;
// using namespace std;

// TEST(StaticShapeInferenceTest, RegionYoloV0) {
//     auto inputs = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{-1, -1, -1, -1});
//     auto op = make_shared<op::v0::RegionYolo>(inputs, 0, 0, 0, true, std::vector<int64_t>{}, 0, 1);

//     check_static_shape(op.get(), {StaticShape{1, 125, 13, 13}}, {StaticShape{1 * 125, 13, 13}});
// }

// TEST(StaticShapeInferenceTest, RegionYoloV0Dynamic) {
//     auto inputs = make_shared<op::v0::Parameter>(element::f32,
//                                                  ov::PartialShape{{1, 11}, {2, 12}, ov::Dimension::dynamic(), {4,
//                                                  14}});
//     auto op = make_shared<op::v0::RegionYolo>(inputs, 4, 80, 5, true, std::vector<int64_t>{}, 1, 3);

//     EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape({{1, 11}, ov::Dimension::dynamic()}));

//     check_static_shape(op.get(), {StaticShape{10, 125, 13, 13}}, {StaticShape{10, 125 * 13 * 13}});
// }
