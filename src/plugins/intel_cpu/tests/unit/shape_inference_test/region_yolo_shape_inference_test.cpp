
// Copyright (C) 2018-2025 Intel Corporation
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

    input_shapes = StaticShapeVector{{10, 8, 12, 6}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({10, 8, 72}));
}

TEST_F(StaticShapeRegionYoloTest, data_input_is_dynamic_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    op = make_op(data, 0, 0, 0, true, std::vector<int64_t>(), 1, 3);

    input_shapes = StaticShapeVector{{2, 2, 3, 4}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 24}));
}

TEST_F(StaticShapeRegionYoloTest, data_input_is_static_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic(4));
    op = make_op(data, 5, 4, 20, false, std::vector<int64_t>{0, 1}, 1, 3);

    input_shapes = StaticShapeVector{{2, 5, 6, 7}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 20, 6, 7}));
}

TEST_F(StaticShapeRegionYoloTest, data_shape_not_compatible_rank_4) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    op = make_op(data, 5, 4, 20, false, std::vector<int64_t>{0, 1}, 1, 3);

    OV_EXPECT_THROW(shape_inference(op.get(), StaticShapeVector({{2, 20, 12, 24, 1}})),
                    NodeValidationFailure,
                    HasSubstr("Input must be a tensor of rank 4, but got"));
}
