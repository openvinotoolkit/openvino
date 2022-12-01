// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, reorg_yolo_stride_2) {
    const auto in_shape = Shape{1, 64, 26, 26};
    size_t stride = 2;
    auto data_param = make_shared<op::Parameter>(element::f32, in_shape);
    auto reorg_yolo = make_shared<op::v0::ReorgYolo>(data_param, stride);

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{1, 256, 13, 13};

    EXPECT_EQ(reorg_yolo->get_output_shape(0), expected_shape);
}

TEST(type_prop, reorg_yolo_stride_2_batch_2) {
    const auto in_shape = Shape{2, 64, 26, 26};
    size_t stride = 2;
    auto data_param = make_shared<op::Parameter>(element::f32, in_shape);
    auto reorg_yolo = make_shared<op::v0::ReorgYolo>(data_param, stride);

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{2, 256, 13, 13};

    EXPECT_EQ(reorg_yolo->get_output_shape(0), expected_shape);
}

TEST(type_prop, reorg_yolo_stride_2_smaller_H) {
    const auto in_shape = Shape{1, 24, 34, 62};
    size_t stride = 2;
    auto data_param = make_shared<op::Parameter>(element::f32, in_shape);
    auto reorg_yolo = make_shared<op::v0::ReorgYolo>(data_param, stride);

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{1, 96, 17, 31};
    EXPECT_EQ(reorg_yolo->get_output_shape(0), expected_shape);
}

TEST(type_prop, reorg_yolo_stride_3) {
    const auto in_shape = Shape{1, 9, 3, 3};
    size_t stride = 3;
    auto data_param = make_shared<op::Parameter>(element::f32, in_shape);
    auto reorg_yolo = make_shared<op::v0::ReorgYolo>(data_param, stride);

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape =
        Shape{in_shape[0], in_shape[1] * stride * stride, in_shape[2] / stride, in_shape[3] / stride};

    EXPECT_EQ(reorg_yolo->get_output_shape(0), expected_shape);
}

TEST(type_prop, reorg_yolo_catch_small_shape_stride) {
    const auto in_shape = Shape{1, 1, 4, 4};
    size_t stride = 2;
    auto data_param = make_shared<op::Parameter>(element::f32, in_shape);
    try {
        // Throw error test: For [N, C, H, W] input shape, C >= (stride*stride) is required.
        auto reorg_yolo = make_shared<op::v0::ReorgYolo>(data_param, stride);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible stride was not detected.";
    } catch (const ngraph_error& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("stride"));
    } catch (...) {
        FAIL() << "Stride size check failed for unexpected reason.";
    }
}
