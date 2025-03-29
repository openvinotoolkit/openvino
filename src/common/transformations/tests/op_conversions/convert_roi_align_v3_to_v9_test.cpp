// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_roi_align_v3_to_v9.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvertROIAlign3To9) {
    {
        const int N = 1;
        const int C = 3;
        const int H = 5;
        const int W = 5;
        const int num_rois = 5;
        const int pooled_height = 3;
        const int pooled_width = 4;
        const auto data_shape = Shape{N, C, H, W};
        const auto rois_shape = Shape{num_rois, 4};

        const auto data = std::make_shared<opset3::Parameter>(element::f32, data_shape);
        const auto rois = std::make_shared<opset3::Parameter>(element::f32, rois_shape);
        const auto batch_indices = std::make_shared<opset3::Parameter>(element::i32, Shape{num_rois});

        auto roi_align = std::make_shared<opset3::ROIAlign>(data,
                                                            rois,
                                                            batch_indices,
                                                            pooled_height,
                                                            pooled_width,
                                                            2,
                                                            1.0f / 16.0f,
                                                            "avg");

        model = std::make_shared<Model>(NodeVector{roi_align}, ParameterVector{data, rois, batch_indices});
        manager.register_pass<ov::pass::ConvertROIAlign3To9>();
    }

    {
        const int N = 1;
        const int C = 3;
        const int H = 5;
        const int W = 5;
        const int num_rois = 5;
        const int pooled_height = 3;
        const int pooled_width = 4;
        const auto data_shape = Shape{N, C, H, W};
        const auto rois_shape = Shape{num_rois, 4};

        const auto data = std::make_shared<opset9::Parameter>(element::f32, data_shape);
        const auto rois = std::make_shared<opset9::Parameter>(element::f32, rois_shape);
        const auto batch_indices = std::make_shared<opset9::Parameter>(element::i32, Shape{num_rois});
        const auto pooling_mode = EnumNames<opset9::ROIAlign::PoolingMode>::as_enum("avg");

        auto roi_align = std::make_shared<opset9::ROIAlign>(data,
                                                            rois,
                                                            batch_indices,
                                                            pooled_height,
                                                            pooled_width,
                                                            2,
                                                            1.0f / 16.0f,
                                                            pooling_mode);

        model_ref = std::make_shared<Model>(NodeVector{roi_align}, ParameterVector{data, rois, batch_indices});
    }
}
