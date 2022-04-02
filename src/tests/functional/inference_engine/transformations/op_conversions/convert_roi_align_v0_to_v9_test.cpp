// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pass/manager.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/convert_roi_align_v0_to_v9.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, ConvertROIAlign0To9) {
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

        function = std::make_shared<Function>(NodeVector{roi_align}, ParameterVector{data, rois, batch_indices});
        manager.register_pass<pass::ConvertROIAlign0To9>();
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

        function_ref = std::make_shared<Function>(NodeVector{roi_align}, ParameterVector{data, rois, batch_indices});
    }
}