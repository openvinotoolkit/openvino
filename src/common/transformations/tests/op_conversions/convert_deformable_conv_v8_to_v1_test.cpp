// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvertDeformableConv8to1) {
    {
        const Strides strides{1, 1};
        const CoordinateDiff padding{0, 0};
        const Strides dilations{1, 1};

        const Shape input_shape{1, 1, 4, 4};
        const Shape filter_shape{1, 1, 2, 2};
        const Shape offsets_shape{1, 8, 3, 3};

        auto data = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto filter = std::make_shared<opset8::Parameter>(element::f32, filter_shape);
        auto offsets = std::make_shared<opset8::Parameter>(element::f32, offsets_shape);

        auto deformable_conv = std::make_shared<opset8::DeformableConvolution>(data,
                                                                               offsets,
                                                                               filter,
                                                                               strides,
                                                                               padding,
                                                                               padding,
                                                                               dilations);

        model = std::make_shared<Model>(NodeVector{deformable_conv}, ParameterVector{data, filter, offsets});
        manager.register_pass<ov::pass::ConvertDeformableConv8To1>();
    }

    {
        const Strides strides{1, 1};
        const CoordinateDiff padding{0, 0};
        const Strides dilations{1, 1};

        const Shape input_shape{1, 1, 4, 4};
        const Shape filter_shape{1, 1, 2, 2};
        const Shape offsets_shape{1, 8, 3, 3};

        auto data = std::make_shared<opset1::Parameter>(element::f32, input_shape);
        auto filter = std::make_shared<opset1::Parameter>(element::f32, filter_shape);
        auto offsets = std::make_shared<opset1::Parameter>(element::f32, offsets_shape);

        auto deformable_conv = std::make_shared<opset1::DeformableConvolution>(data,
                                                                               offsets,
                                                                               filter,
                                                                               strides,
                                                                               padding,
                                                                               padding,
                                                                               dilations);

        model_ref = std::make_shared<Model>(NodeVector{deformable_conv}, ParameterVector{data, filter, offsets});
    }
}

TEST_F(TransformationTestsF, ConvertDeformableConv8to1_mask) {
    {
        const Strides strides{1, 1};
        const CoordinateDiff padding{0, 0};
        const Strides dilations{1, 1};

        const Shape input_shape{1, 1, 4, 4};
        const Shape filter_shape{1, 1, 2, 2};
        const Shape offsets_shape{1, 8, 3, 3};
        const Shape mask_shape{1, 4, 3, 3};

        auto data = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto filter = std::make_shared<opset8::Parameter>(element::f32, filter_shape);
        auto offsets = std::make_shared<opset8::Parameter>(element::f32, offsets_shape);
        auto mask = std::make_shared<opset8::Parameter>(element::f32, mask_shape);

        auto deformable_conv = std::make_shared<opset8::DeformableConvolution>(data,
                                                                               offsets,
                                                                               filter,
                                                                               mask,
                                                                               strides,
                                                                               padding,
                                                                               padding,
                                                                               dilations);

        model = std::make_shared<Model>(NodeVector{deformable_conv}, ParameterVector{data, filter, mask, offsets});

        pass::Manager manager;
        manager.register_pass<ov::pass::ConvertDeformableConv8To1>();
    }
}

TEST_F(TransformationTestsF, ConvertDeformableConv8to1_bilinear_interpolation_padding) {
    {
        const Strides strides{1, 1};
        const CoordinateDiff padding{0, 0};
        const Strides dilations{1, 1};

        const Shape input_shape{1, 1, 4, 4};
        const Shape filter_shape{1, 1, 2, 2};
        const Shape offsets_shape{1, 8, 3, 3};

        auto data = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto filter = std::make_shared<opset8::Parameter>(element::f32, filter_shape);
        auto offsets = std::make_shared<opset8::Parameter>(element::f32, offsets_shape);

        auto deformable_conv = std::make_shared<opset8::DeformableConvolution>(data,
                                                                               offsets,
                                                                               filter,
                                                                               strides,
                                                                               padding,
                                                                               padding,
                                                                               dilations,
                                                                               op::PadType::EXPLICIT,
                                                                               1,
                                                                               1,
                                                                               true);

        model = std::make_shared<Model>(NodeVector{deformable_conv}, ParameterVector{data, filter, offsets});

        pass::Manager manager;
        manager.register_pass<ov::pass::ConvertDeformableConv8To1>();
    }
}
