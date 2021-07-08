// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvertDeformableConv8to1) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
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

        f = std::make_shared<Function>(NodeVector{deformable_conv}, ParameterVector{data, filter, offsets});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertDeformableConv8To1>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
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

        f_ref = std::make_shared<Function>(NodeVector{deformable_conv}, ParameterVector{data, filter, offsets});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertDeformableConv8to1_mask) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
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

        f = std::make_shared<Function>(NodeVector{deformable_conv}, ParameterVector{data, filter,
                                                                                    mask, offsets});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertDeformableConv8To1>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    // mask input is provided, DeformableConvolution-8 must remain
    ASSERT_EQ(count_ops_of_type<opset1::DeformableConvolution>(f), 0);
    ASSERT_EQ(count_ops_of_type<opset8::DeformableConvolution>(f), 1);
}

TEST(TransformationTests, ConvertDeformableConv8to1_bilinear_interpolation_padding) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
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

        f = std::make_shared<Function>(NodeVector{deformable_conv}, ParameterVector{data, filter, offsets});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertDeformableConv8To1>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    //  use_bilinear_interpolation_padding is true, DeformableConvolution-8 must remain
    ASSERT_EQ(count_ops_of_type<opset1::DeformableConvolution>(f), 0);
    ASSERT_EQ(count_ops_of_type<opset8::DeformableConvolution>(f), 1);
}
