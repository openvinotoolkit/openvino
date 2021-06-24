// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvertInterpolate1ToInterpolate4) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data_node = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 4, 30, 30});
        auto out_shape_node = opset1::Constant::create(element::i32, Shape{4}, {2, 4, 40, 40});

        auto interpolate1_attr = op::v0::InterpolateAttrs();
        interpolate1_attr.axes = AxisSet(std::vector<size_t>{0, 1, 2, 3});
        interpolate1_attr.mode = "nearest";
        interpolate1_attr.align_corners = false;
        interpolate1_attr.antialias = false;
        interpolate1_attr.pads_begin = std::vector<size_t>{0, 0, 0, 0};
        interpolate1_attr.pads_end = std::vector<size_t>{0, 0, 0, 0};

        auto interpolate1 = std::make_shared<opset1::Interpolate>(data_node, out_shape_node, interpolate1_attr);

        f = std::make_shared<Function>(NodeVector{interpolate1}, ParameterVector{data_node});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvertInterpolate1ToInterpolate4>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data_node = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 4, 30, 30});
        auto out_shape_node = opset1::Constant::create(element::i32, Shape{4}, {2, 4, 40, 40});
        auto default_scales_node = opset1::Constant::create(ngraph::element::f32, Shape{4}, {1.f, 1.f, 4.0f / 3.0f, 4.0f / 3.0f});
        auto axes_node = opset1::Constant::create(ngraph::element::i64, Shape{4}, {0, 1, 2, 3});

        auto interpolate4_attr = opset4::Interpolate::InterpolateAttrs(opset4::Interpolate::InterpolateMode::nearest,
            opset4::Interpolate::ShapeCalcMode::sizes, std::vector<size_t>{0, 0, 0, 0}, std::vector<size_t>{0, 0, 0, 0},
            opset4::Interpolate::CoordinateTransformMode::asymmetric, opset4::Interpolate::NearestMode::simple,
            false, -0.75);

        auto interpolate4 = std::make_shared<opset4::Interpolate>(data_node, out_shape_node, default_scales_node, axes_node, interpolate4_attr);

        f_ref = std::make_shared<Function>(NodeVector{interpolate4}, ParameterVector{data_node});
    }

    auto res = compare_functions(f, f_ref, true, false, false, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertInterpolate1ToInterpolate4_1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data_node = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 4, 30, 30});
        auto out_shape_node = opset1::Constant::create(element::i32, Shape{2}, {40, 40});

        auto interpolate1_attr = op::v0::InterpolateAttrs();
        interpolate1_attr.axes = AxisSet(std::vector<size_t>{2, 3});
        interpolate1_attr.mode = "linear";
        interpolate1_attr.align_corners = false;
        interpolate1_attr.antialias = true;
        interpolate1_attr.pads_begin = std::vector<size_t>{0, 0, 0, 0};
        interpolate1_attr.pads_end = std::vector<size_t>{0, 0, 0, 0};

        auto interpolate1 = std::make_shared<opset1::Interpolate>(data_node, out_shape_node, interpolate1_attr);

        f = std::make_shared<Function>(NodeVector{interpolate1}, ParameterVector{data_node});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvertInterpolate1ToInterpolate4>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data_node = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 4, 30, 30});
        auto out_shape_node = opset1::Constant::create(element::i32, Shape{2}, {40, 40});
        auto default_scales_node = opset1::Constant::create(ngraph::element::f32, Shape{2}, {4.0f / 3.0f, 4.0f / 3.0f});
        auto axes_node = opset1::Constant::create(ngraph::element::i64, Shape{2}, {2, 3});

        auto interpolate4_attr = opset4::Interpolate::InterpolateAttrs(opset4::Interpolate::InterpolateMode::linear_onnx,
            opset4::Interpolate::ShapeCalcMode::sizes, std::vector<size_t>{0, 0, 0, 0}, std::vector<size_t>{0, 0, 0, 0},
            opset4::Interpolate::CoordinateTransformMode::asymmetric, opset4::Interpolate::NearestMode::simple,
            true, -0.75);

        auto interpolate4 = std::make_shared<opset4::Interpolate>(data_node, out_shape_node, default_scales_node, axes_node, interpolate4_attr);

        f_ref = std::make_shared<Function>(NodeVector{interpolate4}, ParameterVector{data_node});
    }

    auto res = compare_functions(f, f_ref, true, false, false, true, true);
    ASSERT_TRUE(res.first) << res.second;
}
