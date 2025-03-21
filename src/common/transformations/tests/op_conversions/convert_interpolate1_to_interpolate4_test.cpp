// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvertInterpolate1ToInterpolate4) {
    {
        auto data_node = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 4, 30, 30});
        auto out_shape_node = op::v0::Constant::create(element::i32, Shape{4}, {2, 4, 40, 40});

        auto interpolate1_attr = op::v0::Interpolate::Attributes();
        interpolate1_attr.axes = AxisSet(std::vector<size_t>{0, 1, 2, 3});
        interpolate1_attr.mode = "nearest";
        interpolate1_attr.align_corners = false;
        interpolate1_attr.antialias = false;
        interpolate1_attr.pads_begin = std::vector<size_t>{0, 0, 0, 0};
        interpolate1_attr.pads_end = std::vector<size_t>{0, 0, 0, 0};

        auto interpolate1 = std::make_shared<op::v0::Interpolate>(data_node, out_shape_node, interpolate1_attr);

        model = std::make_shared<Model>(NodeVector{interpolate1}, ParameterVector{data_node});

        manager.register_pass<ov::pass::ConvertInterpolate1ToInterpolate4>();
    }

    {
        auto data_node = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 4, 30, 30});
        auto out_shape_node = op::v0::Constant::create(element::i32, Shape{4}, {2, 4, 40, 40});
        auto default_scales_node =
            op::v0::Constant::create(element::f32, Shape{4}, {1.f, 1.f, 4.0f / 3.0f, 4.0f / 3.0f});
        auto axes_node = op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 2, 3});

        auto interpolate4_attr =
            op::v4::Interpolate::InterpolateAttrs(op::v4::Interpolate::InterpolateMode::NEAREST,
                                                  op::v4::Interpolate::ShapeCalcMode::SIZES,
                                                  std::vector<size_t>{0, 0, 0, 0},
                                                  std::vector<size_t>{0, 0, 0, 0},
                                                  op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
                                                  op::v4::Interpolate::NearestMode::SIMPLE,
                                                  false,
                                                  -0.75);

        auto interpolate4 = std::make_shared<op::v4::Interpolate>(data_node,
                                                                  out_shape_node,
                                                                  default_scales_node,
                                                                  axes_node,
                                                                  interpolate4_attr);

        model_ref = std::make_shared<Model>(NodeVector{interpolate4}, ParameterVector{data_node});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertInterpolate1ToInterpolate4_1) {
    {
        auto data_node = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 4, 30, 30});
        auto out_shape_node = op::v0::Constant::create(element::i32, Shape{2}, {40, 40});

        auto interpolate1_attr = op::v0::Interpolate::Attributes();
        interpolate1_attr.axes = AxisSet(std::vector<size_t>{2, 3});
        interpolate1_attr.mode = "linear";
        interpolate1_attr.align_corners = false;
        interpolate1_attr.antialias = true;
        interpolate1_attr.pads_begin = std::vector<size_t>{0, 0, 0, 0};
        interpolate1_attr.pads_end = std::vector<size_t>{0, 0, 0, 0};

        auto interpolate1 = std::make_shared<op::v0::Interpolate>(data_node, out_shape_node, interpolate1_attr);

        model = std::make_shared<Model>(NodeVector{interpolate1}, ParameterVector{data_node});

        manager.register_pass<ov::pass::ConvertInterpolate1ToInterpolate4>();
    }

    {
        auto data_node = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 4, 30, 30});
        auto out_shape_node = op::v0::Constant::create(element::i32, Shape{2}, {40, 40});
        auto default_scales_node = op::v0::Constant::create(element::f32, Shape{2}, {4.0f / 3.0f, 4.0f / 3.0f});
        auto axes_node = op::v0::Constant::create(element::i64, Shape{2}, {2, 3});

        auto interpolate4_attr =
            op::v4::Interpolate::InterpolateAttrs(op::v4::Interpolate::InterpolateMode::LINEAR_ONNX,
                                                  op::v4::Interpolate::ShapeCalcMode::SIZES,
                                                  std::vector<size_t>{0, 0, 0, 0},
                                                  std::vector<size_t>{0, 0, 0, 0},
                                                  op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
                                                  op::v4::Interpolate::NearestMode::SIMPLE,
                                                  true,
                                                  -0.75);

        auto interpolate4 = std::make_shared<op::v4::Interpolate>(data_node,
                                                                  out_shape_node,
                                                                  default_scales_node,
                                                                  axes_node,
                                                                  interpolate4_attr);

        model_ref = std::make_shared<Model>(NodeVector{interpolate4}, ParameterVector{data_node});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST(TransformationTests, DynamiShapeInterpolate1To4) {
    auto data_node = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 5, {1, 10}, -1});
    auto out_shape_node = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    auto interpolate1_attr = op::v0::Interpolate::Attributes();
    interpolate1_attr.axes = AxisSet(std::vector<size_t>{2, 3});
    interpolate1_attr.mode = "linear";
    interpolate1_attr.align_corners = false;
    interpolate1_attr.antialias = true;
    interpolate1_attr.pads_begin = std::vector<size_t>{0, 0, 0, 0};
    interpolate1_attr.pads_end = std::vector<size_t>{0, 0, 0, 0};

    auto interpolate1 = std::make_shared<op::v0::Interpolate>(data_node, out_shape_node, interpolate1_attr);
    auto f = std::make_shared<Model>(NodeVector{interpolate1}, ParameterVector{data_node, out_shape_node});

    auto manager = ov::pass::Manager();
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::ConvertInterpolate1ToInterpolate4>();
    manager.run_passes(f);

    ASSERT_TRUE(ov::op::util::has_op_with_type<op::v4::Interpolate>(f));
}
