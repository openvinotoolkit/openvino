// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include "openvino/core/function.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/compress_float_constants.hpp"
#include "transformations/common_optimizations/mark_precision_sensitive_subgraphs.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov;
using namespace ngraph;

TEST(TransformationTests, CompressConstants_f32) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, Shape{ 1, 3, 12, 12 });
        auto const_weights = opset8::Constant::create(element::f32,
            Shape{ 1, 3, 3, 3 },
            { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        auto conv = std::make_shared<opset8::Convolution>(input,
            const_weights,
            Strides{1, 1},
            CoordinateDiff{0, 0},
            CoordinateDiff{0, 0},
            Strides{1, 1});
        auto const_scales = opset8::Constant::create(element::f32, Shape{ 1 }, { 1.4 });

        auto shape = std::make_shared<opset8::ShapeOf>(conv);
        auto convert1 = std::make_shared<opset8::Convert>(shape, element::f32);
        auto mul = std::make_shared<opset8::Multiply>(convert1, const_scales);
        auto convert2 = std::make_shared<opset8::Convert>(mul, element::i32);

        auto default_scales_node = opset8::Constant::create(element::f32, Shape{ 4 }, { 1., 1., 1.4, 1.4 });
        auto axes_node = opset8::Constant::create(element::i64, Shape{ 4 }, { 0, 1, 2, 3 });

        auto interpolate4_attr = opset8::Interpolate::InterpolateAttrs(opset8::Interpolate::InterpolateMode::NEAREST,
            opset8::Interpolate::ShapeCalcMode::SIZES, std::vector<size_t>{0, 0, 0, 0}, std::vector<size_t>{0, 0, 0, 0},
            opset8::Interpolate::CoordinateTransformMode::ASYMMETRIC, opset8::Interpolate::NearestMode::SIMPLE,
            false, -0.75);

        auto resize = std::make_shared<opset8::Interpolate>(conv, convert2, default_scales_node, axes_node, interpolate4_attr);

        f = std::make_shared<Function>(NodeVector{ resize }, ParameterVector{ input });

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::MarkPrecisionSensitiveSubgraphs>();
        manager.register_pass<pass::CompressFloatConstants>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, Shape{ 1, 3, 12, 12 });
        auto const_weights = opset8::Constant::create(element::f16,
            Shape{ 1, 3, 3, 3 },
            { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        auto convert_ins1 = std::make_shared<opset8::Convert>(const_weights, element::f32);
        auto conv = std::make_shared<opset8::Convolution>(input,
            convert_ins1,
            Strides{ 1, 1 },
            CoordinateDiff{ 0, 0 },
            CoordinateDiff{ 0, 0 },
            Strides{ 1, 1 });
        auto const_scales = opset8::Constant::create(element::f32, Shape{ 1 }, { 1.4 });

        auto shape = std::make_shared<opset8::ShapeOf>(conv);
        auto convert1 = std::make_shared<opset8::Convert>(shape, element::f32);
        auto mul = std::make_shared<opset8::Multiply>(convert1, const_scales);
        auto convert2 = std::make_shared<opset8::Convert>(mul, element::i32);

        auto default_scales_node = opset8::Constant::create(element::f32, Shape{ 4 }, { 1., 1., 1.4, 1.4 });
        auto axes_node = opset8::Constant::create(element::i64, Shape{ 4 }, { 0, 1, 2, 3 });

        auto interpolate4_attr = opset8::Interpolate::InterpolateAttrs(opset8::Interpolate::InterpolateMode::NEAREST,
            opset8::Interpolate::ShapeCalcMode::SIZES, std::vector<size_t>{0, 0, 0, 0}, std::vector<size_t>{0, 0, 0, 0},
            opset8::Interpolate::CoordinateTransformMode::ASYMMETRIC, opset8::Interpolate::NearestMode::SIMPLE,
            false, -0.75);

        auto resize = std::make_shared<opset8::Interpolate>(conv, convert2, default_scales_node, axes_node, interpolate4_attr);

        f_ref = std::make_shared<Function>(NodeVector{ resize }, ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CompressConstants_f64) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset8::Parameter>(element::f64, Shape{ 1, 3, 12, 12 });
        auto const_weights = opset8::Constant::create(element::f64,
            Shape{ 1, 3, 3, 3 },
            { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        auto conv = std::make_shared<opset8::Convolution>(input,
            const_weights,
            Strides{ 1, 1 },
            CoordinateDiff{ 0, 0 },
            CoordinateDiff{ 0, 0 },
            Strides{ 1, 1 });
        f = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::MarkPrecisionSensitiveSubgraphs>();
        manager.register_pass<pass::CompressFloatConstants>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::f64, Shape{ 1, 3, 12, 12 });
        auto const_weights = opset8::Constant::create(element::f16,
            Shape{ 1, 3, 3, 3 },
            { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        auto convert_ins1 = std::make_shared<opset8::Convert>(const_weights, element::f64);
        auto conv = std::make_shared<opset8::Convolution>(input,
            convert_ins1,
            Strides{ 1, 1 },
            CoordinateDiff{ 0, 0 },
            CoordinateDiff{ 0, 0 },
            Strides{ 1, 1 });
        f_ref = std::make_shared<Function>(NodeVector{ conv }, ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}
