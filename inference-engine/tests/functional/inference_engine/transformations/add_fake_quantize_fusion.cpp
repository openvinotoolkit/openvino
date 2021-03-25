// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <transformations/common_optimizations/add_fake_quantize_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


TEST(TransformationTests, AddFakeQuantizeFusion) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto add_const = opset5::Constant::create(element::f32, Shape{1}, {2});
        auto add = std::make_shared<opset5::Add>(data, add_const);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low,
                                                         input_high, output_low,
                                                         output_high, 11);
        f = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::AddFakeQuantizeFusion>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {-2});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {18});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low,
                                                         input_high, output_low,
                                                         output_high, 11);
        f_ref = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, AddFakeQuantizeFusionConstantOnFirstInput) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto add_const = opset5::Constant::create(element::f32, Shape{1}, {2});
        auto add = std::make_shared<opset5::Add>(add_const, data);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low,
                                                         input_high, output_low,
                                                         output_high, 11);
        f = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::AddFakeQuantizeFusion>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {-2});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {18});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low,
                                                         input_high, output_low,
                                                         output_high, 11);
        f_ref = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, AddFakeQuantizeFusionReshape) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto add_const = opset5::Constant::create(element::f32, Shape{3, 1, 1}, {2, 3, 4});
        auto add = std::make_shared<opset5::Add>(data, add_const);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low,
                                                         input_high, output_low,
                                                         output_high, 11);
        f = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::AddFakeQuantizeFusion>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-2, -3, -4});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {18, 17, 16});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low,
                                                         input_high, output_low,
                                                         output_high, 11);
        f_ref = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeAddFakeQuantizeFusionNotAConstant) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto add_2nd_input = std::make_shared<opset5::Parameter>(element::f32, Shape{1});
        auto add = std::make_shared<opset5::Add>(data, add_2nd_input);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low,
                                                         input_high, output_low,
                                                         output_high, 11);
        f = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data, add_2nd_input});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::AddFakeQuantizeFusion>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto add_2nd_input = std::make_shared<opset5::Parameter>(element::f32, Shape{1});
        auto add = std::make_shared<opset5::Add>(data, add_2nd_input);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low,
                                                         input_high, output_low,
                                                         output_high, 11);
        f_ref = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data, add_2nd_input});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}
