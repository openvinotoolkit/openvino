// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <transformations/common_optimizations/relu_fake_quantize_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


TEST(TransformationTests, ReluFakeQuantizeFusion) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto relu = std::make_shared<opset5::Relu>(data);
        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {0, 0, 0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {20, 20, 20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(relu, input_low,
                                                         input_high, output_low,
                                                         output_high, 11);
        f = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ReluFakeQuantizeFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {0, 0, 0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {20, 20, 20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low,
                                                         input_high, output_low,
                                                         output_high, 11);
        f_ref = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReluFakeQuantizeFusionNegativeInputLow) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto relu = std::make_shared<opset5::Relu>(data);
        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {2, -2, -2});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {20, 20, 20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(relu, input_low,
                                                         input_high, output_low,
                                                         output_high, 11);
        f = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ReluFakeQuantizeFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto relu = std::make_shared<opset5::Relu>(data);
        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {2, -2, -2});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {20, 20, 20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(relu, input_low,
                                                         input_high, output_low,
                                                         output_high, 11);
        f_ref = std::make_shared<Function>(NodeVector{fq}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
