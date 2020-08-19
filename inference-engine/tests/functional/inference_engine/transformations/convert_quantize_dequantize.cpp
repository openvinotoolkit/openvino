// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/convert_quantize_dequantize.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


TEST(TransformationTests, ConvertQuantizeDequantize) {
    size_t levels = 256;
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 1, 2}, {0, 1, 2, 3, 4, 5});
        auto input_low = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0});
        auto input_high = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {5});
        auto output_low = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-128});
        auto output_high = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {127});
        auto fq = std::make_shared<opset1::FakeQuantize>(data, input_low,
                                                         input_high, output_low,
                                                         output_high, levels);
        auto convert1 = std::make_shared<opset1::Convert>(fq, element::i8);
        auto convert2 = std::make_shared<opset1::Convert>(convert1, element::f32);
        auto zero_point = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {2});
        auto sub = std::make_shared<opset1::Subtract>(convert2, zero_point);
        auto scale = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {3});
        auto mul = std::make_shared<opset1::Multiply>(sub, scale);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
        m.register_pass<ngraph::pass::ConstantFolding>();
        m.run_passes(f);
    }

    {
        auto data = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 1, 2}, {0, 1, 2, 3, 4, 5});
        auto input_low = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0});
        auto input_high = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {5});
        auto output_low = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {(-128 - 2) * 3});
        auto output_high = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {(127 - 2) * 3});
        auto fq = std::make_shared<opset1::FakeQuantize>(data, input_low,
                                                         input_high, output_low,
                                                         output_high, levels);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fq}, ngraph::ParameterVector{});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
