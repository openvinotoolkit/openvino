// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/mish_fusion.hpp>
#include <transformations/common_optimizations/softplus_to_mish_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

// LPT to nGraph migration: temporary disabling unexpected not reproduced fails on CI:
// https://openvino-ci.intel.com/job/private-ci/job/ie/job/build-linux-ubuntu18_i386/478/
TEST_F(TransformationTestsF, MishFusing) {
    {
        auto input0 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto exp = std::make_shared<ngraph::opset4::Exp>(input0);
        auto input_const = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {-1});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, input_const);
        auto log = std::make_shared<ngraph::opset4::Log>(add);
        auto tanh = std::make_shared<ngraph::opset4::Tanh>(log);
        auto mul = std::make_shared<ngraph::opset4::Multiply>(input0, tanh);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input0});

        manager.register_pass<ngraph::pass::MishFusion>();
    }

    {
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto mish = std::make_shared<ngraph::opset4::Mish>(data);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mish}, ngraph::ParameterVector{data});
    }
}


TEST_F(TransformationTestsF, MishWithSoftPlusFusing) {
    {
        auto input0 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto softplus = std::make_shared<ngraph::opset4::SoftPlus>(input0);
        auto tanh = std::make_shared<ngraph::opset4::Tanh>(softplus);
        auto mul = std::make_shared<ngraph::opset4::Multiply>(input0, tanh);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input0});

        manager.register_pass<ngraph::pass::SoftPlusToMishFusion>();
    }

    {
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto mish = std::make_shared<ngraph::opset4::Mish>(data);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mish}, ngraph::ParameterVector{data});
    }
}
