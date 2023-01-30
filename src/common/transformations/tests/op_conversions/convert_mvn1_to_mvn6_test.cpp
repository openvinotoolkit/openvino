// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/convert_mvn1_to_mvn6.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, ConvertMVN1ToMVN6) {
    {
        auto data = std::make_shared<ngraph::opset2::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto mvn = std::make_shared<ngraph::op::v0::MVN>(data, false, true, 1e-5);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::ConvertMVN1ToMVN6>();
    }

    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {2, 3});
        auto mvn =
            std::make_shared<ngraph::op::v6::MVN>(data, axes_const, true, 1e-5f, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertMVN1ToMVN6_across_channels) {
    {
        auto data = std::make_shared<ngraph::opset2::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto mvn = std::make_shared<ngraph::op::v0::MVN>(data, true, true, 1e-5);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::ConvertMVN1ToMVN6>();
    }

    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 2, 3});
        auto mvn =
            std::make_shared<ngraph::op::v6::MVN>(data, axes_const, true, 1e-5f, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertMVN1ToMVN6_5D) {
    {
        auto data = std::make_shared<ngraph::opset2::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4, 5});
        auto mvn = std::make_shared<ngraph::op::v0::MVN>(data, false, true, 1e-5);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::ConvertMVN1ToMVN6>();
    }

    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4, 5});
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {2, 3, 4});
        auto mvn =
            std::make_shared<ngraph::op::v6::MVN>(data, axes_const, true, 1e-5f, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});
    }
}

namespace {
struct ConvertMVN1ToMVN6_OutOfFloat32Eps_params {
    double eps_d;
    float eps_f;
};

class ConvertMVN1ToMVN6_OutOfFloat32Eps : public testing::WithParamInterface<ConvertMVN1ToMVN6_OutOfFloat32Eps_params>,
                                          public TransformationTestsF {};

TEST_P(ConvertMVN1ToMVN6_OutOfFloat32Eps, Limits) {
    manager.register_pass<ov::pass::ConvertMVN1ToMVN6>();

    const auto& params = GetParam();
    {
        auto data = std::make_shared<ngraph::opset2::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto mvn = std::make_shared<ngraph::op::v0::MVN>(data, true, true, params.eps_d);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});
    }
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3, 4});
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 2, 3});
        auto mvn = std::make_shared<ngraph::op::v6::MVN>(data,
                                                         axes_const,
                                                         true,
                                                         params.eps_f,
                                                         ngraph::op::MVNEpsMode::INSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mvn}, ngraph::ParameterVector{data});
    }
}

const auto out_of_f32_epsilons =
    std::vector<ConvertMVN1ToMVN6_OutOfFloat32Eps_params>{{1e-39, std::numeric_limits<float>::min()},
                                                          {1e-3, 1e-3f},
                                                          {1e+39, std::numeric_limits<float>::max()}};
}  // namespace

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         ConvertMVN1ToMVN6_OutOfFloat32Eps,
                         ::testing::ValuesIn(out_of_f32_epsilons));
