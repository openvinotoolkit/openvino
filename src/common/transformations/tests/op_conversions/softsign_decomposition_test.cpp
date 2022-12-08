// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <transformations/op_conversions/softsign_decomposition.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, SoftSignDecomposition) {
    {
        auto data = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto softsign = std::make_shared<ngraph::opset9::SoftSign>(data);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{softsign}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::SoftSignDecomposition>();
    }

    {
        auto input = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto abs = std::make_shared<ngraph::opset9::Abs>(input);
        auto add = std::make_shared<ngraph::opset9::Add>(abs, ngraph::opset9::Constant::create(ngraph::element::f32,
                                                                                               ngraph::Shape{1},
                                                                                               {1}));
        auto div = std::make_shared<ngraph::opset9::Divide>(input, add);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SoftSignDecompositionFP16) {
    {
        auto data = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f16, ngraph::Shape{3, 1, 2});
        auto softsign = std::make_shared<ngraph::opset9::SoftSign>(data);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{softsign}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::SoftSignDecomposition>();
    }

    {
        auto input = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f16, ngraph::Shape{3, 1, 2});
        auto abs = std::make_shared<ngraph::opset9::Abs>(input);
        auto add = std::make_shared<ngraph::opset9::Add>(abs, ngraph::opset9::Constant::create(ngraph::element::f16,
                                                                                               ngraph::Shape{1},
                                                                                               {1}));
        auto div = std::make_shared<ngraph::opset9::Divide>(input, add);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});
    }
}
