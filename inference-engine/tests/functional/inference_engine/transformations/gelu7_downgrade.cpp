// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>


#include <ngraph/function.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <transformations/op_conversions/gelu7_downgrade.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, Gelu7Downgrade) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto gelu = std::make_shared<ngraph::opset7::Gelu>(input, ngraph::op::GeluApproximationMode::ERF);
        gelu->set_friendly_name("gelu7");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{gelu}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::Gelu7Downgrade>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto gelu = std::make_shared<ngraph::opset2::Gelu>(input);
        gelu->set_friendly_name("gelu7");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gelu}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto output_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(output_node->get_friendly_name() == "gelu7") << "Transformation Gelu7Downgrade should keep output names.\n";
}

