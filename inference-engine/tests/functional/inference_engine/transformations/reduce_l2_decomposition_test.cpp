// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/reduce_l2_decomposition.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ReduceL2DecompositionTest) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto axes = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i32, ngraph::Shape{1});
        auto reduce_l2 = std::make_shared<ngraph::opset4::ReduceL2>(data, axes, true);
        reduce_l2->set_friendly_name("reduce_l2");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reduce_l2}, ngraph::ParameterVector{data, axes});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ReduceL2Decomposition>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto axes = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i32, ngraph::Shape{1});
        auto pow = std::make_shared<ngraph::opset4::Power>(data, ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{}, {2.0}));
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes, true);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(reduce_sum);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{sqrt}, ngraph::ParameterVector{data, axes});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto output_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(output_node->get_friendly_name() == "reduce_l2") << "Transformation ReduceL2Decomposition should keep output names.\n";
}
