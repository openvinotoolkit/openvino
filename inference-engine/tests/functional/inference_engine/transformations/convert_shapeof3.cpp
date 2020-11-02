// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <transformations/op_conversions/convert_shapeof3.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertShapeOf3WithI64) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto shapeof = std::make_shared<ngraph::opset3::ShapeOf>(input, ngraph::element::i64);
        shapeof->set_friendly_name("shapeof");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{shapeof}, ngraph::ParameterVector{input});

        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertShapeOf3().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto shapeof = std::make_shared<ngraph::opset1::ShapeOf>(input);
        shapeof->set_friendly_name("shapeof");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{shapeof}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto output_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(output_node->get_friendly_name() == "shapeof") << "Transformation ConvertShapeOf3 should keep output names.\n";
}

TEST(TransformationTests, ConvertShapeOf3WithI32) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto shapeof = std::make_shared<ngraph::opset3::ShapeOf>(input, ngraph::element::i32);
        shapeof->set_friendly_name("shapeof");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{shapeof}, ngraph::ParameterVector{input});

        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertShapeOf3().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto shapeof = std::make_shared<ngraph::opset1::ShapeOf>(input);
        auto convert = std::make_shared<ngraph::opset1::Convert>(shapeof, ngraph::element::i32);
        convert->set_friendly_name("shapeof");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{convert}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto output_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(output_node->get_friendly_name() == "shapeof") << "Transformation ConvertShapeOf3 should keep output names.\n";
}
