// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/common_optimizations/simplify_shape_of_sub_graph.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, SharedShapeOfTest) {
    ngraph::Shape input_shape { 120, 4 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto shapeof1_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);
        auto shapeof2_i64 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i64);
        auto shapeof3_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);
        auto shapeof4_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);
        auto shapeof5_i64 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i64);
        auto shapeof6_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);
        auto shapeof7_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);

        auto shapeof1_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof1_i32, ngraph::element::i64);
        auto shapeof3_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof3_i32, ngraph::element::i64);
        auto shapeof4_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof4_i32, ngraph::element::i64);
        auto shapeof6_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof6_i32, ngraph::element::i64);
        auto shapeof7_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof7_i32, ngraph::element::i64);

        ngraph::OutputVector inputs_of_concat {shapeof1_i32_convert, shapeof2_i64, shapeof3_i32_convert, shapeof4_i32_convert,
                                               shapeof5_i64, shapeof6_i32_convert, shapeof7_i32_convert};

        auto concat = std::make_shared<ngraph::opset8::Concat>(inputs_of_concat, 0);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SharedShapeOf>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto shapeof1_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);
        auto shapeof2_i64 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i64);

        auto shapeof1_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof1_i32, ngraph::element::i64);
        auto shapeof3_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof1_i32, ngraph::element::i64);
        auto shapeof4_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof1_i32, ngraph::element::i64);
        auto shapeof6_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof1_i32, ngraph::element::i64);
        auto shapeof7_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof1_i32, ngraph::element::i64);

        ngraph::OutputVector inputs_of_concat {shapeof1_i32_convert, shapeof2_i64, shapeof3_i32_convert, shapeof4_i32_convert,
                                               shapeof2_i64, shapeof6_i32_convert, shapeof7_i32_convert};

        auto concat = std::make_shared<ngraph::opset8::Concat>(inputs_of_concat, 0);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SharedShapeOfTestI64Only) {
    ngraph::Shape input_shape { 120, 4 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto shapeof1_i64 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i64);
        auto shapeof2_i64 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i64);
        auto shapeof3_i64 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i64);

        ngraph::OutputVector inputs_of_concat {shapeof1_i64, shapeof2_i64, shapeof3_i64};

        auto concat = std::make_shared<ngraph::opset8::Concat>(inputs_of_concat, 0);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SharedShapeOf>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto shapeof1_i64 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i64);

        ngraph::OutputVector inputs_of_concat {shapeof1_i64, shapeof1_i64, shapeof1_i64};

        auto concat = std::make_shared<ngraph::opset8::Concat>(inputs_of_concat, 0);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SharedShapeOfTestI32Only) {
    ngraph::Shape input_shape { 120, 4 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto shapeof1_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);
        auto shapeof2_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);
        auto shapeof3_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);
        auto shapeof4_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);
        auto shapeof5_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);

        auto shapeof1_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof1_i32, ngraph::element::i64);
        auto shapeof2_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof2_i32, ngraph::element::i64);
        auto shapeof3_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof3_i32, ngraph::element::i64);
        auto shapeof4_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof4_i32, ngraph::element::i64);
        auto shapeof5_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof5_i32, ngraph::element::i64);

        ngraph::OutputVector inputs_of_concat {shapeof1_i32_convert, shapeof2_i32_convert, shapeof3_i32_convert,
                                               shapeof4_i32_convert, shapeof5_i32_convert};

        auto concat = std::make_shared<ngraph::opset8::Concat>(inputs_of_concat, 0);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SharedShapeOf>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto shapeof1_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);

        auto shapeof1_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof1_i32, ngraph::element::i64);
        auto shapeof2_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof1_i32, ngraph::element::i64);
        auto shapeof3_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof1_i32, ngraph::element::i64);
        auto shapeof4_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof1_i32, ngraph::element::i64);
        auto shapeof5_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof1_i32, ngraph::element::i64);

        ngraph::OutputVector inputs_of_concat {shapeof1_i32_convert, shapeof2_i32_convert, shapeof3_i32_convert,
                                               shapeof4_i32_convert, shapeof5_i32_convert};

        auto concat = std::make_shared<ngraph::opset8::Concat>(inputs_of_concat, 0);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SharedShapeOfTestMixed) {
    ngraph::Shape input_shape { 120, 4 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto shapeof1 = std::make_shared<ngraph::opset1::ShapeOf>(input);
        auto shapeof2_i64 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i64);
        auto shapeof3_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);
        auto shapeof4 = std::make_shared<ngraph::opset1::ShapeOf>(input);
        auto shapeof5_i64 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i64);
        auto shapeof6_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);
        auto shapeof7_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);

        auto shapeof3_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof3_i32, ngraph::element::i64);
        auto shapeof6_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof6_i32, ngraph::element::i64);
        auto shapeof7_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof7_i32, ngraph::element::i64);

        ngraph::OutputVector inputs_of_concat {shapeof1, shapeof2_i64, shapeof3_i32_convert, shapeof4,
                                               shapeof5_i64, shapeof6_i32_convert, shapeof7_i32_convert};

        auto concat = std::make_shared<ngraph::opset8::Concat>(inputs_of_concat, 0);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SharedShapeOf>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto shapeof1 = std::make_shared<ngraph::opset1::ShapeOf>(input);
        auto shapeof2_i32 = std::make_shared<ngraph::opset8::ShapeOf>(input, ngraph::element::i32);

        auto shapeof3_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof2_i32, ngraph::element::i64);
        auto shapeof6_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof2_i32, ngraph::element::i64);
        auto shapeof7_i32_convert = std::make_shared<ngraph::opset8::Convert>(shapeof2_i32, ngraph::element::i64);

        ngraph::OutputVector inputs_of_concat {shapeof1, shapeof1, shapeof3_i32_convert, shapeof1,
                                               shapeof1, shapeof6_i32_convert, shapeof7_i32_convert};

        auto concat = std::make_shared<ngraph::opset8::Concat>(inputs_of_concat, 0);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
    }
}
