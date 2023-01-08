// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <vpu/ngraph/transformations/convert_transpose_presicion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, TransposeWithU8DataConvertToTransposeWithFP16Data) {
    // needs to be converted, the presicion after transformation will be changed,
    // transpose (wrong precision) -> convert (input) -> transpose.
    auto element_type_first = ngraph::element::Type_t::f16;
    auto element_type_sec = ngraph::element::Type_t::u8;

    ngraph::Shape input_shape { 1, 100, 120, 150 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(element_type_sec, input_shape);
        auto transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, { 0, 3, 1, 2 });
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(input, transpose_perm);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ transpose }, ngraph::ParameterVector{ input });
        manager.register_pass<vpu::ConvertTransposePrecision>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(element_type_sec, input_shape);
        auto first_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, { 0, 3, 1, 2 });
        auto first_transpose = std::make_shared<ngraph::opset8::Transpose>(input, first_transpose_perm);
        ov::Output<ov::Node> new_in = first_transpose->input_value(0);
        auto convert = std::make_shared<ngraph::opset1::Convert>(new_in, element_type_first);
        auto transpose_node_fp16 = std::make_shared<ngraph::opset8::Transpose>(convert, first_transpose_perm);
        disable_rt_info_check();

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ transpose_node_fp16 }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, TransposeWithFP16DataDoNotConvertTranspose) {
    // no need to use convert, the presicion after transformation will be tha same.
    auto element_type_first = ngraph::element::Type_t::f16;

    ngraph::Shape input_shape { 1, 100, 120, 150 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(element_type_first, input_shape);
        auto transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, { 0, 3, 1, 2 });
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(input, transpose_perm);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ transpose }, ngraph::ParameterVector{ input });
        manager.register_pass<vpu::ConvertTransposePrecision>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(element_type_first, input_shape);
        auto first_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, { 0, 3, 1, 2 });
        auto first_transpose = std::make_shared<ngraph::opset8::Transpose>(input, first_transpose_perm);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ first_transpose }, ngraph::ParameterVector{ input });
    }
}
