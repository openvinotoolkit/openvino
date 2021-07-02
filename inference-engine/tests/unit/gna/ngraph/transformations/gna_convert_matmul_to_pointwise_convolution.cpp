// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <tuple>

#include "transformations/convert_matmul_to_pointwise_convolution.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include <ngraph/pass/visualize_tree.hpp> // DEBUG

namespace testing {

// TODO: check MatMul input != 2 or output != 2 rank

/*
FIXME: error in ConvertMatmulToPointWiseConvolution
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{16, 4}, {1});
        auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(input_params, constant);

        auto result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::ConvertMatmulToPointWiseConvolution>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
*/

TEST(TransformationTests, ConvertMatmulToPointWiseConvolutionTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{16, 8};
    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        auto constant_node = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{8, 8}, {1});
        auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(input_params, constant_node);

        auto result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::ConvertMatmulToPointWiseConvolution>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);
        auto constant_node = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{8, 8}, {1});

        auto const_reshape_before = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                            ngraph::Shape{4},
                                                                            ngraph::Shape{1, 1, 16, 8});
        auto reshape_before =  std::make_shared<ngraph::opset7::Reshape>(input_params, const_reshape_before, false);
        
        auto const_transpose_before = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                                        ngraph::Shape{4},
                                                                        ngraph::Shape{0, 3, 1, 2});
        auto transpose_before = std::make_shared<ngraph::opset7::Transpose>(reshape_before, const_transpose_before);

        auto weights_reshape_const = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                                ngraph::Shape{4}, ngraph::Shape{8, 8, 1, 1});
        auto weights_reshaped =  std::make_shared<ngraph::opset7::Reshape>(constant_node, weights_reshape_const, false);

        auto conv_node = std::make_shared<ngraph::opset7::Convolution>(transpose_before,
                                                                        weights_reshaped,
                                                                        ngraph::Strides{1, 1},
                                                                        ngraph::CoordinateDiff{0, 0},
                                                                        ngraph::CoordinateDiff{0, 0},
                                                                        ngraph::Strides{1, 1},
                                                                        ngraph::op::PadType::VALID);
        
        auto const_transpose_after = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                                      ngraph::Shape{4},
                                                                      ngraph::Shape{0, 2, 3, 1});
        auto transpose_after = std::make_shared<ngraph::opset7::Transpose>(conv_node, const_transpose_after);

        auto const_reshape_after = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                              ngraph::Shape{2},
                                                                              ngraph::Shape{16, 8});
        auto reshape_after =  std::make_shared<ngraph::opset7::Reshape>(transpose_after, const_reshape_after, false);

        auto result = std::make_shared<ngraph::opset7::Result>(reshape_after);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ConvertMatmulToPointWiseConvolutionFqTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{16, 8};
    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        auto constant_node = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{8, 8}, {1});

        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fq_node = std::make_shared<ngraph::opset7::FakeQuantize>(constant_node, input_low,
                                                                      input_high, output_low,
                                                                      output_high, 11);

        auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(input_params, fq_node);

        auto result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::ConvertMatmulToPointWiseConvolution>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);
        auto constant_node = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{8, 8}, {1});

        auto const_reshape_before = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                            ngraph::Shape{4},
                                                                            ngraph::Shape{1, 1, 16, 8});
        auto reshape_before =  std::make_shared<ngraph::opset7::Reshape>(input_params, const_reshape_before, false);
        
        auto const_transpose_before = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                                        ngraph::Shape{4},
                                                                        ngraph::Shape{0, 3, 1, 2});
        auto transpose_before = std::make_shared<ngraph::opset7::Transpose>(reshape_before, const_transpose_before);

        // FQ node between input constant and reshape
        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fq_node = std::make_shared<ngraph::opset7::FakeQuantize>(constant_node, input_low,
                                                                      input_high, output_low,
                                                                      output_high, 11);
        //

        auto weights_reshape_const = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                                ngraph::Shape{4}, ngraph::Shape{8, 8, 1, 1});
        auto weights_reshaped =  std::make_shared<ngraph::opset7::Reshape>(fq_node, weights_reshape_const, false);

        auto conv_node = std::make_shared<ngraph::opset7::Convolution>(transpose_before,
                                                                        weights_reshaped,
                                                                        ngraph::Strides{1, 1},
                                                                        ngraph::CoordinateDiff{0, 0},
                                                                        ngraph::CoordinateDiff{0, 0},
                                                                        ngraph::Strides{1, 1},
                                                                        ngraph::op::PadType::VALID);
        
        auto const_transpose_after = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                                      ngraph::Shape{4},
                                                                      ngraph::Shape{0, 2, 3, 1});
        auto transpose_after = std::make_shared<ngraph::opset7::Transpose>(conv_node, const_transpose_after);

        auto const_reshape_after = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                              ngraph::Shape{2},
                                                                              ngraph::Shape{16, 8});
        auto reshape_after =  std::make_shared<ngraph::opset7::Reshape>(transpose_after, const_reshape_after, false);

        auto result = std::make_shared<ngraph::opset7::Result>(reshape_after);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}


} // namespace testing
