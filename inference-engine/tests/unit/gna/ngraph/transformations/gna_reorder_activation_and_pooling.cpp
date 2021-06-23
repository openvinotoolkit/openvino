// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/reorder_activation_and_pooling.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

namespace testing {

using AddPtr = std::shared_ptr<ngraph::opset7::Add>;
using ConstPtr = std::shared_ptr<ngraph::opset7::Constant>;
using ConvolutionPtr = std::shared_ptr<ngraph::opset7::Convolution>;
using FakeQuantizePtr = std::shared_ptr<ngraph::opset7::FakeQuantize>;
using MatMulPtr = std::shared_ptr<ngraph::opset7::MatMul>;
using ParameterPtr = std::shared_ptr<ngraph::opset7::Parameter>;
using ReluPtr = std::shared_ptr<ngraph::opset7::Relu>;
using ReshapePtr = std::shared_ptr<ngraph::opset7::Reshape>;
using ResultPtr = std::shared_ptr<ngraph::opset7::Result>;
using TransposePtr = std::shared_ptr<ngraph::opset7::Transpose>;
using MaxPoolPtr = std::shared_ptr<ngraph::opset7::MaxPool>;

TEST(TransformationTests, ReorderActivationAndPooling) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        ParameterPtr input_params_convolution = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});
        ParameterPtr input_params_add = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                        ngraph::Shape{1, 3, 64, 64});                                                                        
        ConvolutionPtr convolution_operation;
        {
            auto weights = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                        ngraph::Shape{3, 3, 1, 1}, {1});
            auto bias = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                     ngraph::Shape{3, 1, 1}, {1});
            convolution_operation = std::make_shared<ngraph::opset7::Convolution>(input_params_convolution,
                                                                  weights,
                                                                  ngraph::Strides{1, 1},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::CoordinateDiff{0, 0},
                                                                  ngraph::Strides{1, 1});
        }

        AddPtr add_operation = std::make_shared<ngraph::opset7::Add>(convolution_operation,
                                                                        input_params_add);

        ReluPtr relu_operation = std::make_shared<ngraph::opset7::Relu>(add_operation);
#if 0
        MaxPoolPtr max_pool_operation = std::make_shared<ngraph::opset7::MaxPool>(relu_operation,
                                                                            ngraph::Strides{1},
                                                                            ngraph::Shape{0},
                                                                            ngraph::Shape{0},
                                                                            ngraph::Shape{3});
#endif
        ResultPtr result = std::make_shared<ngraph::opset7::Result>(relu_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_convolution, input_params_add});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::ReorderActivationAndPooling>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

}

} // namespace testing
