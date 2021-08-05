// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mul_conv_fusion.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "shared_test_classes/subgraph/mul_conv_fusion.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string MulConvFusion::getTestCaseName(const testing::TestParamInfo<MulConvFusionParams> &obj) {
    ngraph::NodeTypeInfo conv_type;
    ngraph::Shape input_shape, weights_shape, const_shape;
    ngraph::element::Type precision;
    std::string device;
    std::tie(conv_type, input_shape, weights_shape, const_shape, precision, std::ignore, device) = obj.param;
    std::ostringstream results;

    results << conv_type.name << "_";
    results << "input" << CommonTestUtils::vec2str(input_shape) << "_";
    results << "weights" << CommonTestUtils::vec2str(weights_shape) << "_";
    results << "const" << CommonTestUtils::vec2str(const_shape) << "_";
    results << "precision=" << precision << "_";
    results << "device=" << device;
    return results.str();
}

void MulConvFusion::SetUp() {
    ngraph::NodeTypeInfo conv_type;
    ngraph::Shape input_shape, weights_shape, const_shape;
    ngraph::element::Type precision;
    bool is_negative;
    std::tie(conv_type, input_shape, weights_shape, const_shape, precision, is_negative, targetDevice) = this->GetParam();
    auto param = std::make_shared<ngraph::opset8::Parameter>(precision, input_shape);
    auto spatial_dims = input_shape.size() - 2;

    auto mul_const = ngraph::builder::makeConstant<float>(precision, const_shape, {}, true);
    auto mul = std::make_shared<ngraph::opset8::Multiply>(param, mul_const);
    ngraph::Shape strides(spatial_dims, 1);
    std::vector<ptrdiff_t> pad_begin(spatial_dims, 0), pad_end(spatial_dims, 0);
    auto weights = ngraph::builder::makeConstant<float>(precision, weights_shape, {}, true);
    std::shared_ptr<ngraph::Node> conv;
    if (conv_type == ngraph::opset8::Convolution::type_info) {
        conv = std::make_shared<ngraph::opset8::Convolution>(mul, weights, strides, pad_begin, pad_end, strides);
    } else if (conv_type == ngraph::opset8::GroupConvolution::type_info) {
        conv = std::make_shared<ngraph::opset8::GroupConvolution>(mul, weights, strides, pad_begin, pad_end, strides);
    } else if (conv_type == ngraph::opset8::ConvolutionBackpropData::type_info) {
        conv = std::make_shared<ngraph::opset8::ConvolutionBackpropData>(mul, weights, strides, pad_begin, pad_end, strides);
    } else if (conv_type == ngraph::opset8::GroupConvolutionBackpropData::type_info) {
        conv = std::make_shared<ngraph::opset8::GroupConvolutionBackpropData>(mul, weights, strides, pad_begin, pad_end, strides);
    } else {
        throw ngraph::ngraph_error("Unsupported type");
    }

    function = std::make_shared<ngraph::Function>(ngraph::OutputVector{conv}, ngraph::ParameterVector{param});
    auto cloned_function = ngraph::clone_function(*function);

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::MultiplyConvolutionFusion>();
    manager.run_passes(cloned_function);

    bool functions_equal = false;
    std::tie(functions_equal, std::ignore) = compare_functions(function, cloned_function, true);
    if (!is_negative) {
        ASSERT_EQ(cloned_function->get_ops().size(), 4);
        ASSERT_FALSE(functions_equal);
    } else {
        ASSERT_TRUE(functions_equal);
    }
}
} // namespace SubgraphTestsDefinitions
