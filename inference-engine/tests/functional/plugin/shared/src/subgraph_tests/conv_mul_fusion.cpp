// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <debug.h>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_tests/conv_mul_fusion.hpp"
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/pass/constant_folding.hpp>

namespace LayerTestsDefinitions {

std::string ConvMultiply::getTestCaseName(const testing::TestParamInfo<ConvMultiplyParams> &obj) {
    ngraph::NodeTypeInfo conv_type;
    ngraph::Shape input_shape, weights_shape, const_shape;
    ngraph::element::Type precision;
    std::string targetName;
    int64_t expected_number_of_ops;
    std::tie(conv_type, input_shape, weights_shape, const_shape, expected_number_of_ops, precision, targetName) = obj.param;
    std::ostringstream results;

    results << conv_type.name << "_";
    results << "Input" << CommonTestUtils::vec2str(input_shape);
    results << "Weights" << CommonTestUtils::vec2str(weights_shape);
    results << "Const" << CommonTestUtils::vec2str(const_shape);
    results << "netPRC=" << precision << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void ConvMultiply::SetUp() {
    ngraph::NodeTypeInfo conv_type;
    ngraph::Shape input_shape, weights_shape, const_shape;
    ngraph::element::Type precision;
    int64_t expected_number_of_ops;
    std::tie(conv_type, input_shape, weights_shape, const_shape, expected_number_of_ops, precision, targetDevice) = this->GetParam();

    {
        auto param = std::make_shared<ngraph::opset4::Parameter>(precision, input_shape);
        auto spatial_dims = input_shape.size() - 2;

        ngraph::Shape strides(spatial_dims, 1);
        std::vector<ptrdiff_t> pad_begin(spatial_dims, 0), pad_end(spatial_dims, 0);
        auto weights = ngraph::builder::makeConstant(precision, weights_shape, {}, true);
        auto mul_const = ngraph::builder::makeConstant(precision, const_shape, {}, true);
        std::shared_ptr<ngraph::Node> conv;
        if (conv_type == ngraph::opset4::Convolution::type_info) {
            conv = std::make_shared<ngraph::opset4::Convolution>(param, weights, strides, pad_begin, pad_end, strides);
        } else if (conv_type == ngraph::opset4::GroupConvolution::type_info) {
            conv = std::make_shared<ngraph::opset4::GroupConvolution>(param, weights, strides, pad_begin, pad_end, strides);
        } else if (conv_type == ngraph::opset4::ConvolutionBackpropData::type_info) {
            conv = std::make_shared<ngraph::opset4::ConvolutionBackpropData>(param, weights, strides, pad_begin, pad_end, strides);
        } else if (conv_type == ngraph::opset4::GroupConvolutionBackpropData::type_info) {
            conv = std::make_shared<ngraph::opset4::GroupConvolutionBackpropData>(param, weights, strides, pad_begin, pad_end, strides);
        } else {
            throw ngraph::ngraph_error("Unsupported type");
        }

        auto mul = std::make_shared<ngraph::opset4::Multiply>(conv, mul_const);

        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{mul}, ngraph::ParameterVector{param}, "conv_multiply");
    }

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::ConvolutionMultiplyFusion>();
    manager.register_pass<ngraph::pass::GroupConvolutionMultiplyFusion>();
    manager.register_pass<ngraph::pass::ConvolutionBackpropDataMultiplyFusion>();
    manager.register_pass<ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.run_passes(function);

    ASSERT_EQ(function->get_ops().size(), expected_number_of_ops);
}

TEST_P(ConvMultiply, CompareWithRefs) {
    Run();
}
} // namespace LayerTestsDefinitions
