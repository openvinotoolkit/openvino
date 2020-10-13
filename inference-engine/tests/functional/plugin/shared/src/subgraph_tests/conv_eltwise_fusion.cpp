// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <debug.h>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_tests/conv_eltwise_fusion.hpp"
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/op_conversions/convert_convolutions.hpp>
#include <transformations/common_optimizations/conv_bias_fusion.hpp>

namespace LayerTestsDefinitions {

std::string ConvEltwiseFusion::getTestCaseName(const testing::TestParamInfo<ConvEltwiseFusionParams> &obj) {
    ngraph::NodeTypeInfo conv_type, eltwise_type;
    std::tuple<ngraph::NodeTypeInfo, int64_t> t;
    ngraph::Shape input_shape, weights_shape, const_shape;
    ngraph::element::Type precision;
    std::string targetName;
    int64_t expected_number_of_ops;
    std::tie(conv_type, t, input_shape, weights_shape, const_shape, precision, targetName) = obj.param;
    std::tie(eltwise_type, expected_number_of_ops) = t;
    std::ostringstream results;

    results << conv_type.name << "_";
    results << eltwise_type.name << "_";
    results << "Input" << CommonTestUtils::vec2str(input_shape);
    results << "Weights" << CommonTestUtils::vec2str(weights_shape);
    results << "Const" << CommonTestUtils::vec2str(const_shape);
    results << "netPRC=" << precision << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void ConvEltwiseFusion::SetUp() {
    ngraph::NodeTypeInfo conv_type, eltwise_type;
    std::tuple<ngraph::NodeTypeInfo, int64_t> t;
    ngraph::Shape input_shape, weights_shape, const_shape;
    ngraph::element::Type precision;
    int64_t expected_number_of_ops;
    std::tie(conv_type, t, input_shape, weights_shape, const_shape, precision, targetDevice) = this->GetParam();
    std::tie(eltwise_type, expected_number_of_ops) = t;
    ngraph::pass::Manager manager;
    {
        auto param = std::make_shared<ngraph::opset4::Parameter>(precision, input_shape);
        auto spatial_dims = input_shape.size() - 2;

        ngraph::Shape strides(spatial_dims, 1);
        std::vector<ptrdiff_t> pad_begin(spatial_dims, 0), pad_end(spatial_dims, 0);
        auto weights = ngraph::builder::makeConstant<float>(precision, weights_shape, {}, true);
        auto eltwise_const = ngraph::builder::makeConstant<float>(precision, const_shape, {}, true);
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

        std::shared_ptr<ngraph::Node> eltwise;
        if (eltwise_type == ngraph::opset4::Multiply::type_info) {
            eltwise = std::make_shared<ngraph::opset4::Multiply>(conv, eltwise_const);
            manager.register_pass<ngraph::pass::ConvolutionMultiplyFusion>();
            manager.register_pass<ngraph::pass::GroupConvolutionMultiplyFusion>();
            manager.register_pass<ngraph::pass::ConvolutionBackpropDataMultiplyFusion>();
            manager.register_pass<ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion>();
        } else if (eltwise_type == ngraph::opset4::Add::type_info) {
            eltwise = std::make_shared<ngraph::opset4::Add>(conv, eltwise_const);
            manager.register_pass<ngraph::pass::ConvertConvolutions>();
            manager.register_pass<ngraph::pass::ConvFusion>();
        } else {
            throw ngraph::ngraph_error("Unsupported type");
        }

        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{eltwise}, ngraph::ParameterVector{param}, "conv_eltwise");
    }

    manager.register_pass<ngraph::pass::ConstantFolding>();

    auto cloned_function = ngraph::clone_function(*function);
    manager.run_passes(cloned_function);

    ASSERT_EQ(cloned_function->get_ops().size(), expected_number_of_ops);
}

TEST_P(ConvEltwiseFusion, CompareWithRefs) {
    Run();
}
} // namespace LayerTestsDefinitions
