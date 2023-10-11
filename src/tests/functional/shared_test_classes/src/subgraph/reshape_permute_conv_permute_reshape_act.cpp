// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/reshape_permute_conv_permute_reshape_act.hpp"

namespace SubgraphTestsDefinitions {
    std::string ConvReshapeAct::getTestCaseName(const testing::TestParamInfo<ConvReshapeActParams>& obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::array<size_t, 4> input_shape;
        std::array<size_t, 2> kernel_shape;
        size_t output_channels;
        std::map<std::string, std::string> configuration;


        std::tie(netPrecision, targetName, input_shape, kernel_shape, output_channels, configuration) = obj.param;
        std::ostringstream results;

        results << "IS=" << ov::test::utils::vec2str(std::vector<size_t>(input_shape.begin(), input_shape.end())) << "_";
        results << "KS=" << ov::test::utils::vec2str(std::vector<size_t>(kernel_shape.begin(), kernel_shape.end())) << "_";
        results << "OC=" << output_channels << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName;
        for (auto const& configItem : configuration) {
            results << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return results.str();
    }

    void ConvReshapeAct::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::array<size_t, 4> input_shape;
        std::array<size_t, 2> kernel_shape;
        size_t output_channels;
        std::map<std::string, std::string> additional_config;

        std::tie(netPrecision, targetDevice, input_shape, kernel_shape, output_channels, additional_config) = this->GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());

        const std::size_t input_dim = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        std::vector<size_t> input_dims { 1, input_dim };
        std::vector<size_t> reshape_in_dims = std::vector<size_t>(input_shape.begin(), input_shape.end());
        std::vector<size_t> permute_in_order = { 0, 3, 1, 2 };
        std::vector<size_t> permute_out_order = { 0, 2, 3, 1 };
        std::vector<size_t> reshape_out_dims = { 1, input_shape[0] * input_shape[1] * (input_shape[2] - kernel_shape[1] + 1) * output_channels };

        ov::ParameterVector input_parameter {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(input_dims))};

        auto reshape_in_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
            ngraph::Shape{4},
            reshape_in_dims);
        auto reshape_in = std::make_shared<ngraph::op::v1::Reshape>(input_parameter[0], reshape_in_pattern, false);

        auto permute_in_params = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64,
            ngraph::Shape{4},
            ngraph::Shape{permute_in_order});
        auto permute_in = std::make_shared<ngraph::opset1::Transpose>(reshape_in, permute_in_params);

        auto conv = ngraph::builder::makeConvolution(permute_in, ngPrc, {kernel_shape[0], kernel_shape[1]}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
            ngraph::op::PadType::VALID, output_channels);

        auto permute_out_params = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64,
            ngraph::Shape{4},
            permute_out_order);
        auto permute_out = std::make_shared<ngraph::opset1::Transpose>(conv, permute_out_params);

        auto reshape_out_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
            ngraph::Shape{2},
            std::vector<size_t>{reshape_out_dims});

        auto reshape_out = std::make_shared<ngraph::op::v1::Reshape>(permute_out, reshape_out_pattern, false);

        auto tanh = std::make_shared<ngraph::op::Tanh>(reshape_out);

        function = std::make_shared<ngraph::Function>(tanh, input_parameter, "conv_reshape_act");
    }

    void ConvReshapeAct::Run() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        LoadNetwork();

        inferRequest = executableNetwork.CreateInferRequest();
        inputs.clear();

        for (const auto &input : cnnNetwork.getInputsInfo()) {
            const auto &info = input.second;
            auto tensorDesc = info->getTensorDesc();

            auto blob = FuncTestUtils::createAndFillBlobFloat(tensorDesc, 2, -1, 100, 111);

            FuncTestUtils::fillInputsBySinValues(blob);
            inferRequest.SetBlob(info->name(), blob);
            inputs.push_back(blob);
        }
        inferRequest.Infer();

        threshold = 0.1;
        Validate();
    }
} // namespace SubgraphTestsDefinitions
