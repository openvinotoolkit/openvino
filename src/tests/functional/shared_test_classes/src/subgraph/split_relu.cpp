// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/split_relu.hpp"

namespace SubgraphTestsDefinitions {
    std::string SplitRelu::getTestCaseName(const testing::TestParamInfo<SplitReluTuple> &obj) {
        std::vector<std::vector<size_t>> input;
        std::vector<size_t > connect_input;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::map<std::string, std::string> additional_config;
        std::tie(input, connect_input, netPrecision, targetName, additional_config) = obj.param;
        std::ostringstream results;

        results << "IS=" << ov::test::utils::vec2str(input[0]) << "_";
        results << "ConnectInput=" << ov::test::utils::vec2str(connect_input) << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        return results.str();
    }

    void SplitRelu::SetUp() {
        std::vector<std::vector<size_t>> inputs;
        std::vector<size_t> connect_index;
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> additional_config;
        std::tie(inputs, connect_index, netPrecision, targetDevice, additional_config) = this->GetParam();
        configuration.insert(additional_config.begin(), additional_config.end());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector input;
        for (auto&& shape : inputs) {
            input.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape)));
        }
        auto split = ngraph::builder::makeSplit(input[0], ngPrc, 4, 1);
        ngraph::ResultVector results;

        for (size_t i : connect_index) {
            auto relu = std::make_shared<ngraph::opset1::Relu>(split->output(i));
            results.push_back(std::make_shared<ngraph::opset1::Result>(relu));
        }
        function = std::make_shared<ngraph::Function>(results, input, "split_relu");
    }
} // namespace SubgraphTestsDefinitions
