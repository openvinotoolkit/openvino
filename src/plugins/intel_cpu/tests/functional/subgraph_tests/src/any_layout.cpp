// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/shape.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        ov::Shape,
        std::map<std::string, std::string>   // Device config
> AnyLayoutTestParamsSet;


class AnyLayoutOnInputsAndOutputs : public ::testing::TestWithParam<AnyLayoutTestParamsSet> {
public:
    static std::string getTestCaseName(::testing::TestParamInfo<AnyLayoutTestParamsSet> obj) {
        std::ostringstream result;
        ov::Shape shape;
        std::map<std::string, std::string> additionalConfig;
        std::tie(shape, additionalConfig) = obj.param;
        result << "shape=" << shape;
        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second;
            }
        }

        return result.str();
    }

protected:
    std::shared_ptr<ngraph::Function>
    create_test_function(const ov::Shape & shape) {
        auto param = std::make_shared<ngraph::op::Parameter>(ov::element::f32, shape);

        float shift = 1.0f;
        auto shift_node = std::make_shared<ngraph::op::Constant>(ov::element::f32, ov::Shape{1}, &shift);

        auto add = std::make_shared<ngraph::op::v1::Add>(param, shift_node);

        auto result = std::make_shared<ngraph::op::Result>(add);

        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param});
    }

    void Run() {
        ov::Shape shape;
        std::map<std::string, std::string> additionalConfig;
        std::tie(shape, additionalConfig) = GetParam();
        auto shape_size = ov::shape_size(shape);

        std::vector<float> input_data(shape_size, 2);
        std::vector<float> output_data(shape_size);
        std::vector<float> expected_output(shape_size, 3);

        // Create CNNNetwork
        auto ngraph_function = create_test_function(shape);
        auto cnn = InferenceEngine::CNNNetwork(ngraph_function);

        // Fill inputs and outputs
        std::vector<std::string> input_names;
        std::vector<std::string> out_names;
        for (const auto& it : cnn.getInputsInfo()) {
            input_names.push_back(it.first);
        }
        for (const auto& it : cnn.getOutputsInfo()) {
            out_names.push_back(it.first);
        }

        BlobMap inputBlobs;
        BlobMap outputBlobs;

        TensorDesc tensorDescInp1(Precision::FP32, shape, Layout::ANY);
        TensorDesc tensorDescOut(Precision::FP32, shape, Layout::ANY);

        inputBlobs[input_names[0]] = make_shared_blob<float>(tensorDescInp1, input_data.data());
        outputBlobs[out_names[0]]  = make_shared_blob<float>(tensorDescOut, output_data.data());

        // Load network
        Core ie;
        ExecutableNetwork executable_network = ie.LoadNetwork(cnn, "CPU", additionalConfig);

        // Infer
        InferRequest infer_request = executable_network.CreateInferRequest();
        infer_request.SetInput(inputBlobs);
        infer_request.SetOutput(outputBlobs);
        infer_request.Infer();

        ASSERT_EQ(output_data, expected_output);
    }
};

TEST_P(AnyLayoutOnInputsAndOutputs, CheckExpectedResult) {
    Run();
}

static std::vector<ov::Shape> AnyLayoutOnInputsAndOutputsParams = {
    ov::Shape{ 1, 2, 3, 4 },
    ov::Shape{ 1, 2, 3, 4, 5 },
    ov::Shape{ 1, 2, 3, 4, 5, 6 },
};


INSTANTIATE_TEST_SUITE_P(AnyLayoutOnInputsAndOutputs,
                         AnyLayoutOnInputsAndOutputs,
                         ::testing::Combine(
                             ::testing::ValuesIn(AnyLayoutOnInputsAndOutputsParams),
                             ::testing::ValuesIn({cpuEmptyPluginConfig, cpuFP16PluginConfig})),
                         AnyLayoutOnInputsAndOutputs::getTestCaseName);

}   // namespace SubgraphTestsDefinitions
