// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <ie_plugin_config.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"

#include "subgraph_tests/input_conv.hpp"

#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string InputConvTest::getTestCaseName(testing::TestParamInfo<inputConvParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> inputShape;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    size_t output_channels;
    bool with_bias;
    std::tie(netPrecision, targetDevice, configuration, inputShape, output_channels, with_bias) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "OC=" << output_channels << "_";
    result << "bias=" << with_bias << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

InferenceEngine::Blob::Ptr InputConvTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
    blob->allocate();
    auto precision = info.getPrecision();

    auto* rawBlobDataPtr = blob->buffer().as<float*>();
    for (size_t i = 0; i < blob->size(); i++) {
        float value = i % 16;
        if (typeid(precision) == typeid(typename InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type)) {
            rawBlobDataPtr[i] = ngraph::float16(value).to_bits();
        } else {
            rawBlobDataPtr[i] = value;
        }
    }
    return blob;
}

void InputConvTest::SetUp() {
    auto generateWeights = [](std::size_t out_channels, std::size_t kernel_size) {
        std::vector<float> res;
        for (int i = 0; i < out_channels; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                res.emplace_back(1.0f);
            }
        }

        return res;
    };

    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    std::vector<size_t> inputShape;
    size_t output_channels;
    bool with_bias;
    std::tie(netPrecision, targetDevice, tempConfig, inputShape, output_channels, with_bias) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    size_t kernel_y = 1;
    size_t kernel_x = 9;
    auto params = ngraph::builder::makeParams(ngPrc, { inputShape });

    auto conv_0 = ngraph::builder::makeConvolution(params[0], ngPrc, { kernel_y, kernel_x }, { 1, 1 }, { 0, 0 },
        { 0, 0 }, { 1, 1 }, ngraph::op::PadType::VALID, output_channels, with_bias,
        generateWeights(output_channels, kernel_x));

    //permute accepts and return 2-byte values. If it's the last operation in the model, that's why the output is incorrect for int16 then
    size_t num_output_width = (((inputShape[1] * inputShape[2] * inputShape[3] - kernel_x * kernel_y) / inputShape[1]) + 1);
    std::vector<size_t> outFormShapes_0 = { 1, output_channels * num_output_width };
    auto pattern_0 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes_0);
    auto reshape_0 = std::make_shared<ngraph::opset1::Reshape>(conv_0, pattern_0, false);

    ngraph::ResultVector results {std::make_shared<ngraph::op::Result>(reshape_0)};
    function = std::make_shared<ngraph::Function>(results, params, "InputConvTest");
}

TEST_P(InputConvTest, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
