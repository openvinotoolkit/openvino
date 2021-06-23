// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <tuple>
#include <string>

#include <ie_core.hpp>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

typedef std::tuple<
        std::vector<std::vector<size_t>>,                // Input shape
        InferenceEngine::Precision,         // Network Precision
        std::string,                        // Target Device
        std::map<std::string, std::string>  //Configuration
> ConvertMatmulToFcPassParams;

namespace LayerTestsDefinitions {
class ConvertMatmulToFcPass : public testing::WithParamInterface<ConvertMatmulToFcPassParams >,
                              public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvertMatmulToFcPassParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<std::vector<size_t>> inputShape;
        std::tie(inputShape, netPrecision, targetDevice, configuration) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_IS=" << CommonTestUtils::vec2str(inputShape[1]) << "_";
        result << "_CS=" << CommonTestUtils::vec2str(inputShape[0]) << "_";
        return result.str();
    }

InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = CommonTestUtils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<std::vector<size_t>> inputShape;
        std::tie(inputShape, netPrecision, targetDevice, configuration) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, { inputShape[1] });
        std::vector<float> weights = CommonTestUtils::generate_float_numbers(inputShape[0][0] * inputShape[0][1], -0.2f, 0.2f);
        auto const_mult2 = ngraph::builder::makeConstant<float>(ngPrc, inputShape[0], weights);

        auto const_eltwise = ngraph::builder::makeConstant<float>(ngPrc,
                {inputShape[0][0], inputShape[1][1]}, {1.0f});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(const_mult2, params[0], false, false);

        auto eltwise = std::make_shared<ngraph::opset1::Multiply>(matmul, const_eltwise);
        function = std::make_shared<ngraph::Function>(eltwise, params, "ConvertMatmulToFC");
    }
};

TEST_P(ConvertMatmulToFcPass, CompareWithRefImpl) {
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const std::vector<std::map<std::string, std::string>> configs = {
        {
                {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
                {"GNA_COMPACT_MODE", "NO"}
        },
        {
                {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                {"GNA_COMPACT_MODE", "NO"}
        }
};

const std::vector<std::vector<std::vector<size_t>>> input_shapes = {
        {{1, 8}, {8, 1}},
        {{128, 8}, {8, 1}},
        {{8, 8}, {8, 8}},
        {{1, 16}, {16, 8}}
};


INSTANTIATE_TEST_SUITE_P(smoke_convert_matmul_to_fc, ConvertMatmulToFcPass,
                        ::testing::Combine(
                                ::testing::ValuesIn(input_shapes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(configs)),
                        ConvertMatmulToFcPass::getTestCaseName);
} // namespace LayerTestsDefinitions