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
        std::vector<size_t>,                // Input shape
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
        std::vector<size_t> inputShape;
        std::tie(inputShape, netPrecision, targetDevice, configuration) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        return result.str();
    }

//    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
//        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 1, 0, 10);
//    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::tie(inputShape, netPrecision, targetDevice, configuration) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, { {16, 2} });
        auto const_mult2 = ngraph::builder::makeConstant<float>(ngPrc, {1, 16}, {1.0f});
//        auto clamp = std::make_shared<ngraph::opset1::Clamp>(params[0], 0.f, 0.5f);

        auto const_eltwise = ngraph::builder::makeConstant<float>(ngPrc, {1, 2}, {1.0f});
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

const std::vector<std::vector<size_t>> input_shapes = {
        {128, 1},
//        {16, 2},
//        {16, 4}
};

INSTANTIATE_TEST_CASE_P(smoke_convert_matmul_to_fc, ConvertMatmulToFcPass,
                        ::testing::Combine(
                                ::testing::ValuesIn(input_shapes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(configs)),
                        ConvertMatmulToFcPass::getTestCaseName);
} // namespace LayerTestsDefinitions