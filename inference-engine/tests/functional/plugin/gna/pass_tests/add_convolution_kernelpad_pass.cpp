// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

typedef std::tuple<
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>  //Configuration
>  addConvolutionKernelPadParams;

namespace LayerTestsDefinitions {

class AddConvolutionKernelPadPassTest : public testing::WithParamInterface<addConvolutionKernelPadParams>,
                                        public LayerTestsUtils::LayerTestsCommon {
    public:
        static std::string getTestCaseName(testing::TestParamInfo<addConvolutionKernelPadParams> obj) {
            InferenceEngine::Precision netPrecision;
            std::string targetDevice;
            std::map<std::string, std::string> configuration;
            std::tie(netPrecision, targetDevice, configuration) = obj.param;

            std::ostringstream result;
            result << "netPRC=" << netPrecision.name() << "_";
            result << "targetDevice=" << targetDevice << "_";
            for (auto const& configItem : configuration) {
                result << "_configItem=" << configItem.first << "_" << configItem.second;
            }
            return result.str();
        }

    protected:
        void SetUp() override {
            InferenceEngine::Precision netPrecision;
            std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
            auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

            auto params = ngraph::builder::makeParams(ngPrc, { {1, 1, 1, 512 } });

            auto conv1 = ngraph::builder::makeConvolution(params[0], ngPrc, { 1, 297 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                ngraph::op::PadType::VALID, 32);

            ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(conv1) };
            function = std::make_shared<ngraph::Function>(results, params, "AddConvolutionKernelPadPass");
        }
};

    TEST_P(AddConvolutionKernelPadPassTest, CompareWithRefImpl) {
        Run();
    };

    const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
        {
            {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
            {"GNA_SCALE_FACTOR_0", "1638.4"}
        }
    };

    //Item value was not in supported set of values
    INSTANTIATE_TEST_CASE_P(DISABLED_ConvolutionPass, AddConvolutionKernelPadPassTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs)),
        AddConvolutionKernelPadPassTest::getTestCaseName);

} // namespace LayerTestsDefinitions

