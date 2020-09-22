// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <ie_core.hpp>
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ie_preprocess.hpp"
#include "functional_test_utils/behavior_test_utils.hpp"

namespace {
void setInputNetworkPrecision(InferenceEngine::CNNNetwork &network, InferenceEngine::InputsDataMap &inputs_info,
                              InferenceEngine::Precision input_precision) {
    inputs_info = network.getInputsInfo();
    ASSERT_EQ(1u, inputs_info.size());
    inputs_info.begin()->second->setPrecision(input_precision);
}

}

namespace BehaviorTestsDefinitions {

using  PreprocessingPrecisionConvertParams = std::tuple<
        InferenceEngine::Precision,         // Input precision
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
>;

struct PreprocessingPrecisionConvertTest :
        public testing::WithParamInterface<PreprocessingPrecisionConvertParams>,
        LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PreprocessingPrecisionConvertParams> obj) {
        InferenceEngine::Precision  inPrc;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(inPrc, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "inPRC=" << inPrc.name() << "_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_" << configItem.second << "_";
            }
        }
        return result.str();
    }

    void SetUp() override {
        // This test:
        // - Strive to test the plugin internal preprocessing (precision conversion) only.
        //   Thus (logically) no-op graph is used.
        // - Reference code mimic the preprocessing via extra ngraph Convert operation.
        // - Create/uses two (different) graphs here : one to feed the the plugin and one calculate the reference result.

        SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);

        std::tie(inPrc, targetDevice, configuration) = this->GetParam();

        bool specialZero = true;

        std::vector<size_t> inputShape    {4, 4};

        auto make_ngraph = [&](bool with_extra_conv) {
            auto in_prec = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(with_extra_conv ? inPrc : decltype(inPrc)(InferenceEngine::Precision::FP32));
            auto paramsIn = ngraph::builder::makeParams(in_prec, {inputShape});
            auto paramIn = ngraph::helpers::convert2OutputVector(
                    ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));

            auto toF32 = std::make_shared<ngraph::opset1::Convert>(paramIn[0], ngraph::element::Type_t::f32);

            auto constNode = std::make_shared<ngraph::opset1::Constant>(
                    ngraph::element::Type_t::i64, ngraph::Shape{inputShape.size()}, inputShape);
            auto reshape = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(
                    std::make_shared<ngraph::opset1::Reshape>(with_extra_conv ? toF32 : paramIn[0], constNode, specialZero));
            ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reshape)};
            return std::make_shared<ngraph::Function>(results, paramsIn, "Reshape");
        };

        function            = make_ngraph(false);
        reference_function  = make_ngraph(true);  //use extra ops to mimic the preprocessing
    }

    void Validate() override {
        //force the reference implementation to use graph with extra Convert operation
        function = reference_function;
        LayerTestsUtils::LayerTestsCommon::Validate();
    }

public:
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> reference_function;
};


TEST_P(PreprocessingPrecisionConvertTest, InternalPluginPrecisionConvert) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}
}  // namespace BehaviorTestsDefinitions
