// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <ie_core.hpp>
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ie_preprocess.hpp"
#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {

using  PreprocessingPrecisionConvertParams = std::tuple<
        InferenceEngine::Precision,         // Input precision
        unsigned,                           // channels number
        bool,                               // Use normal (i.e. SetInput() or unusal i.e. GetBlob()) inut method
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
>;

struct PreprocessingPrecisionConvertTest :
        public testing::WithParamInterface<PreprocessingPrecisionConvertParams>,
        LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PreprocessingPrecisionConvertParams> obj) {
        InferenceEngine::Precision  inPrc;
        bool useSetInput;
        unsigned channels;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(inPrc, channels, useSetInput, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "inPRC=" << inPrc.name() << "_";
        result << channels << "Ch" << "_";
        result << (useSetInput ? "SetInput" : "GetBlob") << "_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_" << configItem.second << "_";
            }
        }
        return result.str();
    }

    // Need to override Infer() due to usage of GetBlob() as input method.
    // Mostly a copy of LayerTestsCommon::Infer()
    void Infer() override {
        inferRequest = executableNetwork.CreateInferRequest();
        inputs.clear();

        for (const auto &input : executableNetwork.GetInputsInfo()) {
            const auto &info = input.second;
            auto blob = GenerateInput(*info);
            if (!use_set_input) {
                InferenceEngine::Blob::Ptr input = inferRequest.GetBlob(info->name());
                blob_copy(blob, input);
            } else {
                inferRequest.SetBlob(info->name(), blob);
            }

            inputs.push_back(blob);
        }
        if (configuration.count(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED) &&
            configuration.count(InferenceEngine::PluginConfigParams::YES)) {
            auto batchSize = executableNetwork.GetInputsInfo().begin()->second->getTensorDesc().getDims()[0] / 2;
            inferRequest.SetBatch(batchSize);
        }
        inferRequest.Infer();
    }

    void SetUp() override {
        // This test:
        // - Strive to test the plugin internal preprocessing (precision conversion) only.
        //   Thus (logically) no-op graph is used.
        // - Reference code mimic the preprocessing via extra ngraph Convert operation.
        // - Create/uses two (different) graphs here : one to feed the the plugin and one calculate the reference result.

        SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);

        std::tie(inPrc, channels, use_set_input, targetDevice, configuration) = this->GetParam();
        outPrc = inPrc;

        bool specialZero = true;

        std::vector<size_t> inputShape(channels, 4);

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
        // w/a: copy of original function is required to provide correct op coverage report (overflow of convert counter issue)
        auto copyOriginalFunction = function;
        //force the reference implementation to use graph with extra Convert operation
        function = reference_function;
        LayerTestsUtils::LayerTestsCommon::Validate();
        function = copyOriginalFunction;
    }

public:
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> reference_function;
    bool use_set_input = true;
    unsigned channels = 0;
};


TEST_P(PreprocessingPrecisionConvertTest, InternalPluginPrecisionConvert) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}
}  // namespace BehaviorTestsDefinitions
