// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   std::vector<size_t>                  // shape to split
                   >
    convAlignFilterParams;

namespace LayerTestsDefinitions {

class ConvolutionAlignFilterTest : public testing::WithParamInterface<convAlignFilterParams>,
                                   public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convAlignFilterParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> splitInputShape;
        std::tie(netPrecision, targetDevice, configuration, splitInputShape) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_SIS=" << ov::test::utils::vec2str(splitInputShape);
        return result.str();
    }

protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> splitInputShape;
        std::tie(netPrecision, targetDevice, configuration, splitInputShape) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        size_t in_total_dims_size =
            std::accumulate(std::begin(splitInputShape), std::end(splitInputShape), 1, std::multiplies<size_t>());
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, in_total_dims_size})};
        auto pattern1 =
            std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{2}, splitInputShape);
        auto reshape1 = std::make_shared<ngraph::opset8::Reshape>(params[0], pattern1, false);
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto split = ngraph::builder::makeSplit(reshape1, ngPrc, 2, 0);
        OPENVINO_SUPPRESS_DEPRECATED_END

        auto relu1 = std::make_shared<ngraph::opset8::Relu>(split->output(0));
        auto relu2 = std::make_shared<ngraph::opset8::Relu>(split->output(1));

        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{relu1, relu2}, 0);
        auto pattern2 = std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{2},
                                                                   ngraph::Shape{1, in_total_dims_size});
        auto reshape2 = std::make_shared<ngraph::opset8::Reshape>(concat, pattern2, false);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(reshape2)};
        function = std::make_shared<ngraph::Function>(results, params, "ConvAlignFilter");
        functionRefs = ngraph::clone_function(*function);
    }
};

TEST_P(ConvolutionAlignFilterTest, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
                                                                 {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

const std::vector<std::vector<size_t>> shapes{{16, 24}, {8, 32}};

INSTANTIATE_TEST_SUITE_P(smoke_conv_align_filter,
                         ConvolutionAlignFilterTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(shapes)),
                         ConvolutionAlignFilterTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
