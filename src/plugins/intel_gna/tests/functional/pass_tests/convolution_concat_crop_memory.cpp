// Copyright (C) 2018-2022 Intel Corporation
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
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

typedef std::tuple<
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>, // Configuration
    std::vector<size_t>                 // shape to split
> cccmParams;

namespace LayerTestsDefinitions {

class CCCMTest : public testing::WithParamInterface<cccmParams>,
                                   public LayerTestsUtils::LayerTestsCommon {
    public:
    static std::string getTestCaseName(testing::TestParamInfo<cccmParams> obj) {
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
            result << "_SIS=" << CommonTestUtils::vec2str(splitInputShape);
            return result.str();
        }

    protected:
        InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
            InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
            blob->allocate();

            auto* rawBlobDataPtr = blob->buffer().as<float*>();
            std::vector<float> values = CommonTestUtils::generate_float_numbers(blob->size(), -2.f, 2.f);
            for (size_t i = 0; i < blob->size(); i++) {
                rawBlobDataPtr[i] = values[i];
            }
            return blob;
        }

        void SetUp() override {
            InferenceEngine::Precision netPrecision;
            std::vector<size_t> inputShape = {1, 64};
            std::tie(netPrecision, targetDevice, configuration, std::ignore) = this->GetParam();

            auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

            size_t in_total_dims_size = ngraph::shape_size(inputShape);
            auto params = ngraph::builder::makeParams(ngPrc, {{1, in_total_dims_size}});
            auto pattern1 =
                std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{2}, inputShape);
            auto reshape1 = std::make_shared<ngraph::opset8::Reshape>(params[0], pattern1, false);
            ngraph::VariableInfo vi{};
            vi.data_shape = ngraph::PartialShape(inputShape);
            vi.variable_id = "test_variable";
            vi.data_type = ngraph::element::Type_t::f32;
            const auto var = std::make_shared<ngraph::Variable>(vi);
            std::vector<float> initial_state = CommonTestUtils::generate_float_numbers(in_total_dims_size, -3.f, 3.f);
            auto initial_state_node =
                ngraph::builder::makeConstant(ov::element::Type_t::f32, inputShape, initial_state);
            auto readValue = std::make_shared<ngraph::opset9::ReadValue>(initial_state_node, var);
            ngraph::OutputVector toConcat{readValue, reshape1};
            const int axis = 1;
            auto concat = ngraph::builder::makeConcat(toConcat, axis);

            const auto concat_shape = concat->get_output_shape(0);
            const auto concat_shape_size = ngraph::shape_size(concat_shape);

            auto etlwise_data = CommonTestUtils::generate_float_numbers(concat_shape_size, -1.f, 1.f);
            auto etlwise_node = ngraph::builder::makeConstant(ov::element::Type_t::f32, concat_shape, etlwise_data);
            auto etlwise_result_node = std::make_shared<ngraph::opset9::Multiply>(concat, etlwise_node);

            ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(etlwise_result_node)};
            auto split_node = ngraph::builder::makeSplit(concat, ngPrc, 2, axis);

            auto assign_node = std::make_shared<ngraph::opset9::Assign>(split_node->output(1), var);
            ngraph::SinkVector sinks{assign_node};
            function = std::make_shared<ngraph::Function>(results, sinks, params, "CCCM");
            ov::pass::Serialize("a1.xml", "a1.bin").run_on_function(function);
        }
};

TEST_P(CCCMTest, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_FP32"}
    }
};

const std::vector<std::vector<size_t>> shapes {
    {16, 24},
};

INSTANTIATE_TEST_SUITE_P(smoke_conv_align_filter, CCCMTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs),
            ::testing::ValuesIn(shapes)),
        CCCMTest::getTestCaseName);

} // namespace LayerTestsDefinitions