// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "openvino/opsets/opset9.hpp"

// using namespace ov::intel_gna::ngraph_util;
using namespace ov::opset9;

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   std::vector<size_t>>                 // Input Shape
    InputConvAddParams;

namespace LayerTestsDefinitions {

class InputConvAddTransposing : public testing::WithParamInterface<InputConvAddParams>,
                                public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InputConvAddParams> obj) {
        InferenceEngine::Precision precision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> input_shape;

        std::tie(precision, targetDevice, configuration, input_shape) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << precision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";

        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_inputShape=" << CommonTestUtils::vec2str(input_shape);

        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision precision;
        std::vector<size_t> input_shape;
        std::tie(precision, targetDevice, configuration, input_shape) = this->GetParam();

        auto ng_precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(precision);
        auto input = std::make_shared<Parameter>(ng_precision, ov::Shape{input_shape});

        const std::vector<size_t> filter_size{1, 1};
        const std::vector<size_t> strides{1, 1};
        const std::vector<ptrdiff_t> pads_begin{0, 0};
        const std::vector<ptrdiff_t> pads_end{0, 0};
        const std::vector<size_t> dilations{1, 1};
        const auto pad_type = ov::op::PadType::EXPLICIT;
        const size_t out_channels_num = 8;
        constexpr auto c_index_in_nchw = 1;
        const auto weights_size = ov::shape_size(filter_size) * out_channels_num * input_shape[c_index_in_nchw];
        auto weights_values = CommonTestUtils::generate_float_numbers(weights_size, -0.2f, 0.2f);

        auto convolution = ngraph::builder::makeConvolution(input,
                                                            ng_precision,
                                                            filter_size,
                                                            strides,
                                                            pads_begin,
                                                            pads_end,
                                                            dilations,
                                                            pad_type,
                                                            out_channels_num,
                                                            false,
                                                            weights_values);
        auto add = std::make_shared<Add>(input, convolution);
        auto res = std::make_shared<Result>(add);
        function = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{input});
    }
};

TEST_P(InputConvAddTransposing, CompareWithRefImpl) {
    Run();
};

const InferenceEngine::Precision net_precisions{InferenceEngine::Precision::FP32};

const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}
};

const std::vector<std::vector<size_t>> input_shapes {
    {1, 8, 32, 16}
};

INSTANTIATE_TEST_SUITE_P(smoke_convolution_transpose_output,
                         InputConvAddTransposing,
                         ::testing::Combine(::testing::Values(net_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(input_shapes)),
                         InputConvAddTransposing::getTestCaseName);

}  // namespace LayerTestsDefinitions
