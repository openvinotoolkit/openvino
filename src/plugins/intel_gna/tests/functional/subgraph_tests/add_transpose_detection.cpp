// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset9.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

using namespace ov::opset9;

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   std::vector<size_t>,                 // Input Shape
                   ngraph::helpers::InputLayerType,     // Type of Eltwise input
                   size_t>                              // Order of Eltwise input

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
        ngraph::helpers::InputLayerType input_eltwise_type;
        size_t input_eltwise_order;

        std::tie(precision, targetDevice, configuration, input_shape, input_eltwise_type, input_eltwise_order) =
            obj.param;

        std::ostringstream result;
        result << "netPRC=" << precision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";

        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_inputShape=" << ov::test::utils::vec2str(input_shape);
        result << "_input_eltwise_type=" << input_eltwise_type;
        result << "_input_eltwise_order=" << input_eltwise_order;

        return result.str();
    }

protected:
    const std::vector<size_t> filter_size{1, 1};
    const std::vector<size_t> strides{1, 1};
    const std::vector<ptrdiff_t> pads_begin{0, 0};
    const std::vector<ptrdiff_t> pads_end{0, 0};
    const std::vector<size_t> dilations{1, 1};
    const ov::op::PadType pad_type = ov::op::PadType::EXPLICIT;
    const size_t out_channels_num = 8;
    const size_t c_index_in_nchw = 1;

    void SetUp() override {
        InferenceEngine::Precision precision;
        std::vector<size_t> input_shape;
        ngraph::helpers::InputLayerType input_eltwise_type;
        size_t input_eltwise_order;
        std::tie(precision, targetDevice, configuration, input_shape, input_eltwise_type, input_eltwise_order) =
            this->GetParam();

        auto ng_precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(precision);

        auto input = std::make_shared<Parameter>(ng_precision, ov::Shape{input_shape});

        const auto weights_size = ov::shape_size(filter_size) * out_channels_num * input_shape[c_index_in_nchw];
        auto weights_values = ov::test::utils::generate_float_numbers(weights_size, -0.2f, 0.2f);

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
        std::shared_ptr<Add> add;
        if (input_eltwise_type == ngraph::helpers::InputLayerType::CONSTANT) {
            auto const_node = std::make_shared<Constant>(ng_precision, ov::Shape{input_shape});
            add = (input_eltwise_order == 0) ? std::make_shared<Add>(const_node, convolution)
                                             : std::make_shared<Add>(convolution, const_node);
        } else if (input_eltwise_type == ngraph::helpers::InputLayerType::PARAMETER) {
            add = (input_eltwise_order == 0) ? std::make_shared<Add>(input, convolution)
                                             : std::make_shared<Add>(convolution, input);
        }
        auto res = std::make_shared<Result>(add);
        function = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{input});
    }
};

TEST_P(InputConvAddTransposing, CompareWithRefImpl) {
    Run();
};

const std::vector<ngraph::helpers::InputLayerType> eltwise_input_types = {ngraph::helpers::InputLayerType::CONSTANT,
                                                                          ngraph::helpers::InputLayerType::PARAMETER};

const std::vector<size_t> eltwise_input_order = {0, 1};

const InferenceEngine::Precision net_precisions{InferenceEngine::Precision::FP32};

const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

const std::vector<std::vector<size_t>> input_shapes{{1, 8, 32, 16}};

INSTANTIATE_TEST_SUITE_P(smoke_add_transpose_detection,
                         InputConvAddTransposing,
                         ::testing::Combine(::testing::Values(net_precisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(eltwise_input_types),
                                            ::testing::ValuesIn(eltwise_input_order)),
                         InputConvAddTransposing::getTestCaseName);

}  // namespace LayerTestsDefinitions
