// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

// Stride-1 (Group)ConvolutionBackpropData whose weights are a model input rather than a constant. The GPU plugin
// executes it as a convolution over 180-degree rotated weights, so the rotation has to be applied at runtime.
using ConvolutionBackpropDataRuntimeWeightsParams = std::tuple<size_t,              // groups
                                                               ov::element::Type>;  // inference precision

class ConvolutionBackpropDataRuntimeWeights : public ov::test::SubgraphBaseStaticTest,
                                              public ::testing::WithParamInterface<ConvolutionBackpropDataRuntimeWeightsParams> {
public:
    static std::string getTestCaseName(const ::testing::TestParamInfo<ConvolutionBackpropDataRuntimeWeightsParams>& obj) {
        const auto& [groups, inference_precision] = obj.param;
        return "groups=" + std::to_string(groups) + "_inference_precision=" + inference_precision.get_type_name();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [groups, inference_precision] = GetParam();
        configuration[ov::hint::inference_precision.name()] = inference_precision.get_type_name();
        if (inference_precision == ov::element::f16) {
            abs_threshold = 0.02;
            rel_threshold = 0.02;
        }

        const size_t in_channels = 8;
        const size_t out_channels = 16;
        const size_t kernel = 3;
        const ov::Strides strides{1, 1};
        const ov::Strides dilations{1, 1};
        const ov::CoordinateDiff pads{1, 1};

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, in_channels, 7, 7});
        std::shared_ptr<ov::Node> deconv;
        std::shared_ptr<ov::op::v0::Parameter> weights;
        if (groups == 1) {
            weights = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{in_channels, out_channels, kernel, kernel});
            deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(input, weights, strides, pads, pads, dilations);
        } else {
            weights = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                              ov::Shape{groups, in_channels / groups, out_channels / groups, kernel, kernel});
            deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(input, weights, strides, pads, pads, dilations);
        }

        auto result = std::make_shared<ov::op::v0::Result>(deconv);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input, weights}, "ConvolutionBackpropDataRuntimeWeights");
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& model_inputs = function->inputs();
        for (size_t i = 0; i < model_inputs.size(); ++i) {
            auto tensor = ov::test::utils::create_and_fill_tensor(model_inputs[i].get_element_type(),
                                                                  target_input_static_shapes[i],
                                                                  ov::test::utils::InputGenerateData(-1, 2, 1000, static_cast<int>(i + 1)));
            inputs.insert({model_inputs[i].get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(ConvolutionBackpropDataRuntimeWeights, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropDataRuntimeWeights,
                         ConvolutionBackpropDataRuntimeWeights,
                         ::testing::Combine(::testing::Values(size_t{1}, size_t{2}), ::testing::Values(ov::element::f16, ov::element::f32)),
                         ConvolutionBackpropDataRuntimeWeights::getTestCaseName);

}  // namespace
