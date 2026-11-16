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

using ov::test::InputShape;

enum class ConvType { Convolution, GroupConvolution, ConvolutionBackpropData, GroupConvolutionBackpropData };

std::ostream& operator<<(std::ostream& os, ConvType type) {
    switch (type) {
    case ConvType::Convolution:
        return os << "Convolution";
    case ConvType::GroupConvolution:
        return os << "GroupConvolution";
    case ConvType::ConvolutionBackpropData:
        return os << "ConvolutionBackpropData";
    case ConvType::GroupConvolutionBackpropData:
        return os << "GroupConvolutionBackpropData";
    }
    return os;
}

// (Group)Convolution / (Group)ConvolutionBackpropData in a dynamic model whose weights are a model input instead of a
// constant, like models which build the filters from another input (e.g. style transfer). The weights can have a dynamic
// shape, so the number of channels and the kernel size are known only at inference time, and their values change on
// every inference.
using ConvolutionRuntimeWeightsDynamicParams = std::tuple<ConvType,
                                                          std::vector<InputShape>,  // data and weights shapes
                                                          ov::element::Type>;       // inference precision

class ConvolutionRuntimeWeightsDynamic : public testing::WithParamInterface<ConvolutionRuntimeWeightsDynamicParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvolutionRuntimeWeightsDynamicParams>& obj) {
        const auto& [type, shapes, inference_precision] = obj.param;
        std::ostringstream result;
        result << type << "_";
        for (const auto& shape : shapes) {
            result << "IS=" << ov::test::utils::partialShape2str({shape.first}) << "_TS=";
            for (const auto& static_shape : shape.second) {
                result << ov::test::utils::vec2str(static_shape) << "_";
            }
        }
        result << "inference_precision=" << inference_precision.get_type_name();
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [type, shapes, inference_precision] = GetParam();
        configuration[ov::hint::inference_precision.name()] = inference_precision.get_type_name();
        if (inference_precision == ov::element::f16) {
            abs_threshold = 0.02;
            rel_threshold = 0.02;
        } else {
            // Outputs are sums of up to 144 products, so allow for a different f32 accumulation order
            abs_threshold = 1e-4;
            rel_threshold = 1e-4;
        }

        init_input_shapes(shapes);

        const ov::Strides strides{1, 1};
        const ov::Strides dilations{1, 1};
        const ov::CoordinateDiff pads{0, 0};

        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[1]);
        std::shared_ptr<ov::Node> conv;
        switch (type) {
        case ConvType::Convolution:
            conv = std::make_shared<ov::op::v1::Convolution>(data, weights, strides, pads, pads, dilations);
            break;
        case ConvType::GroupConvolution:
            conv = std::make_shared<ov::op::v1::GroupConvolution>(data, weights, strides, pads, pads, dilations);
            break;
        case ConvType::ConvolutionBackpropData:
            conv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(data, weights, strides, pads, pads, dilations);
            break;
        case ConvType::GroupConvolutionBackpropData:
            conv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(data, weights, strides, pads, pads, dilations);
            break;
        }

        auto result = std::make_shared<ov::op::v0::Result>(conv);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data, weights}, "ConvolutionRuntimeWeightsDynamic");
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& model_inputs = function->inputs();
        for (size_t i = 0; i < model_inputs.size(); ++i) {
            // A new seed for every inference, so that repeated shapes get new weights values
            auto tensor = ov::test::utils::create_and_fill_tensor(model_inputs[i].get_element_type(),
                                                                  target_input_static_shapes[i],
                                                                  ov::test::utils::InputGenerateData(-1, 2, 1000, ++seed));
            inputs.insert({model_inputs[i].get_node_shared_ptr(), tensor});
        }
    }

private:
    int seed = 0;
};

TEST_P(ConvolutionRuntimeWeightsDynamic, Inference) {
    run();
}

// Each shape is inferred twice in a row to check that the new weights values are used.

// Weights are OIYX
const std::vector<std::vector<InputShape>> convolution_shapes = {
    // Dynamic output channels and kernel size
    {{{1, 8, -1, -1}, {{1, 8, 12, 12}, {1, 8, 12, 12}, {1, 8, 9, 11}, {1, 8, 9, 11}, {1, 8, 12, 12}}},
     {{-1, 8, -1, -1}, {{16, 8, 3, 3}, {16, 8, 3, 3}, {5, 8, 1, 2}, {5, 8, 1, 2}, {16, 8, 3, 3}}}},
    // Static weights shape, dynamic data
    {{{1, 8, -1, -1}, {{1, 8, 12, 12}, {1, 8, 12, 12}, {1, 8, 7, 9}, {1, 8, 7, 9}}},
     {{16, 8, 3, 3}, {{16, 8, 3, 3}, {16, 8, 3, 3}, {16, 8, 3, 3}, {16, 8, 3, 3}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionRuntimeWeightsDynamic,
                         ConvolutionRuntimeWeightsDynamic,
                         ::testing::Combine(::testing::Values(ConvType::Convolution),
                                            ::testing::ValuesIn(convolution_shapes),
                                            ::testing::Values(ov::element::f16, ov::element::f32)),
                         ConvolutionRuntimeWeightsDynamic::getTestCaseName);

// Weights are GOIYX with a static groups count
const std::vector<std::vector<InputShape>> group_convolution_shapes = {
    {{{1, 8, -1, -1}, {{1, 8, 12, 12}, {1, 8, 12, 12}, {1, 8, 9, 11}, {1, 8, 9, 11}}},
     {{2, -1, 4, -1, -1}, {{2, 3, 4, 3, 3}, {2, 3, 4, 3, 3}, {2, 5, 4, 2, 1}, {2, 5, 4, 2, 1}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionRuntimeWeightsDynamic,
                         ConvolutionRuntimeWeightsDynamic,
                         ::testing::Combine(::testing::Values(ConvType::GroupConvolution),
                                            ::testing::ValuesIn(group_convolution_shapes),
                                            ::testing::Values(ov::element::f16, ov::element::f32)),
                         ConvolutionRuntimeWeightsDynamic::getTestCaseName);

// Weights are IOYX, the input channels of the data depend on the weights like in a style swap
const std::vector<std::vector<InputShape>> convolution_backprop_data_shapes = {
    {{{1, -1, -1, -1}, {{1, 16, 10, 10}, {1, 16, 10, 10}, {1, 5, 7, 6}, {1, 5, 7, 6}, {1, 16, 10, 10}}},
     {{-1, 8, -1, -1}, {{16, 8, 3, 3}, {16, 8, 3, 3}, {5, 8, 2, 1}, {5, 8, 2, 1}, {16, 8, 3, 3}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropDataRuntimeWeightsDynamic,
                         ConvolutionRuntimeWeightsDynamic,
                         ::testing::Combine(::testing::Values(ConvType::ConvolutionBackpropData),
                                            ::testing::ValuesIn(convolution_backprop_data_shapes),
                                            ::testing::Values(ov::element::f16, ov::element::f32)),
                         ConvolutionRuntimeWeightsDynamic::getTestCaseName);

// Weights are GIOYX with a static groups count
const std::vector<std::vector<InputShape>> group_convolution_backprop_data_shapes = {
    {{{1, 8, -1, -1}, {{1, 8, 10, 10}, {1, 8, 10, 10}, {1, 8, 7, 6}, {1, 8, 7, 6}}},
     {{2, 4, -1, -1, -1}, {{2, 4, 3, 3, 3}, {2, 4, 3, 3, 3}, {2, 4, 5, 2, 1}, {2, 4, 5, 2, 1}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropDataRuntimeWeightsDynamic,
                         ConvolutionRuntimeWeightsDynamic,
                         ::testing::Combine(::testing::Values(ConvType::GroupConvolutionBackpropData),
                                            ::testing::ValuesIn(group_convolution_backprop_data_shapes),
                                            ::testing::Values(ov::element::f16, ov::element::f32)),
                         ConvolutionRuntimeWeightsDynamic::getTestCaseName);

}  // namespace
