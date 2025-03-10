// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_op/convolution.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/convolution.hpp"

namespace {
using ov::test::InputShape;
using ov::test::convSpecificParams;

typedef std::tuple<
        convSpecificParams,
        ov::element::Type,     // Model type
        InputShape,            // Input shape
        std::string,           // Device name
        bool                   // activation fusing
> convLayerTestParamsSet;


class ConvolutionLayerGPUTestDynamic : public testing::WithParamInterface<convLayerTestParamsSet>,
                                       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerTestParamsSet>& obj) {
        convSpecificParams convParams;
        ov::element::Type model_type;
        InputShape inputShape;
        std::string targetDevice;
        bool activationFusing;
        std::tie(convParams, model_type, inputShape, targetDevice, activationFusing) = obj.param;

        ov::op::PadType padType;
        std::vector<size_t> kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        std::ostringstream result;
        result << "IS=";
        result  << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << ")_";
        result << "K" << ov::test::utils::vec2str(kernel) << "_";
        result << "S" << ov::test::utils::vec2str(stride) << "_";
        result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
        result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
        result << "D=" << ov::test::utils::vec2str(dilation) << "_";
        result << "O=" << convOutChannels << "_";
        result << "AP=" << padType << "_";
        result << "netPRC=" << model_type << "_";
        result << "trgDev=" << targetDevice << "_";
        result << "activationFusing=" << activationFusing;

        return result.str();
    }

protected:
    void SetUp() override {
        convSpecificParams convParams;
        InputShape inputShape;
        auto model_type = ov::element::dynamic;
        bool activationFusing;
        std::tie(convParams, model_type, inputShape, targetDevice, activationFusing) = this->GetParam();

        init_input_shapes({inputShape});

        ov::op::PadType padType;
        std::vector<size_t> kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));

        auto convolutionNode = ov::test::utils::make_convolution(inputParams.front(), model_type, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        if (activationFusing) {
                auto activationNode = ov::test::utils::make_activation(convolutionNode, model_type, ov::test::utils::ActivationTypes::Relu);

                ov::ResultVector results;
                for (size_t i = 0; i < activationNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(activationNode->output(i)));

                function = std::make_shared<ov::Model>(results, inputParams, "Convolution");
        } else {
                ov::ResultVector results;
                for (size_t i = 0; i < convolutionNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(convolutionNode->output(i)));

                function = std::make_shared<ov::Model>(results, inputParams, "Convolution");
        }
    }
};

TEST_P(ConvolutionLayerGPUTestDynamic, Inference) {
    run();
}

// ======== 1D convolutions
const std::vector<ov::test::InputShape> dynInputShapes1D = {
    {
        {1, 10, ov::Dimension::dynamic()},
        {{1, 10, 20}, {1, 10, 30}, {1, 10, 50}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic1DSymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(10),
                        ::testing::ValuesIn({ov::op::PadType::EXPLICIT, ov::op::PadType::VALID})),
                ::testing::Values(ov::element::f16),
                ::testing::ValuesIn(dynInputShapes1D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

const std::vector<std::vector<size_t>> kernels1D = { {3}, {1} };
const std::vector<std::vector<size_t>> strides1D = { {1} };
const std::vector<std::vector<ptrdiff_t>> padBegins1D = { {0}, {1} };
const std::vector<std::vector<ptrdiff_t>> padEnds1D = { {0}, {1} };
const std::vector<std::vector<size_t>> dilations1D = { {1} };
const std::vector<size_t> numOutChannels = { 64, 63 };
const std::vector<InputShape> inputShapes1D = {
        {{}, {{ 2, 64, 7 }}},
        {{}, {{ 1, 67, 7 }}},
        {
            //dynamic shape
            { -1, 64, {1, 200} },
            { //target static shapes
                { 2, 64, 7 },
                { 1, 64, 9 }
            }
        },
        {
            //dynamic shape
            { {1, 200}, 64, -1 },
            { //target static shapes
                { 2, 64, 7 },
                { 1, 64, 5 }
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_ExplicitPad1D, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(kernels1D),
                        ::testing::ValuesIn(strides1D),
                        ::testing::ValuesIn(padBegins1D),
                        ::testing::ValuesIn(padEnds1D),
                        ::testing::ValuesIn(dilations1D),
                        ::testing::ValuesIn(numOutChannels),
                        ::testing::Values(ov::op::PadType::EXPLICIT)),
                ::testing::Values(ov::element::f16),
                ::testing::ValuesIn(inputShapes1D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ======== 2D convolutions
const std::vector<ov::test::InputShape> dynInputShapes2D = {
    {
        {1, 10, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        {{1, 10, 20, 20}, {1, 10, 30, 30}, {1, 10, 40, 20}}
    },
};

// Specific range causes output static shapeS
const std::vector<ov::test::InputShape> dynInputShapes2D_static_output = {
    {
        {1, 128, {1, 2}, {1, 2}},
        {{1, 128, 1, 1}, {1, 128, 2, 2}}
    },
};
// ==== Symmetric pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic2DSymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3, 3}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(10),
                        ::testing::ValuesIn({ov::op::PadType::EXPLICIT, ov::op::PadType::VALID})),
                ::testing::Values(ov::element::f16),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Symmetric auto pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic2DSymAutoPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3, 3}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(10),
                        ::testing::ValuesIn({ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER})),
                ::testing::Values(ov::element::f16),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Asymmetric pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic2D_AsymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3, 3}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{2, 1}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(10),
                        ::testing::ValuesIn({ov::op::PadType::EXPLICIT, ov::op::PadType::VALID})),
                ::testing::Values(ov::element::f16),
                ::testing::ValuesIn(dynInputShapes2D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Static output
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic2D_static_output, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3, 3}),
                        ::testing::Values(std::vector<size_t>{2, 2}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 1}),
                        ::testing::Values(std::vector<size_t>{1, 1}),
                        ::testing::Values(256),
                        ::testing::Values(ov::op::PadType::EXPLICIT)),
                ::testing::Values(ov::element::f32),
                ::testing::ValuesIn(dynInputShapes2D_static_output),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(true)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ======== 3D convolutions
const std::vector<ov::test::InputShape> dynInputShapes3D = {
    {
        {1, 3, ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        {{1, 3, 10, 10, 10}, {1, 3, 20, 20, 10}, {1, 3, 15, 15, 10}}
    },
};

// ==== Symmetric pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic3DSymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3, 3, 3}),
                        ::testing::Values(std::vector<size_t>{1, 1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2, 1}),
                        ::testing::Values(std::vector<size_t>{1, 1, 1}),
                        ::testing::Values(3),
                        ::testing::ValuesIn({ov::op::PadType::EXPLICIT, ov::op::PadType::VALID})),
                ::testing::Values(ov::element::f16),
                ::testing::ValuesIn(dynInputShapes3D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Symmetric auto pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic3DSymAutoPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3, 3, 3}),
                        ::testing::Values(std::vector<size_t>{1, 1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0, 0, 0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0, 0, 0}),
                        ::testing::Values(std::vector<size_t>{1, 1, 1}),
                        ::testing::Values(3),
                        ::testing::ValuesIn({ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER})),
                ::testing::Values(ov::element::f16),
                ::testing::ValuesIn(dynInputShapes3D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

// ==== Asymmetric pad
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic3DAsymPad, ConvolutionLayerGPUTestDynamic,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{3, 3, 3}),
                        ::testing::Values(std::vector<size_t>{1, 1, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{1, 2, 1}),
                        ::testing::Values(std::vector<ptrdiff_t>{2, 1, 1}),
                        ::testing::Values(std::vector<size_t>{1, 1, 1}),
                        ::testing::Values(3),
                        ::testing::ValuesIn({ov::op::PadType::EXPLICIT, ov::op::PadType::VALID})),
                ::testing::Values(ov::element::f16),
                ::testing::ValuesIn(dynInputShapes3D),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamic::getTestCaseName);

typedef std::tuple<
        convSpecificParams,
        ov::element::Type,              // Model type
        std::vector<InputShape>,        // Input shapes
        std::string,                    // Device name
        bool                            // activation fusing
> convLayerFusingTestParamsSet;


class ConvolutionLayerGPUTestDynamicEltwiseFusing : public testing::WithParamInterface<convLayerFusingTestParamsSet>,
                                                    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerFusingTestParamsSet>& obj) {
        convSpecificParams convParams;
        ov::element::Type model_type;
        std::vector<InputShape> inputShapes;
        std::string targetDevice;
        bool activationFusing;
        std::tie(convParams, model_type, inputShapes, targetDevice, activationFusing) = obj.param;

        ov::op::PadType padType;
        std::vector<size_t> kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        std::ostringstream result;
        for (const auto& inputShape : inputShapes) {
                result << "IS=";
                result  << ov::test::utils::partialShape2str({inputShape.first}) << "_";
                result << "TS=(";
                for (const auto& shape : inputShape.second) {
                result << ov::test::utils::vec2str(shape) << "_";
                }
        }
        result << ")_";
        result << "K" << ov::test::utils::vec2str(kernel) << "_";
        result << "S" << ov::test::utils::vec2str(stride) << "_";
        result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
        result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
        result << "D=" << ov::test::utils::vec2str(dilation) << "_";
        result << "O=" << convOutChannels << "_";
        result << "AP=" << padType << "_";
        result << "netPRC=" << model_type << "_";
        result << "trgDev=" << targetDevice << "_";
        result << "activationFusing=" << activationFusing;

        return result.str();
    }

protected:
    void SetUp() override {
        convSpecificParams convParams;
        std::vector<InputShape> inputShapes;
        auto model_type = ov::element::dynamic;
        bool activationFusing;
        std::tie(convParams, model_type, inputShapes, targetDevice, activationFusing) = this->GetParam();

        init_input_shapes({inputShapes});

        ov::op::PadType padType;
        std::vector<size_t> kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));

        auto convolutionNode = ov::test::utils::make_convolution(inputParams.front(), model_type, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);
        if (activationFusing) {
                auto activationNode = ov::test::utils::make_activation(convolutionNode, model_type, ov::test::utils::ActivationTypes::Relu);
                auto eltwiseNode = ov::test::utils::make_eltwise(inputParams.back(), activationNode, ov::test::utils::EltwiseTypes::ADD);

                ov::ResultVector results;
                for (size_t i = 0; i < eltwiseNode->get_output_size(); i++)
                        results.push_back(std::make_shared<ov::op::v0::Result>(eltwiseNode->output(i)));

                function = std::make_shared<ov::Model>(results, inputParams, "Convolution");
        } else {
                auto eltwiseNode = ov::test::utils::make_eltwise(inputParams.back(), convolutionNode, ov::test::utils::EltwiseTypes::ADD);

                ov::ResultVector results;
                for (size_t i = 0; i < eltwiseNode->get_output_size(); i++)
                        results.push_back(std::make_shared<ov::op::v0::Result>(eltwiseNode->output(i)));

                function = std::make_shared<ov::Model>(results, inputParams, "Convolution");
        }
    }
};

TEST_P(ConvolutionLayerGPUTestDynamicEltwiseFusing, Inference) {
    run();
}
const std::vector<std::vector<ov::test::InputShape>> dynInputShapes1D_test = {
        {
        {
                {1, 192, ov::Dimension::dynamic()},
                {{1, 192, 191}}
        },
        {
                {1, 192, ov::Dimension::dynamic()},
                {{1, 192, 1}}
        }
        },
        {
        {
                {ov::Dimension::dynamic(), 192, ov::Dimension::dynamic()},
                {{1, 192, 257}}
        },
        {
                {1, 1, ov::Dimension::dynamic()},
                {{1, 1, 257}}
        }
        },
        {
        {
                {ov::Dimension::dynamic(), 192, ov::Dimension::dynamic()},
                {{1, 192, 257}}
        },
        {
                {1, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                {{1, 1, 1}}
        }
        },
        {
        {
                {ov::Dimension::dynamic(), 192, ov::Dimension::dynamic()},
                {{1, 192, 1}}
        },
        {
                {1, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                {{1, 1, 1}}
        }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic1D_test_0, ConvolutionLayerGPUTestDynamicEltwiseFusing,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(192),
                        ::testing::Values(ov::op::PadType::EXPLICIT)),
                ::testing::Values(ov::element::f32),
                ::testing::ValuesIn(dynInputShapes1D_test),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamicEltwiseFusing::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynInputShapes1D_test1 = {
        {
        {
                {1, 512, ov::Dimension::dynamic()},
                {{1, 512, 191}}
        },
        {
                {1, 512, ov::Dimension::dynamic()},
                {{1, 512, 1}}
        }
        },
        {
        {
                {ov::Dimension::dynamic(), 512, ov::Dimension::dynamic()},
                {{1, 512, 191}}
        },
        {
                {1, 1, ov::Dimension::dynamic()},
                {{1, 1, 191}}
        }
        },
        {
        {
                {ov::Dimension::dynamic(), 512, ov::Dimension::dynamic()},
                {{1, 512, 191}}
        },
        {
                {1, 1, ov::Dimension::dynamic()},
                {{1, 1, 1}}
        }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic1D_test_1, ConvolutionLayerGPUTestDynamicEltwiseFusing,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(512),
                        ::testing::Values(ov::op::PadType::EXPLICIT)),
                ::testing::Values(ov::element::f32),
                ::testing::ValuesIn(dynInputShapes1D_test1),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamicEltwiseFusing::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynInputShapes1D_test2 = {
        {
        {
                {1, 2048, ov::Dimension::dynamic()},
                {{1, 2048, 191}}
        },
        {
                {1, 2048, ov::Dimension::dynamic()},
                {{1, 2048, 1}}
        }
        },
        {
        {
                {1, 2048, ov::Dimension::dynamic()},
                {{1, 2048, 191}}
        },
        {
                {ov::Dimension::dynamic(), 1, ov::Dimension::dynamic()},
                {{1, 1, 191}}
        }
        },
        {
        {
                {1, 2048, ov::Dimension::dynamic()},
                {{1, 2048, 191}}
        },
        {
                {ov::Dimension::dynamic(), 1, ov::Dimension::dynamic()},
                {{1, 1, 1}}
        }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_dynamic1D_test_2, ConvolutionLayerGPUTestDynamicEltwiseFusing,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<ptrdiff_t>{0}),
                        ::testing::Values(std::vector<size_t>{1}),
                        ::testing::Values(2048),
                        ::testing::Values(ov::op::PadType::EXPLICIT)),
                ::testing::Values(ov::element::f32),
                ::testing::ValuesIn(dynInputShapes1D_test2),
                ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                ::testing::Values(false)),
                ConvolutionLayerGPUTestDynamicEltwiseFusing::getTestCaseName);
}  // namespace
