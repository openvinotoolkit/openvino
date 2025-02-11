// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "common_test_utils/node_builders/convolution_backprop_data.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_op/convolution_backprop_data.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/convolution.hpp"

namespace {
using ov::test::InputShape;
using ov::test::convBackpropDataSpecificParams;

using DeconvInputData = std::tuple<InputShape,                           // data shape
                                   ov::test::utils::InputLayerType,      // 'output_shape' input type
                                   std::vector<std::vector<int32_t>>>;   // values for 'output_shape'

using DeconvLayerTestParamsSet = std::tuple<convBackpropDataSpecificParams,
                                            DeconvInputData,
                                            ov::element::Type,
                                            std::string,
                                            std::map<std::string, std::string>>;

class DeconvolutionLayerGPUTest : public testing::WithParamInterface<DeconvLayerTestParamsSet>,
                                  virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DeconvLayerTestParamsSet> obj) {
        convBackpropDataSpecificParams basicParamsSet;
        DeconvInputData inputData;
        ov::element::Type model_type;
        std::string targetDevice;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, inputData, model_type, targetDevice, additionalConfig) = obj.param;

        ov::op::PadType padType;
        std::vector<size_t> kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, outPadding) = basicParamsSet;

        InputShape inputShape;
        ov::test::utils::InputLayerType outShapeType;
        std::vector<std::vector<int32_t>> outShapeData;
        std::tie(inputShape, outShapeType, outShapeData) = inputData;

        std::ostringstream result;
        result << "IS=";
        result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : inputShape.second) {
            result << "(";
            result << ov::test::utils::vec2str(shape);
            result << ")_";
        }
        result << "PRC=" << model_type << "_";
        result << "K=" << ov::test::utils::vec2str(kernel) << "_";
        result << "S=" << ov::test::utils::vec2str(stride) << "_";
        result << "PB=" << ov::test::utils::vec2str(padBegin) << "_";
        result << "PE=" << ov::test::utils::vec2str(padEnd) << "_";
        result << "D=" << ov::test::utils::vec2str(dilation) << "_";
        result << "OP=" << ov::test::utils::vec2str(outPadding) << "_";
        result << "O=" << convOutChannels << "_";
        result << "AP=" << padType << "_";
        result << "OUT_SH=" << outShapeType << "_";
        result << "OUT_D=";
        for (const auto& data : outShapeData) {
            result << "(";
            result << ov::test::utils::vec2str(data);
            result << ")_";
        }
        result << "config=(";
        for (const auto& configEntry : additionalConfig) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        if (function->get_parameters().size() != 1) {
            // WA: output_shape depends on 3rd deconvolution input data
            // but the reference implementation doesn't implement shape inference
            // so we need to build a new ov function and replace the 3rd input parameter with a constant
            // to get valid output shapes
            functionRefs = createGraph({targetInputStaticShapes[0]}, ov::test::utils::InputLayerType::CONSTANT);
        }
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 1) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], outShapeData[inferRequestNum].data());
            } else {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = -20;
                in_data.range = 20;
                in_data.resolution = 256;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            }

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
        inferRequestNum++;
    }

    std::shared_ptr<ov::Model> createGraph(const std::vector<ov::PartialShape>& inShapes, ov::test::utils::InputLayerType outShapeType) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inShapes.front())};
        std::shared_ptr<ov::Node> outShapeNode;
        if (!outShapeData.empty()) {
            if (outShapeType == ov::test::utils::InputLayerType::PARAMETER) {
                OPENVINO_ASSERT(inputDynamicShapes.size() == 2);
                auto outShapeParam = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes.back());
                params.push_back(outShapeParam);
                outShapeNode = outShapeParam;
            } else {
                outShapeNode = ov::op::v0::Constant::create(ov::element::i32, {outShapeData[inferRequestNum].size()}, outShapeData[inferRequestNum]);
            }
        }

        for (size_t i = 0; i < params.size(); i++) {
            params[i]->set_friendly_name(std::string("param_") + std::to_string(i));
        }

        std::shared_ptr<ov::Node> deconv;
        if (!outShapeData.empty()) {
            OPENVINO_ASSERT(outShapeNode != nullptr);
            deconv = ov::test::utils::make_convolution_backprop_data(params[0], outShapeNode, model_type, kernel, stride, padBegin,
                                                                     padEnd, dilation, padType, convOutChannels);
        } else {
            deconv = ov::test::utils::make_convolution_backprop_data(params[0], model_type, kernel, stride, padBegin,
                                                                     padEnd, dilation, padType, convOutChannels, false, outPadding);
        }

        ov::ResultVector results;
        for (size_t i = 0; i < deconv->get_output_size(); i++)
             results.push_back(std::make_shared<ov::op::v0::Result>(deconv->output(i)));

        return std::make_shared<ov::Model>(results, params, "Deconv");
    }

protected:
    void SetUp() override {
        convBackpropDataSpecificParams basicParamsSet;
        DeconvInputData inputData;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, inputData, model_type, targetDevice, additionalConfig) = this->GetParam();

        InputShape inputShape;
        ov::test::utils::InputLayerType outShapeType;
        std::tie(inputShape, outShapeType, outShapeData) = inputData;

        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, outPadding) = basicParamsSet;

        std::vector<InputShape> paramsShapes;
        paramsShapes.push_back(inputShape);
        if (!outShapeData.empty() && outShapeType == ov::test::utils::InputLayerType::PARAMETER) {
            const auto outShapeDims = ov::Shape{outShapeData.front().size()};
            paramsShapes.push_back(InputShape{outShapeDims, std::vector<ov::Shape>(inputShape.second.size(), outShapeDims)});
        }

        init_input_shapes(paramsShapes);

        function = createGraph(inputDynamicShapes, outShapeType);
    }

private:
    ov::element::Type model_type;
    ov::op::PadType padType;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels;
    std::vector<std::vector<int32_t>> outShapeData;
    size_t inferRequestNum = 0;
};

TEST_P(DeconvolutionLayerGPUTest, Inference) {
    run();
}

std::map<std::string, std::string> emptyAdditionalConfig;

const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = { {} };

/* ============= Deconvolution params ============= */
const std::vector<size_t> numOutChannels = { 6 };

/* ============= Deconvolution params (2D) ============= */
const std::vector<std::vector<size_t>> kernels2d = { {3, 3}, {1, 1} };
const std::vector<std::vector<size_t>> strides2d = { {1, 1}, {2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins2d = { {0, 0} };
const std::vector<std::vector<ptrdiff_t>> padEnds2d = { {0, 0} };
const std::vector<std::vector<size_t>> dilations2d = { {1, 1} };

/* ============= Deconvolution (2D) ============= */
const auto convParams_ExplicitPadding_2D = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    ::testing::ValuesIn(strides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ov::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding)
);

const std::vector<DeconvInputData> dyn_2D_inputs_smoke = {
    DeconvInputData{
        InputShape{{-1, 12, -1, -1}, {{1, 12, 7, 7}, {2, 12, 5, 7}, {1, 12, 7, 7}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 12, -1, -1}, {{2, 12, 7, 7}, {2, 12, 5, 7}, {1, 12, 9, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 12, 7, 7}, {{1, 12, 7, 7}, {2, 12, 7, 7}, {1, 12, 7, 7}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{{1, 10}, 12, 7, 7}, {{1, 12, 7, 7}, {2, 12, 7, 7}, {3, 12, 7, 7}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_Dynamic_FP32, DeconvolutionLayerGPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_2D,
        ::testing::ValuesIn(dyn_2D_inputs_smoke),
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(emptyAdditionalConfig)),
    DeconvolutionLayerGPUTest::getTestCaseName);

const std::vector<DeconvInputData> dyn_2D_inputs_with_output_shape = {
    DeconvInputData{
        InputShape{{-1, 12, -1, -1}, {{1, 12, 7, 7}, {2, 12, 5, 7}, {1, 12, 7, 7}}},
        ov::test::utils::InputLayerType::PARAMETER,
        {{15, 15}, {9, 10}, {15, 15}}
    },
    DeconvInputData{
        InputShape{{-1, 12, 7, 7}, {{1, 12, 7, 7}, {2, 12, 7, 7}, {1, 12, 7, 7}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{15, 15}}
    },
    DeconvInputData{
        InputShape{{{1, 10}, 12, 7, 7}, {{1, 12, 7, 7}, {2, 12, 7, 7}, {3, 12, 7, 7}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{15, 15}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_Dynamic_OutputShape_FP32, DeconvolutionLayerGPUTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(std::vector<size_t>{3, 3}),
            ::testing::ValuesIn(strides2d),
            ::testing::ValuesIn(padBegins2d),
            ::testing::ValuesIn(padEnds2d),
            ::testing::ValuesIn(dilations2d),
            ::testing::ValuesIn(numOutChannels),
            ::testing::Values(ov::op::PadType::EXPLICIT),
            ::testing::ValuesIn(emptyOutputPadding)),
        ::testing::ValuesIn(dyn_2D_inputs_with_output_shape),
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(emptyAdditionalConfig)),
    DeconvolutionLayerGPUTest::getTestCaseName);


const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding1d = { {0} };

/* ============= Deconvolution params ============= */
const std::vector<size_t> numOutChannels1d = { 256 };

/* ============= Deconvolution params (1D) ============= */
const std::vector<std::vector<size_t>> kernels1d = { {16} };
const std::vector<std::vector<size_t>> strides1d = { {8} };
const std::vector<std::vector<ptrdiff_t>> padBegins1d = { {4} };
const std::vector<std::vector<ptrdiff_t>> padEnds1d = { {4} };
const std::vector<std::vector<size_t>> dilations1d = { {1} };

/* ============= Deconvolution (1D) ============= */
const auto convParams_ExplicitPadding_1D = ::testing::Combine(
    ::testing::ValuesIn(kernels1d),
    ::testing::ValuesIn(strides1d),
    ::testing::ValuesIn(padBegins1d),
    ::testing::ValuesIn(padEnds1d),
    ::testing::ValuesIn(dilations1d),
    ::testing::ValuesIn(numOutChannels1d),
    ::testing::Values(ov::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding1d)
);

const std::vector<DeconvInputData> dyn_1D_inputs_smoke = {
    DeconvInputData{
        InputShape{{1, 512, -1}, {{1, 512, 577}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {}
    },
};

const std::vector<ov::element::Type> netPrecisions1D = {
        ov::element::f32
};

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_1D_Dynamic_FP32, DeconvolutionLayerGPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_1D,
        ::testing::ValuesIn(dyn_1D_inputs_smoke),
        ::testing::ValuesIn(netPrecisions1D),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(emptyAdditionalConfig)),
    DeconvolutionLayerGPUTest::getTestCaseName);
} // namespace
