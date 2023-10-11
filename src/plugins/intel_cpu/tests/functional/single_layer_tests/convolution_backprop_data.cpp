// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <shared_test_classes/single_layer/convolution_backprop_data.hpp>

#include "cpu_shape.h"
#include "ov_models/builders.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/convolution_params.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using DeconvSpecParams = LayerTestsDefinitions::convBackpropDataSpecificParams;

using DeconvInputData = std::tuple<InputShape,                           // data shape
                                   ngraph::helpers::InputLayerType,      // 'output_shape' input type
                                   std::vector<std::vector<int32_t>>>;   // values for 'output_shape'

using DeconvLayerCPUTestParamsSet = std::tuple<DeconvSpecParams,
                                               DeconvInputData,
                                               ElementType,
                                               fusingSpecificParams,
                                               CPUSpecificParams,
                                               std::map<std::string, std::string>>;

class DeconvolutionLayerCPUTest : public testing::WithParamInterface<DeconvLayerCPUTestParamsSet>,
                                  virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DeconvLayerCPUTestParamsSet> obj) {
        DeconvSpecParams basicParamsSet;
        DeconvInputData inputData;
        ElementType prec;
        fusingSpecificParams fusingParams;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, inputData, prec, fusingParams, cpuParams, additionalConfig) = obj.param;

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, outPadding) = basicParamsSet;

        InputShape inputShape;
        ngraph::helpers::InputLayerType outShapeType;
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
        result << "PRC=" << prec << "_";
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

        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second;
            }
        }

        return result.str();
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 1) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], outShapeData[inferRequestNum].data());
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 2560, 0, 256);
            }

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
        inferRequestNum++;
    }

    void init_ref_function(std::shared_ptr<ov::Model> &funcRef, const std::vector<ov::Shape>& targetInputStaticShapes) override {
        if (function->get_parameters().size() == 1) {
            ngraph::helpers::resize_function(funcRef, targetInputStaticShapes);
        } else {
            // WA: output_shape depends on 3rd deconvolution input data
            // but the reference implementation doesn't implement shape inference
            // so we need to build a new ngraph function and replace the 3rd input parameter with a constant
            // to get valid output shapes
            funcRef = createGraph({targetInputStaticShapes[0]}, ngraph::helpers::InputLayerType::CONSTANT);
        }
    }

    void validate() override {
        auto actualOutputs = get_plugin_outputs();
        if (function->get_parameters().size() == 2) {
            auto pos = std::find_if(inputs.begin(), inputs.end(),
                [](const std::pair<std::shared_ptr<ov::Node>, ov::Tensor> &params) {
                    return params.first->get_friendly_name() == "param_1";
                });
            IE_ASSERT(pos != inputs.end());
            inputs.erase(pos);
        }
        auto expectedOutputs = calculate_refs();
        if (expectedOutputs.empty()) {
                return;
        }
        ASSERT_EQ(actualOutputs.size(), expectedOutputs.size())
                << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

        compare(expectedOutputs, actualOutputs);
    }

    void configure_model() override {
        ov::preprocess::PrePostProcessor p(function);
        {
            auto& params = function->get_parameters();
            for (size_t i = 0; i < params.size(); i++) {
                if (i > 0) {
                    continue;
                }
                if (inType != ov::element::Type_t::undefined) {
                    p.input(i).tensor().set_element_type(inType);
                }
            }
        }
        {
            auto results = function->get_results();
            for (size_t i = 0; i < results.size(); i++) {
                if (outType != ov::element::Type_t::undefined) {
                    p.output(i).tensor().set_element_type(outType);
                }
            }
        }
        function = p.build();
    }

    std::shared_ptr<ov::Model> createGraph(const std::vector<ov::PartialShape>& inShapes, ngraph::helpers::InputLayerType outShapeType) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(prec, inShapes.front())};
        std::shared_ptr<ov::Node> outShapeNode;
        if (!outShapeData.empty()) {
            if (outShapeType == ngraph::helpers::InputLayerType::PARAMETER) {
                IE_ASSERT(inputDynamicShapes.size() == 2);
                auto outShapeParam = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, inputDynamicShapes.back());
                params.push_back(outShapeParam);
                outShapeNode = outShapeParam;
            } else {
                outShapeNode = ngraph::opset8::Constant::create(ngraph::element::i32, {outShapeData[inferRequestNum].size()}, outShapeData[inferRequestNum]);
            }
        }

        for (size_t i = 0; i < params.size(); i++) {
            params[i]->set_friendly_name(std::string("param_") + std::to_string(i));
        }

        std::shared_ptr<ov::Node> deconv;
        if (!outShapeData.empty()) {
            IE_ASSERT(outShapeNode != nullptr);
            deconv = ngraph::builder::makeConvolutionBackpropData(params[0], outShapeNode, prec, kernel, stride, padBegin,
                                                                  padEnd, dilation, padType, convOutChannels);
        } else {
            deconv = ngraph::builder::makeConvolutionBackpropData(params[0], prec, kernel, stride, padBegin,
                                                                  padEnd, dilation, padType, convOutChannels, false, outPadding);
        }

        return makeNgraphFunction(prec, params, deconv, "DeconvCPU");
    }

protected:
    InferenceEngine::SizeVector kernel, stride;

    void SetUp() override {
        rel_threshold = 1e-4f;

        targetDevice = ov::test::utils::DEVICE_CPU;

        DeconvSpecParams basicParamsSet;
        DeconvInputData inputData;
        fusingSpecificParams fusingParams;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, inputData, prec, fusingParams, cpuParams, additionalConfig) = this->GetParam();

        InputShape inputShape;
        ngraph::helpers::InputLayerType outShapeType;
        std::tie(inputShape, outShapeType, outShapeData) = inputData;

        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, outPadding) = basicParamsSet;

        if (additionalConfig[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] == InferenceEngine::PluginConfigParams::YES) {
            inType = outType = prec = ElementType::bf16;
            rel_threshold = 1e-2f;
        } else {
            inType = outType = prec;
        }

        selectedType = makeSelectedTypeStr(selectedType, prec);

        std::vector<InputShape> paramsShapes;
        paramsShapes.push_back(inputShape);
        if (!outShapeData.empty() && outShapeType == ngraph::helpers::InputLayerType::PARAMETER) {
            const auto outShapeDims = ov::Shape{outShapeData.front().size()};
            paramsShapes.push_back(InputShape{outShapeDims, std::vector<ov::Shape>(inputShape.second.size(), outShapeDims)});
        }

        init_input_shapes(paramsShapes);

        function = createGraph(inputDynamicShapes, outShapeType);
    }

private:
    ElementType prec;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels;
    std::vector<std::vector<int32_t>> outShapeData;
    size_t inferRequestNum = 0;
};

TEST_P(DeconvolutionLayerCPUTest, CompareWithRefs) {
    if (!fusedOps.empty()) {
        bool isSupportedParams = stride[stride.size() - 1] <= kernel[kernel.size() - 1];
        if (stride.size() > 1)
            isSupportedParams &= stride[stride.size() - 2] <= kernel[kernel.size() - 2];
        if (stride.size() > 2)
            isSupportedParams &= stride[stride.size() - 3] <= kernel[kernel.size() - 3];
        if (!isSupportedParams) {
            GTEST_SKIP() << "Fusing with strides more than kernel size was disabled, because oneDNN deconvolution doesn't support it" << std::endl;
        }
    }

    run();
    CheckPluginRelatedResults(compiledModel, "Deconvolution");
}

namespace {

/* COMMON PARAMS */
const std::vector<fusingSpecificParams> fusingParamsSet{
        emptyFusingSpec,
        fusingScaleShift
};

const std::map<std::string, std::string> cpuEmptyPluginConfig;
const std::map<std::string, std::string>cpuBF16PluginConfig = { { InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16,
                                                                  InferenceEngine::PluginConfigParams::YES } };
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = { {} };

/* ============= Deconvolution params (planar layout) ============= */
const InferenceEngine::SizeVector numOutChannels_Planar = { 6 };

/* ============= Deconvolution params (blocked layout) ============= */
const InferenceEngine::SizeVector numOutChannels_Blocked = { 64 };

/* ============= Deconvolution params (2D) ============= */
const std::vector<InferenceEngine::SizeVector> kernels2d = { {3, 3}, {1, 1} };
const std::vector<InferenceEngine::SizeVector> strides2d = { {1, 1}, {2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins2d = { {0, 0} };
const std::vector<std::vector<ptrdiff_t>> padEnds2d = { {0, 0} };
const std::vector<InferenceEngine::SizeVector> dilations2d = { {1, 1} };


const std::vector<InferenceEngine::SizeVector> deconvAmxKernels2d = { {3, 3}, {2, 2}};
const std::vector<InferenceEngine::SizeVector> deconvAmxStrides2d = { {2, 2}};

/* ============= Deconvolution params (3D) ============= */
const std::vector<InferenceEngine::SizeVector> kernels3d = { {3, 3, 3}, {1, 1, 1} };
const std::vector<InferenceEngine::SizeVector> strides3d = { {1, 1, 1}, {2, 2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins3d = { {0, 0, 0} };
const std::vector<std::vector<ptrdiff_t>> padEnds3d = { {0, 0, 0} };
const std::vector<InferenceEngine::SizeVector> dilations3d = { {1, 1, 1} };

const std::vector<InferenceEngine::SizeVector> deconvAmxKernels3d = { {3, 3, 3}, {2, 2, 2} };
const std::vector<InferenceEngine::SizeVector> deconvAmxStrides3d = {  {2, 2, 2} };

/* ============= */

/* INSTANCES */
/* ============= Deconvolution (Planar 2D) ============= */
const auto convParams_ExplicitPadding_Planar_2D = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    ::testing::ValuesIn(strides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels_Planar),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding)
);

const std::vector<DeconvInputData> Planar_2D_inputs_smoke = {
    DeconvInputData{
        InputShape{{}, {{ 2, 12, 7, 7 }}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 12, -1, -1}, {{ 1, 12, 7, 7}, { 2, 12, 5, 7}, { 1, 12, 7, 7}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{15, 15}, {9, 10}, {15, 15}}
    }
};

const std::vector<DeconvInputData> Planar_2D_inputs_nightly = {
    DeconvInputData{
        InputShape{{-1, 12, -1, -1}, {{ 2, 12, 7, 7}, { 2, 12, 5, 7}, { 1, 12, 9, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 12, 7, 7}, {{ 1, 12, 7, 7}, { 2, 12, 7, 7}, { 1, 12, 7, 7}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{15, 15}}
    },
    DeconvInputData{
        InputShape{{{1, 10}, 12, 7, 7}, {{ 1, 12, 7, 7}, { 2, 12, 7, 7}, { 3, 12, 7, 7}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{15, 15}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_Planar_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Planar_2D,
        ::testing::ValuesIn(Planar_2D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_Planar_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Planar_2D,
        ::testing::ValuesIn(Planar_2D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D})),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_2D_Planar_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Planar_2D,
        ::testing::ValuesIn(Planar_2D_inputs_nightly),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_2D_Planar_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Planar_2D,
        ::testing::ValuesIn(Planar_2D_inputs_nightly),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D})),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution (Planar 3D) ============= */
const std::vector<DeconvInputData> Planar_3D_inputs_smoke = {
    DeconvInputData{
        InputShape{{}, {{ 2, 12, 7, 7, 7 }}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 12, -1, -1, -1}, {{ 2, 12, 7, 7, 7}, { 2, 12, 5, 7, 7}, { 1, 12, 9, 4, 9}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{15, 15, 15}, {9, 10, 10}, {9, 9, 9}}
    }
};

const std::vector<DeconvInputData> Planar_3D_inputs_nightly = {
    DeconvInputData{
        // -1 will result deconv use 64 to infer output shape, for 3d output shape is too big for gemm bwd kernel
        //  to buffer the intermedia results
        InputShape{{-1, 12, {5, 9}, {4, 7}, {7, 9}}, {{ 2, 12, 7, 7, 7}, { 2, 12, 5, 7, 7}, { 1, 12, 9, 4, 9}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 12, -1, -1, -1}, {{ 2, 12, 7, 7, 7}, { 2, 12, 5, 7, 7}, { 1, 12, 9, 4, 9}, { 2, 12, 7, 7, 7}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{10, 16, 16}}
    },
    DeconvInputData{
        InputShape{{{1, 10}, 12, 7, 7, 7}, {{ 2, 12, 7, 7, 7}, { 1, 12, 7, 7, 7}, { 3, 12, 7, 7, 7}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{15, 15, 15}}
    }
};

const auto convParams_ExplicitPadding_Planar_3D = ::testing::Combine(
    ::testing::ValuesIn(kernels3d),
    ::testing::ValuesIn(strides3d),
    ::testing::ValuesIn(padBegins3d),
    ::testing::ValuesIn(padEnds3d),
    ::testing::ValuesIn(dilations3d),
    ::testing::ValuesIn(numOutChannels_Planar),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_Planar_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Planar_3D,
        ::testing::ValuesIn(Planar_3D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_Planar_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Planar_3D,
        ::testing::ValuesIn(Planar_3D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D})),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_3D_Planar_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Planar_3D,
        ::testing::ValuesIn(Planar_3D_inputs_nightly),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_3D_Planar_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Planar_3D,
        ::testing::ValuesIn(Planar_3D_inputs_nightly),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D})),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution (Blocked 2D) ============= */
const std::vector<DeconvInputData> Blocked_2D_inputs_smoke = {
    DeconvInputData{
        InputShape{{}, {{ 2, 67, 7, 7 }}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 67, -1, -1}, {{ 2, 67, 7, 7}, { 2, 67, 5, 7}, { 1, 67, 9, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{15, 15}, {9, 10}, {9, 9}}
    }
};


const auto convParams_ExplicitPadding_Blocked_2D_nightly = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    // Use 7x7 with stride 1 is too small to generate 15x15 output. It needs a big negative pad which will result
    //  avx512 kernel not to be selected.
    ::testing::ValuesIn({strides2d[1]}),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels_Blocked),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding)
);

const std::vector<DeconvInputData> Blocked_2D_inputs_nightly = {
    DeconvInputData{
        InputShape{{-1, 67, -1, -1}, {{ 2, 67, 7, 7}, { 2, 67, 5, 7}, { 1, 67, 9, 4}, { 2, 67, 7, 7}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 67, -1, -1}, {{ 2, 67, 7, 7}, { 2, 67, 5, 7}, { 1, 67, 9, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{15, 15}}
    },
    DeconvInputData{
        InputShape{{ {1, 10}, 67, 7, 7}, {{ 2, 67, 7, 7}, { 3, 67, 7, 7}, { 1, 67, 7, 7}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{15, 15}}
    }
};

const auto convParams_ExplicitPadding_Blocked_2D = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    ::testing::ValuesIn(strides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels_Blocked),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding)
);

const auto convParams_ExplicitPadding_AMX_2D = ::testing::Combine(
    ::testing::ValuesIn(deconvAmxKernels2d),
    ::testing::ValuesIn(deconvAmxStrides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels_Blocked),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_Blocked_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Blocked_2D,
        ::testing::ValuesIn(Blocked_2D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D, conv_avx2_2D})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_Blocked_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Blocked_2D,
        ::testing::ValuesIn(Blocked_2D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D})),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_NSPC_BF16_AMX_NO_FUSING, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_AMX_2D,
        ::testing::ValuesIn(Blocked_2D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn({emptyFusingSpec}),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_nspc_amx})),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_NSPC_INT8_AMX, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_AMX_2D,
        ::testing::ValuesIn(Blocked_2D_inputs_smoke),
        ::testing::Values(ElementType::i8),
        ::testing::ValuesIn({emptyFusingSpec, fusingClampRoundAddRelu}),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_nspc_amx})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_2D_Blocked_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Blocked_2D_nightly,
        ::testing::ValuesIn(Blocked_2D_inputs_nightly),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D, conv_avx2_2D})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_2D_Blocked_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Blocked_2D_nightly,
        ::testing::ValuesIn(Blocked_2D_inputs_nightly),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D})),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution (Blocked 3D) ============= */
const std::vector<DeconvInputData> Blocked_3D_inputs_smoke = {
    DeconvInputData{
        InputShape{{}, {{ 2, 35, 7, 7, 7 }}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 35, -1, -1, -1}, {{ 1, 35, 5, 5, 5}, { 2, 35, 5, 7, 5}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{7, 7, 7}, {7, 9, 7}}
    }
};

const auto convParams_ExplicitPadding_Blocked_3D_nightly = ::testing::Combine(
    ::testing::ValuesIn(kernels3d),
    ::testing::ValuesIn({strides3d[0]}),
    ::testing::ValuesIn(padBegins3d),
    ::testing::ValuesIn(padEnds3d),
    ::testing::ValuesIn(dilations3d),
    ::testing::Values(32),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding)
);

const std::vector<DeconvInputData> Blocked_3D_inputs_nightly = {
    DeconvInputData{
        InputShape{{-1, 35, -1, -1, -1}, {{ 1, 35, 5, 5, 5}, { 2, 35, 5, 7, 5}, { 1, 35, 5, 5, 5}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 35, -1, -1, -1}, {{ 1, 35, 5, 5, 5}, { 2, 35, 5, 7, 5}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{7, 7, 7}}
    },
    DeconvInputData{
        InputShape{{{1, 10}, 35, 5, 5, 5}, {{ 1, 35, 5, 5, 5}, { 2, 35, 5, 5, 5}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{7, 7, 7}}
    }
};

const auto convParams_ExplicitPadding_Blocked_3D = ::testing::Combine(
    ::testing::ValuesIn(kernels3d),
    ::testing::ValuesIn(strides3d),
    ::testing::ValuesIn(padBegins3d),
    ::testing::ValuesIn(padEnds3d),
    ::testing::ValuesIn(dilations3d),
    ::testing::Values(32),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding)
);

const auto convParams_ExplicitPadding_AMX_3D = ::testing::Combine(
    ::testing::ValuesIn(deconvAmxKernels3d),
    ::testing::ValuesIn(deconvAmxStrides3d),
    ::testing::ValuesIn(padBegins3d),
    ::testing::ValuesIn(padEnds3d),
    ::testing::ValuesIn(dilations3d),
    ::testing::Values(32),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_Blocked_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Blocked_3D,
        ::testing::ValuesIn(Blocked_3D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_Blocked_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Blocked_3D,
        ::testing::ValuesIn(Blocked_3D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_NSPC_BF16_AMX_NO_FUSING, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_AMX_3D,
        ::testing::ValuesIn(Blocked_3D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn({emptyFusingSpec}),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D_nspc_amx})),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_NSPC_INT8_AMX, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_AMX_3D,
        ::testing::ValuesIn(Blocked_3D_inputs_smoke),
        ::testing::Values(ElementType::i8),
        ::testing::ValuesIn({emptyFusingSpec, fusingClampRoundAddRelu}),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D_nspc_amx})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_3D_Blocked_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Blocked_3D_nightly,
        ::testing::ValuesIn(Blocked_3D_inputs_nightly),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_3D_Blocked_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_Blocked_3D_nightly,
        ::testing::ValuesIn(Blocked_3D_inputs_nightly),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (2D) ============= */
const auto convParams_ExplicitPadding_1x1_2D = ::testing::Combine(
        ::testing::Values(InferenceEngine::SizeVector({1, 1})),
        ::testing::Values(InferenceEngine::SizeVector({1, 1})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(InferenceEngine::SizeVector({1, 1})),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_1x1_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_1x1_2D,
        ::testing::ValuesIn(Blocked_2D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_1x1, conv_avx2_2D_1x1})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_1x1_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        convParams_ExplicitPadding_1x1_2D,
        ::testing::ValuesIn(Blocked_2D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_1x1, conv_avx2_2D_1x1})),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Reorder + Deconvolution ============= */
INSTANTIATE_TEST_SUITE_P(smoke_reorder_Deconv_2D, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(kernels2d),
                           ::testing::Values(InferenceEngine::SizeVector{1, 1}),
                           ::testing::ValuesIn(padBegins2d),
                           ::testing::ValuesIn(padEnds2d),
                           ::testing::ValuesIn(dilations2d),
                           ::testing::ValuesIn(numOutChannels_Blocked),
                           ::testing::Values(ngraph::op::PadType::EXPLICIT),
                           ::testing::ValuesIn(emptyOutputPadding)),
        ::testing::Values(DeconvInputData{InputShape{{-1, 67, -1, -1}, {{ 1, 67, 7, 7}, { 1, 67, 9, 4}, { 1, 67, 5, 7}, { 1, 67, 7, 7}, { 1, 67, 9, 4}}},
                                                     ngraph::helpers::InputLayerType::PARAMETER,
                                                     {{15, 15}, {9, 9}, {9, 10}, {15, 15}, {9, 9}}}),
        ::testing::Values(ElementType::f32),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution auto padding tests ============= */
const std::vector<DeconvInputData> inputs_2D_AutoPadding = {
    DeconvInputData{
        InputShape{{}, {{ 2, 67, 7, 7 }}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 67, -1, -1}, {{ 1, 67, 9, 4}, { 2, 67, 5, 7}, { 1, 67, 9, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 67, -1, -1}, {{ 2, 67, 7, 7}, { 2, 67, 5, 7}, { 1, 67, 9, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{15, 15}}
    },
    DeconvInputData{
        InputShape{{-1, 67, -1, -1}, {{ 1, 67, 9, 4}, { 2, 67, 5, 7}, { 1, 67, 9, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{9, 9}, {9, 10}, {9, 9}}
    }
};

const auto deconvParams_AutoPadding_2D = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    ::testing::ValuesIn(strides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels_Blocked),
    ::testing::Values(ngraph::op::PadType::SAME_UPPER, ngraph::op::PadType::SAME_LOWER),
    ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_AutoPadding_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        deconvParams_AutoPadding_2D,
        ::testing::ValuesIn(inputs_2D_AutoPadding),
        ::testing::Values(ElementType::f32),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D, conv_avx512_2D})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

const std::vector<DeconvInputData> inputs_3D_AutoPadding = {
    DeconvInputData{
        InputShape{{-1, 2, 4, {32, 64}, {32, 64}}, {{1, 2, 4, 32, 32}, {1, 2, 4, 40, 40}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{8, 64, 64}, {8, 80, 80}}
    },
    DeconvInputData{
        InputShape{
            {1, 64, 5, {1, std::numeric_limits<ov::Dimension::value_type>::max()}, {1, std::numeric_limits<ov::Dimension::value_type>::max()}},
            {{1, 64, 5, 8, 8}}
        },
        ngraph::helpers::InputLayerType::CONSTANT,
        {{10, 16, 16}}
    },
};

const auto deconvParams_AutoPadding_3D = ::testing::Combine(
    ::testing::Values(kernels3d[0]),
    ::testing::Values(strides3d[1]),
    ::testing::ValuesIn(padBegins3d),
    ::testing::ValuesIn(padEnds3d),
    ::testing::ValuesIn(dilations3d),
    ::testing::Values(1),
    ::testing::Values(ngraph::op::PadType::SAME_UPPER, ngraph::op::PadType::SAME_LOWER),
    ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_AutoPadding_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        deconvParams_AutoPadding_3D,
        ::testing::ValuesIn(inputs_3D_AutoPadding),
        ::testing::Values(ElementType::f32),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D, conv_avx512_3D})),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

const auto deconvParams_AutoPadding_2D_AMX = ::testing::Combine(
    ::testing::ValuesIn(deconvAmxKernels2d),
    ::testing::ValuesIn(deconvAmxStrides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::Values(256),
    ::testing::Values(ngraph::op::PadType::SAME_UPPER, ngraph::op::PadType::SAME_LOWER),
    ::testing::ValuesIn(emptyOutputPadding)
);

const DeconvInputData inputs_2D_AutoPadding_AMX = {
    InputShape{{-1, 512, -1, -1}, {{ 1, 512, 32, 51}, { 1, 512, 68, 101}}},
    ngraph::helpers::InputLayerType::PARAMETER,
    {{64, 101}, {135, 202}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_AutoPadding_AMX_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        deconvParams_AutoPadding_2D_AMX,
        ::testing::Values(inputs_2D_AutoPadding_AMX),
        ::testing::Values(ElementType::f32),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_nspc_brgconv_amx})),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
