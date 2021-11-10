// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/convolution_params.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "functional_test_utils/ov_tensor_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include <shared_test_classes/single_layer/group_convolution_backprop_data.hpp>
#include "openvino/core/preprocess/pre_post_process.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using GroupDeconvSpecParams = LayerTestsDefinitions::groupConvBackpropSpecificParams;

using DeconvInputData = std::tuple<InputShape,                           // data shape
                                   ngraph::helpers::InputLayerType,      // 'output_shape' input type
                                   std::vector<std::vector<int32_t>>>;   // values for 'output_shape'

using GroupDeconvLayerCPUTestParamsSet = std::tuple<GroupDeconvSpecParams,
                                                    DeconvInputData,
                                                    ElementType,
                                                    CPUSpecificParams,
                                                    fusingSpecificParams,
                                                    std::map<std::string, std::string>>;

class GroupDeconvolutionLayerCPUTest : public testing::WithParamInterface<GroupDeconvLayerCPUTestParamsSet>,
                                       virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GroupDeconvLayerCPUTestParamsSet> obj) {
        GroupDeconvSpecParams basicParamsSet;
        DeconvInputData inputData;
        ElementType prec;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, inputData, prec, cpuParams, fusingParams, additionalConfig) = obj.param;

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
        size_t convOutChannels, groupNum;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, groupNum, padType, outPadding) = basicParamsSet;

        InputShape inputShape;
        ngraph::helpers::InputLayerType outShapeType;
        std::vector<std::vector<int32_t>> outShapeData;
        std::tie(inputShape, outShapeType, outShapeData) = inputData;

        std::ostringstream result;
        result << "IS=";
        result << CommonTestUtils::partialShape2str({inputShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : inputShape.second) {
            result << "(";
            result << CommonTestUtils::vec2str(shape);
            result << ")_";
        }
        result << "PRC=" << prec << "_";
        result << "K=" << CommonTestUtils::vec2str(kernel) << "_";
        result << "S=" << CommonTestUtils::vec2str(stride) << "_";
        result << "PB=" << CommonTestUtils::vec2str(padBegin) << "_";
        result << "PE=" << CommonTestUtils::vec2str(padEnd) << "_";
        result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
        result << "OP=" << CommonTestUtils::vec2str(outPadding) << "_";
        result << "O=" << convOutChannels << "_";
        result << "G=" << groupNum << "_";
        result << "AP=" << padType << "_";
        result << "OUT_SH=" << outShapeType << "_";
        result << "OUT_D=";
        for (const auto& data : outShapeData) {
            result << "(";
            result << CommonTestUtils::vec2str(data);
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
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::runtime::Tensor tensor;

            if (i == 1) {
                tensor = ov::runtime::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], outShapeData[inferRequestNum].data());
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 2560, 0, 256);
            }

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
        inferRequestNum++;
    }

    void init_ref_function(std::shared_ptr<ov::Function> &funcRef, const std::vector<ov::Shape>& targetInputStaticShapes) override {
        if (function->get_parameters().size() == 1) {
            ngraph::helpers::resize_function(funcRef, targetInputStaticShapes);
        } else {
            funcRef = createGraph({targetInputStaticShapes[0]}, ngraph::helpers::InputLayerType::CONSTANT);
        }
    }

    void validate() override {
        if (function->get_parameters().size() == 2) {
            auto pos = std::find_if(inputs.begin(), inputs.end(),
                [](const std::pair<std::shared_ptr<ov::Node>, ov::runtime::Tensor> &params) {
                    return params.first->get_friendly_name() == "param_1";
                });
            IE_ASSERT(pos != inputs.end());
            inputs.erase(pos);
        }
        SubgraphBaseTest::validate();
    }

    void configure_model() override {
        ov::preprocess::PrePostProcessor p;
        {
            auto& params = function->get_parameters();
            for (size_t i = 0; i < params.size(); i++) {
                if (i > 0) {
                    continue;
                }
                if (inType != ov::element::Type_t::undefined) {
                    p.input(ov::preprocess::InputInfo(i)
                            .tensor(ov::preprocess::InputTensorInfo().set_element_type(inType)));
                }
            }
        }
        {
            auto results = function->get_results();
            for (size_t i = 0; i < results.size(); i++) {
                if (outType != ov::element::Type_t::undefined) {
                    p.output(ov::preprocess::OutputInfo(i)
                                 .tensor(ov::preprocess::OutputTensorInfo().set_element_type(outType)));
                }
            }
        }
        function = p.build(function);
    }

    std::shared_ptr<ov::Function> createGraph(const std::vector<ov::PartialShape>& inShapes, ngraph::helpers::InputLayerType outShapeType) {
        auto params = ngraph::builder::makeDynamicParams(prec, {inShapes.front()});
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
            deconv = ngraph::builder::makeGroupConvolutionBackpropData(params[0], outShapeNode, prec, kernel, stride, padBegin,
                                                                       padEnd, dilation, padType, convOutChannels, groupNum);
        } else {
            deconv = ngraph::builder::makeGroupConvolutionBackpropData(params[0], prec, kernel, stride, padBegin,
                                                                       padEnd, dilation, padType, convOutChannels, groupNum, false, outPadding);
        }

        return makeNgraphFunction(prec, params, deconv, "GroupDeconvCPU");
    }

protected:
    void SetUp() override {
        rel_threshold = 1e-4f;

        targetDevice = CommonTestUtils::DEVICE_CPU;

        GroupDeconvSpecParams basicParamsSet;
        DeconvInputData inputData;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, inputData, prec, cpuParams, fusingParams, additionalConfig) = this->GetParam();

        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, groupNum,  padType, outPadding) = basicParamsSet;

        InputShape inputShape;
        ngraph::helpers::InputLayerType outShapeType;
        std::tie(inputShape, outShapeType, outShapeData) = inputData;

        if (additionalConfig[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] == InferenceEngine::PluginConfigParams::YES) {
            inType = outType = prec = ElementType::bf16;
            rel_threshold = 1e-2f;
        } else {
            inType = outType = prec;
        }

        // TODO: replace with makeSelectedTypeStr
        selectedType += std::string("_") + InferenceEngine::details::convertPrecision(prec).name();

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
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels, groupNum;
    ngraph::helpers::InputLayerType outShapeType;
    std::vector<std::vector<int32_t>> outShapeData;
    size_t inferRequestNum = 0;
};

TEST_P(GroupDeconvolutionLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

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
    CheckPluginRelatedResults(executableNetwork, "Deconvolution");
}

namespace {

/* GROUP CONV TEST UTILS */
std::vector<GroupDeconvLayerCPUTestParamsSet> filterParamsSetForDevice(std::vector<GroupDeconvLayerCPUTestParamsSet> paramsSet) {
    std::vector<GroupDeconvLayerCPUTestParamsSet> resParamsSet;
    const int cpuParamsIndex = 3;
    const int selectedTypeIndex = 3;

    for (auto param : paramsSet) {
        auto cpuParams = std::get<cpuParamsIndex>(param);
        auto selectedTypeStr = std::get<selectedTypeIndex>(cpuParams);

        if (selectedTypeStr.find("jit") != std::string::npos && !InferenceEngine::with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("avx512") != std::string::npos && !InferenceEngine::with_cpu_x86_avx512f())
            continue;

        resParamsSet.push_back(param);
    }

    return resParamsSet;
}
/* ===================== */

/* COMMON PARAMS */
std::vector<fusingSpecificParams> fusingParamsSet {
        emptyFusingSpec,
        fusingScaleShift,
};
const std::map<std::string, std::string> cpuEmptyPluginConfig;
const std::map<std::string, std::string> cpuBF16PluginConfig = { { InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16,
                                                                   InferenceEngine::PluginConfigParams::YES } };

const std::vector<std::vector<size_t >> emptyOutputShape = {{}};
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = {{}};

/* ============= GroupConvolution params (planar layout) ============= */
const InferenceEngine::SizeVector numOutChannels_Planar = {6};
const InferenceEngine::SizeVector numGroups_Planar = {2, 3};

/* ============= GroupConvolution params (blocked layout) ============= */
const InferenceEngine::SizeVector numOutChannels_Blocked = {64};
const InferenceEngine::SizeVector numGroups_Blocked = {2, 4};

/* ============= GroupConvolution params (DW) ============= */
const InferenceEngine::SizeVector numOutChannels_DW = {32};
const InferenceEngine::SizeVector numGroups_DW = {32};

/* ============= GroupConvolution params (2D) ============= */
const std::vector<InferenceEngine::SizeVector> kernels2d = {{3, 3}, {1, 1}};
const std::vector<InferenceEngine::SizeVector> strides2d = {{1, 1}, {2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins2d = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2d = {{0, 0}};
const std::vector<InferenceEngine::SizeVector> dilations2d = {{1, 1}};

/* ============= GroupConvolution params (3D) ============= */
const std::vector<InferenceEngine::SizeVector> kernels3d = {{3, 3, 3}, {1, 1, 1}};
const std::vector<InferenceEngine::SizeVector> strides3d = {{1, 1, 1}, {2, 2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins3d = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3d = {{0, 0, 0}};
const std::vector<InferenceEngine::SizeVector> dilations3d = {{1, 1, 1}};
/* ============= */


/* INSTANCES */
/* ============= GroupConvolution (Planar 2D) ============= */
const std::vector<DeconvInputData> Planar_2D_inputs = {
    DeconvInputData{
        InputShape{{}, {{ 2, 12, 7, 7 }}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 12, -1, -1}, {{ 2, 12, 7, 7}, { 2, 12, 5, 7}, { 1, 12, 9, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 12, -1, -1}, {{ 2, 12, 7, 7}, { 2, 12, 5, 7}, { 1, 12, 9, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{15, 15}}
    },
    DeconvInputData{
        InputShape{{-1, 12, -1, -1}, {{ 2, 12, 7, 7}, { 2, 12, 5, 7}, { 1, 12, 9, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{15, 15}, {9, 10}, {9, 9}}
    }
};

const auto groupConvParams_ExplicitPadding_Planar_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Planar),
        ::testing::ValuesIn(numGroups_Planar),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_Planar_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        groupConvParams_ExplicitPadding_Planar_2D,
        ::testing::ValuesIn(Planar_2D_inputs),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_Planar_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        groupConvParams_ExplicitPadding_Planar_2D,
        ::testing::ValuesIn(Planar_2D_inputs),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (Planar 3D) ============= */
const std::vector<DeconvInputData> Planar_3D_inputs = {
    DeconvInputData{
        InputShape{{}, {{ 2, 12, 7, 7, 7 }}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 12, -1, -1, -1}, {{ 2, 12, 7, 7, 7}, { 2, 12, 5, 7, 7}, { 1, 12, 9, 4, 9}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 12, -1, -1, -1}, {{ 2, 12, 7, 7, 7}, { 2, 12, 5, 7, 7}, { 1, 12, 9, 4, 9}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{15, 15, 15}}
    },
    DeconvInputData{
        InputShape{{-1, 12, -1, -1, -1}, {{ 2, 12, 7, 7, 7}, { 2, 12, 5, 7, 7}, { 1, 12, 9, 4, 9}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{15, 15, 15}, {9, 10, 10}, {9, 9, 9}}
    }
};

const auto groupConvParams_ExplicitPadding_Planar_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::ValuesIn(numOutChannels_Planar),
        ::testing::ValuesIn(numGroups_Planar),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_3D_Planar_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        groupConvParams_ExplicitPadding_Planar_3D,
        ::testing::ValuesIn(Planar_3D_inputs),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_3D_Planar_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        groupConvParams_ExplicitPadding_Planar_3D,
        ::testing::ValuesIn(Planar_3D_inputs),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (Blocked 2D) ============= */
const std::vector<DeconvInputData> Blocked_2D_inputs = {
    DeconvInputData{
        InputShape{{}, {{ 2, 64, 7, 7 }}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 64, -1, -1}, {{ 2, 64, 7, 7}, { 2, 64, 5, 7}, { 1, 64, 9, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 64, -1, -1}, {{ 2, 64, 7, 7}, { 2, 64, 5, 7}, { 1, 64, 9, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{15, 15}}
    },
    DeconvInputData{
        InputShape{{-1, 64, -1, -1}, {{ 2, 64, 7, 7}, { 2, 64, 5, 7}, { 1, 64, 9, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{15, 15}, {9, 10}, {9, 9}}
    }
};

const auto groupConvParams_ExplicitPadding_Blocked_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::ValuesIn(numGroups_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_Blocked_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        groupConvParams_ExplicitPadding_Blocked_2D,
        ::testing::ValuesIn(Blocked_2D_inputs),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_Blocked_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        groupConvParams_ExplicitPadding_Blocked_2D,
        ::testing::ValuesIn(Blocked_2D_inputs),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (Blocked 3D) ============= */
const std::vector<DeconvInputData> Blocked_3D_inputs = {
    DeconvInputData{
        InputShape{{}, {{ 2, 32, 7, 7, 7 }}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 32, -1, -1, -1}, {{ 1, 32, 5, 5, 5}, { 2, 32, 5, 7, 5}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 32, -1, -1, -1}, {{ 1, 32, 5, 5, 5}, { 2, 32, 5, 7, 5}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{7, 7, 7}}
    },
    DeconvInputData{
        InputShape{{-1, 32, -1, -1, -1}, {{ 1, 32, 5, 5, 5}, { 2, 32, 5, 7, 5}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{7, 7, 7}, {7, 9, 7}}
    }
};

const auto groupConvParams_ExplicitPadding_Blocked_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::Values(32),
        ::testing::ValuesIn(numGroups_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_3D_Blocked_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        groupConvParams_ExplicitPadding_Blocked_3D,
        ::testing::ValuesIn(Blocked_3D_inputs),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_3D_Blocked_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        groupConvParams_ExplicitPadding_Blocked_3D,
        ::testing::ValuesIn(Blocked_3D_inputs),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (DW 2D) ============= */
const std::vector<DeconvInputData> dw_2D_inputs = {
    DeconvInputData{
        InputShape{{}, {{ 2, 32, 7, 7 }}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 32, -1, -1}, {{ 1, 32, 5, 5}, { 2, 32, 5, 7}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {}
    },
    DeconvInputData{
        InputShape{{-1, 32, -1, -1}, {{ 1, 32, 5, 5}, { 2, 32, 5, 7}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{7, 7}}
    },
    DeconvInputData{
        InputShape{{-1, 32, -1, -1}, {{ 1, 32, 5, 5}, { 2, 32, 5, 7}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{7, 7}, {7, 9}}
    }
};

const auto groupConvParams_ExplicitPadding_DW_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_DW),
        ::testing::ValuesIn(numGroups_DW),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_DW_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        groupConvParams_ExplicitPadding_DW_2D,
        ::testing::ValuesIn(dw_2D_inputs),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_dw_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_DW_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        groupConvParams_ExplicitPadding_DW_2D,
        ::testing::ValuesIn(dw_2D_inputs),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_dw_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
