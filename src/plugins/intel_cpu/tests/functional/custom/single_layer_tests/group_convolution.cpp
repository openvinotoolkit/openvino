// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/group_convolution.hpp"

#include "common_test_utils/node_builders/group_convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/convolution_params.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {

using groupConvSpecificParams = ov::test::groupConvSpecificParams;

typedef std::tuple<groupConvSpecificParams,
                   ElementType,
                   ElementType,  // Input precision
                   ElementType,  // Output precision
                   InputShape,   // Input shapes
                   std::string>
    groupConvLayerTestsParamsSet;

typedef std::tuple<groupConvLayerTestsParamsSet, CPUSpecificParams, fusingSpecificParams, ov::AnyMap>
    groupConvLayerCPUTestParamsSet;

class GroupConvolutionLayerCPUTest : public testing::WithParamInterface<groupConvLayerCPUTestParamsSet>,
                                     virtual public SubgraphBaseTest,
                                     public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<groupConvLayerCPUTestParamsSet> obj) {
        groupConvLayerTestsParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        ov::AnyMap additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = obj.param;

        groupConvSpecificParams groupConvParams;
        ElementType netType;
        ElementType inType, outType;
        InputShape inputShape;
        std::string targetDevice;
        std::tie(groupConvParams, netType, inType, outType, inputShape, targetDevice) = basicParamsSet;
        ov::op::PadType padType;
        std::vector<size_t> kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels, numGroups;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvParams;

        std::ostringstream result;
        result << "IS=";
        result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
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
        result << "G=" << numGroups << "_";
        result << "AP=" << padType << "_";
        result << "netPRC=" << netType << "_";
        result << "inPRC=" << inType << "_";
        result << "outPRC=" << outType << "_";
        result << "trgDev=" << targetDevice;

        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second.as<std::string>();
            }
        }

        return result.str();
    }

protected:
    bool isBias = false;

    void checkBiasFusing(ov::CompiledModel& execNet) const {
        auto execGraph = execNet.get_runtime_model();
        ASSERT_NE(nullptr, execGraph);

        bool foundConv = false;
        for (const auto& node : execGraph->get_ops()) {
            const auto& rtInfo = node->get_rt_info();
            auto getExecValue = [&rtInfo](const std::string& paramName) -> std::string {
                auto it = rtInfo.find(paramName);
                OPENVINO_ASSERT(rtInfo.end() != it);
                return it->second.as<std::string>();
            };

            if (getExecValue(ov::exec_model_info::LAYER_TYPE) == "Convolution") {
                foundConv = true;
                ASSERT_EQ(3, node->inputs().size());
                break;
            }
        }

        ASSERT_TRUE(foundConv) << "Can't find Convolution node";
    }

    std::shared_ptr<ov::Node> modifyGraph(const ov::element::Type& ngPrc,
                                          ov::ParameterVector& params,
                                          const std::shared_ptr<ov::Node>& lastNode) override {
        auto retNode = CpuTestWithFusing::modifyGraph(ngPrc, params, lastNode);
        std::shared_ptr<ov::Node> opToShapeInfer = nullptr;
        for (auto& targetShapes : targetStaticShapes) {
            for (size_t i = targetShapes.size(); i < params.size(); ++i) {
                const auto& shape = params[i]->get_output_partial_shape(0);
                if (shape.is_static()) {
                    targetShapes.push_back(shape.get_shape());
                } else {
                    // It is assumed that in such tests we have second parameter only if sum fusion is tested.
                    // Considering this fact, we need to set the appropriate static shape for the second term of the sum
                    // operation, and it has to match the convolution output shape. So the most suitable solution here
                    // is to perform shape inference on the convolution node
                    if (!opToShapeInfer) {
                        ov::OutputVector inputsForShapeInfer;
                        for (size_t j = 0; j < lastNode->get_input_size(); j++) {
                            if (ov::is_type<ov::op::v0::Constant>(lastNode->get_input_node_ptr(j))) {
                                inputsForShapeInfer.push_back(lastNode->get_input_node_shared_ptr(j));
                            } else {
                                inputsForShapeInfer.push_back(
                                    std::make_shared<ov::op::v0::Parameter>(lastNode->get_input_element_type(j),
                                                                            lastNode->get_input_partial_shape(j)));
                            }
                        }
                        opToShapeInfer = lastNode->clone_with_new_inputs(inputsForShapeInfer);
                    }

                    std::vector<ov::Shape> secondParameterShapes;
                    if (auto parameter = dynamic_cast<ov::op::v0::Parameter*>(opToShapeInfer->get_input_node_ptr(0))) {
                        parameter->set_partial_shape(targetShapes.front());
                        parameter->validate_and_infer_types();
                    }
                    opToShapeInfer->validate_and_infer_types();
                    targetShapes.push_back(opToShapeInfer->get_output_shape(0));
                }
            }
        }
        return retNode;
    }

    void SetUp() override {
        rel_threshold = 1e-4f;

        groupConvLayerTestsParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        ov::AnyMap additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = this->GetParam();

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        if (postOpMgrPtr)
            isBias = postOpMgrPtr->getFusedOpsNames() == "Add(PerChannel)";

        groupConvSpecificParams groupConvParams;
        InputShape inputShape;
        auto netType = ElementType::undefined;
        std::tie(groupConvParams, netType, inType, outType, inputShape, targetDevice) = basicParamsSet;

        init_input_shapes({inputShape});

        if (configuration.count(ov::hint::inference_precision.name()) &&
            ov::element::bf16 == configuration[ov::hint::inference_precision.name()].as<ov::element::Type>()) {
            selectedType += "_bf16";
            rel_threshold = 1e-2f;
        } else {
            selectedType = makeSelectedTypeStr(selectedType, netType);
        }

        // according to range propagation feature, resolution of generated inputs data for parameters moved from 32 to 32768
        // 'real' part of input data was changed and some fails became visible for cases with Elu and FakeQuantize, so let's setup abs_threshold
        if (fusedOps.size() == 3 && fusedOps[1] == std::string("Elu") && fusedOps[2] == std::string("FakeQuantize")) {
            abs_threshold = 5e-3f;
        }

        ov::op::PadType padType;
        std::vector<size_t> kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels, numGroups;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvParams;

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netType, shape));

        auto groupConv = ov::test::utils::make_group_convolution(params[0],
                                                                 netType,
                                                                 kernel,
                                                                 stride,
                                                                 padBegin,
                                                                 padEnd,
                                                                 dilation,
                                                                 padType,
                                                                 convOutChannels,
                                                                 numGroups);
        function = makeNgraphFunction(netType, params, groupConv, "groupConvolution");
    }
};

using ExpectFallbackGroupConvolutionLayerCPUTest = GroupConvolutionLayerCPUTest;

TEST_P(ExpectFallbackGroupConvolutionLayerCPUTest, CompareWithRefs) {
    run();
    if (isBias) {
        checkBiasFusing(compiledModel);
    }
    ASSERT_TRUE(!selectedType.empty()) << "Node type is not defined.";
    auto function = compiledModel.get_runtime_model();
    for (const auto& node : function->get_ops()) {
        const auto& rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string& paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            OPENVINO_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };
        if ("Convolution" == getExecValue(ov::exec_model_info::LAYER_TYPE)) {
            auto primType = getExecValue(ov::exec_model_info::IMPL_TYPE);
            ASSERT_TRUE(selectedType != primType) << "primType is unexpected: " << primType;
        }
    }
}

TEST_P(GroupConvolutionLayerCPUTest, CompareWithRefs) {
    run();
    if (isBias) {
        checkBiasFusing(compiledModel);
    }
    CheckPluginRelatedResults(compiledModel, "Convolution");
}

namespace {

/* GROUP CONV TEST UTILS */
std::vector<groupConvLayerCPUTestParamsSet> filterParamsSetForDevice(
    std::vector<groupConvLayerCPUTestParamsSet> paramsSet) {
    std::vector<groupConvLayerCPUTestParamsSet> resParamsSet;
    const int cpuParamsIndex = 1;
    const int selectedTypeIndex = 3;
    const int configIndex = 3;

    for (auto param : paramsSet) {
        auto cpuParams = std::get<cpuParamsIndex>(param);
        auto selectedTypeStr = std::get<selectedTypeIndex>(cpuParams);

        if (selectedTypeStr.find("jit") != std::string::npos && !ov::with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("sse42") != std::string::npos && !ov::with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("avx2") != std::string::npos && !ov::with_cpu_x86_avx2())
            continue;
        if (selectedTypeStr.find("avx512") != std::string::npos && !ov::with_cpu_x86_avx512f())
            continue;
        if (selectedTypeStr.find("amx") != std::string::npos && !ov::with_cpu_x86_avx512_core_amx())
            continue;
        auto additionalConfig = std::get<configIndex>(param);
        if (additionalConfig.count(ov::hint::inference_precision.name()) &&
            ov::element::bf16 == additionalConfig[ov::hint::inference_precision.name()].as<ov::element::Type>() &&
            !ov::with_cpu_x86_bfloat16()) {
            continue;
        }
        resParamsSet.push_back(param);
    }

    return resParamsSet;
}

std::vector<CPUSpecificParams> filterCPUInfoForDeviceSupportBF16(std::vector<CPUSpecificParams> CPUParams) {
    std::vector<CPUSpecificParams> resParamsSet;
    if (ov::with_cpu_x86_bfloat16()) {
        return filterCPUInfoForDevice(CPUParams);
    }
    return resParamsSet;
}
/* ===================== */

/* COMMON PARAMS */
const std::vector<fusingSpecificParams> fusingParamsSet{emptyFusingSpec,
                                                        // eltwise
                                                        fusingRelu,
                                                        fusingPRelu1D,
                                                        // depthwise
                                                        fusingReluScaleShift,
                                                        // fake quantize
                                                        fusingFakeQuantizePerTensorRelu,
                                                        fusingFakeQuantizePerChannelRelu,
                                                        // sum
                                                        fusingSumEluFQ,
                                                        fusingSum};

const std::vector<fusingSpecificParams> fusingParamsSetBF16{emptyFusingSpec,
                                                            // eltwise
                                                            fusingRelu,
                                                            // depthwise
                                                            fusingReluScaleShift,
                                                            // sum
                                                            fusingSum};

/* ============= GroupConvolution params (planar layout) ============= */
const std::vector<size_t> numOutChannels_Gemm = {6};
const std::vector<size_t> numGroups_Gemm = {2, 3};

/* ============= GroupConvolution params (blocked layout) ============= */
const std::vector<size_t> numOutChannels_Blocked = {64};
const std::vector<size_t> numGroups_Blocked = {2, 4};

/* ============= GroupConvolution params (DW) ============= */
const std::vector<size_t> numOutChannels_DW = {32};
const std::vector<size_t> numGroups_DW = {32};

/* ============= GroupConvolution params (1D) ============= */
const std::vector<std::vector<size_t>> kernels1d = {{3}, {1}};
const std::vector<std::vector<size_t>> strides1d = {{1}, {2}};
const std::vector<std::vector<ptrdiff_t>> padBegins1d = {{0}, {1}};
const std::vector<std::vector<ptrdiff_t>> padEnds1d = {{0}};
const std::vector<std::vector<size_t>> dilations1d = {{1}, {2}};

/* ============= GroupConvolution params (2D) ============= */
const std::vector<std::vector<size_t>> kernels2d = {{3, 3}, {1, 1}};
const std::vector<std::vector<size_t>> strides2d = {{1, 1}, {2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins2d = {{0, 0}, {1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds2d = {{0, 0}};
const std::vector<std::vector<size_t>> dilations2d = {{1, 1}, {2, 2}};

/* ============= GroupConvolution params (3D) ============= */
const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}, {1, 1, 1}};
const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}, {2, 2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins3d = {{0, 0, 0}, {1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds3d = {{0, 0, 0}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}, {2, 2, 2}};
/* ============= */

/* INSTANCES */
/* ============= GroupConvolution (GEMM 1D) ============= */
const auto groupConvParams_ExplicitPadding_Gemm_1D = ::testing::Combine(::testing::ValuesIn(kernels1d),
                                                                        ::testing::ValuesIn(strides1d),
                                                                        ::testing::ValuesIn(padBegins1d),
                                                                        ::testing::ValuesIn(padEnds1d),
                                                                        ::testing::ValuesIn(dilations1d),
                                                                        ::testing::ValuesIn(numOutChannels_Gemm),
                                                                        ::testing::ValuesIn(numGroups_Gemm),
                                                                        ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_Gemm_1D = {conv_gemm_1D, conv_gemm_1D_nspc};

std::vector<InputShape> inShapesGemm1D = {{{}, {{2, 12, 7}}},
                                          {// dynamic shape
                                           {{1, 200}, 12, {1, 200}},
                                           {// target static shapes
                                            {2, 12, 7},
                                            {1, 12, 5}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_1D_Gemm_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_Gemm_1D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inShapesGemm1D),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Gemm_1D)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_1D_Gemm_with_bias_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_Gemm_1D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inShapesGemm1D),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Gemm_1D)),
                                            ::testing::Values(fusingAddPerChannel),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_1D_Gemm_BF16,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_Gemm_1D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inShapesGemm1D),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(
                                                {conv_gemm_1D})),  // todo: [AV] what about conv_gemm_1D_nspc?
                                            ::testing::ValuesIn(fusingParamsSetBF16),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (GEMM 2D) ============= */
const auto groupConvParams_ExplicitPadding_Gemm_2D = ::testing::Combine(::testing::ValuesIn(kernels2d),
                                                                        ::testing::ValuesIn(strides2d),
                                                                        ::testing::ValuesIn(padBegins2d),
                                                                        ::testing::ValuesIn(padEnds2d),
                                                                        ::testing::ValuesIn(dilations2d),
                                                                        ::testing::ValuesIn(numOutChannels_Gemm),
                                                                        ::testing::ValuesIn(numGroups_Gemm),
                                                                        ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_Gemm_2D = {conv_gemm_2D, conv_gemm_2D_nspc};

std::vector<InputShape> inShapesGemm2D = {{{}, {{2, 12, 7, 7}}},
                                          {// dynamic shape
                                           {{1, 200}, 12, -1, {1, 200}},
                                           {// target static shapes
                                            {2, 12, 7, 7},
                                            {1, 12, 5, 5}}}};

std::vector<InputShape> inShapesGemm2D_cache = {{{}, {{2, 12, 7, 7}}},
                                                {// dynamic shape
                                                 {{1, 200}, 12, -1, {1, 200}},
                                                 {// target static shapes
                                                  {1, 12, 5, 5},
                                                  {1, 12, 7, 7},
                                                  {1, 12, 5, 5}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_2D_Gemm_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_Gemm_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inShapesGemm2D_cache),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Gemm_2D)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_2D_Gemm_with_bias_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_Gemm_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inShapesGemm2D_cache),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Gemm_2D)),
                                            ::testing::Values(fusingAddPerChannel),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_2D_Gemm_BF16,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_Gemm_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inShapesGemm2D),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Gemm_2D)),
                                            ::testing::ValuesIn(fusingParamsSetBF16),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (Gemm 3D) ============= */
const auto groupConvParams_ExplicitPadding_Gemm_3D = ::testing::Combine(::testing::ValuesIn(kernels3d),
                                                                        ::testing::ValuesIn(strides3d),
                                                                        ::testing::ValuesIn(padBegins3d),
                                                                        ::testing::ValuesIn(padEnds3d),
                                                                        ::testing::ValuesIn(dilations3d),
                                                                        ::testing::ValuesIn(numOutChannels_Gemm),
                                                                        ::testing::ValuesIn(numGroups_Gemm),
                                                                        ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_Gemm_3D = {conv_gemm_3D, conv_gemm_3D_nspc};

std::vector<InputShape> inShapesGemm3D = {{{}, {{2, 12, 7, 7, 7}}},
                                          {// dynamic shape
                                           {{1, 200}, 12, -1, {1, 200}, -1},
                                           {// target static shapes
                                            {2, 12, 7, 7, 7},
                                            {1, 12, 5, 5, 5}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_3D_Gemm_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_Gemm_3D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inShapesGemm3D),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Gemm_3D)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_3D_Gemm_with_bias_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_Gemm_3D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inShapesGemm3D),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Gemm_3D)),
                                            ::testing::Values(fusingAddPerChannel),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_3D_Gemm_BF16,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_Gemm_3D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inShapesGemm3D),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Gemm_3D)),
                                            ::testing::ValuesIn(fusingParamsSetBF16),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution params (brgemm_1D) ============= */
const std::vector<std::vector<size_t>> kernels_brgemm_1d = {{3}};
const std::vector<std::vector<size_t>> strides_brgemm_1d = {{1}, {2}};
const std::vector<std::vector<ptrdiff_t>> padBegins_brgemm_1d = {{0}, {1}};
const std::vector<std::vector<ptrdiff_t>> padEnds_brgemm_1d = {{0}};
const std::vector<std::vector<size_t>> dilations_brgemm_1d = {{1}, {2}};

/* ============= GroupConvolution params (brgemm_2D) ============= */
const std::vector<std::vector<size_t>> kernels_brgemm_2d = {{3, 3}};
const std::vector<std::vector<size_t>> strides_brgemm_2d = {{1, 1}, {2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins_brgemm_2d = {{0, 0}, {1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds_brgemm_2d = {{0, 0}};
const std::vector<std::vector<size_t>> dilations_brgemm_2d = {{1, 1}, {2, 2}};

/* ============= GroupConvolution params (brgemm_3D) ============= */
const std::vector<std::vector<size_t>> kernels_brgemm_3d = {{3, 3, 3}};
const std::vector<std::vector<size_t>> strides_brgemm_3d = {{1, 1, 1}, {2, 2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins_brgemm_3d = {{0, 0, 0}, {1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds_brgemm_3d = {{0, 0, 0}};
const std::vector<std::vector<size_t>> dilations_brgemm_3d = {{1, 1, 1}, {2, 2, 2}};
/* ============= */

const std::vector<size_t> numGroups_brgemm_Blocked = {2};

/* ============= GroupConvolution (brgemm 1D) ============= */
const auto groupConvParams_ExplicitPadding_brgemm_1D = ::testing::Combine(::testing::ValuesIn(kernels_brgemm_1d),
                                                                          ::testing::ValuesIn(strides_brgemm_1d),
                                                                          ::testing::ValuesIn(padBegins_brgemm_1d),
                                                                          ::testing::ValuesIn(padEnds_brgemm_1d),
                                                                          ::testing::ValuesIn(dilations_brgemm_1d),
                                                                          ::testing::ValuesIn(numOutChannels_Blocked),
                                                                          ::testing::ValuesIn(numGroups_brgemm_Blocked),
                                                                          ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_brgemm_1D_BF16 = {conv_avx512_1D_nspc_brgconv};

const std::vector<CPUSpecificParams> CPUParams_brgemm_1D_FP32 = {conv_avx512_1D_nspc_brgconv};

std::vector<InputShape> inputShapes_brgemm_1d = {{{}, {{2, 64, 7}}},
                                                 {// dynamic shapes
                                                  {-1, 64, {1, 200}},
                                                  {// target static shapes
                                                   {2, 64, 7},
                                                   {1, 64, 9}}},
                                                 {// dynamic shapes
                                                  {{-1, 64, -1}},
                                                  {// target static shapes
                                                   {2, 64, 7},
                                                   {1, 64, 14}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_brgemm_1D_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_1D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes_brgemm_1d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_brgemm_1D_FP32)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_brgemm_1D_FP32_fusingBias,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_1D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes_brgemm_1d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_brgemm_1D_FP32)),
                                            ::testing::Values(fusingAddPerChannel),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConv_brgemm_1D_BF16,
    GroupConvolutionLayerCPUTest,
    ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_1D,
                                          ::testing::Values(ElementType::f32),
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::ValuesIn(inputShapes_brgemm_1d),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::ValuesIn(filterCPUInfoForDeviceSupportBF16(CPUParams_brgemm_1D_BF16)),
                       ::testing::ValuesIn(fusingParamsSetBF16),
                       ::testing::Values(cpu_bf16_plugin_config)),
    GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (brgemm_2D) ============= */
const auto groupConvParams_ExplicitPadding_brgemm_2D = ::testing::Combine(::testing::ValuesIn(kernels_brgemm_2d),
                                                                          ::testing::ValuesIn(strides_brgemm_2d),
                                                                          ::testing::ValuesIn(padBegins_brgemm_2d),
                                                                          ::testing::ValuesIn(padEnds_brgemm_2d),
                                                                          ::testing::ValuesIn(dilations_brgemm_2d),
                                                                          ::testing::ValuesIn(numOutChannels_Blocked),
                                                                          ::testing::ValuesIn(numGroups_brgemm_Blocked),
                                                                          ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_brgemm_2D_FP32 = {conv_avx512_2D_nspc_brgconv};

const std::vector<CPUSpecificParams> CPUParams_brgemm_2D_BF16 = {conv_avx512_2D_nspc_brgconv,
                                                                 conv_avx512_2D_nspc_brgconv_amx};

std::vector<InputShape> inputShapes_brgemm_2d = {{{}, {{1, 64, 7, 7}}},
                                                 {// dynamic shapes
                                                  {-1, 64, -1, {1, 200}},
                                                  {// target static shapes
                                                   {2, 64, 7, 7},
                                                   {1, 64, 9, 9}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_brgemm_2D_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes_brgemm_2d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_brgemm_2D_FP32)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

std::vector<InputShape> inputShapes_brgemm_2d_dynBatch = {{// dynamic shapes
                                                           {{1, 10}, 64, {7, 9}, {7, 9}},
                                                           {// target static shapes
                                                            {2, 64, 7, 7},
                                                            {1, 64, 9, 9},
                                                            {3, 64, 9, 9}}}};

INSTANTIATE_TEST_SUITE_P(nightly_GroupConv_brgemm_2D_FP32_dynBatch,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes_brgemm_2d_dynBatch),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_brgemm_2D_FP32)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

std::vector<InputShape> inputShapes_brgemm_2d_cache = {{// dynamic shapes
                                                        {-1, 64, -1, {1, 200}},
                                                        {
                                                            // target static shapes
                                                            {1, 64, 7, 7},
                                                            {1, 64, 9, 9},
                                                            {1, 64, 7, 7},
                                                        }}};

INSTANTIATE_TEST_SUITE_P(nightly_GroupConv_brgemm_2D_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes_brgemm_2d_cache),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_brgemm_2D_FP32)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConv_brgemm_2D_BF16,
    GroupConvolutionLayerCPUTest,
    ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_2D,
                                          ::testing::Values(ElementType::f32),
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::ValuesIn(inputShapes_brgemm_2d),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::ValuesIn(filterCPUInfoForDeviceSupportBF16(CPUParams_brgemm_2D_BF16)),
                       ::testing::ValuesIn(fusingParamsSetBF16),
                       ::testing::Values(cpu_bf16_plugin_config)),
    GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (brgemm_3D) ============= */
const auto groupConvParams_ExplicitPadding_brgemm_3D = ::testing::Combine(::testing::ValuesIn(kernels_brgemm_3d),
                                                                          ::testing::ValuesIn(strides_brgemm_3d),
                                                                          ::testing::ValuesIn(padBegins_brgemm_3d),
                                                                          ::testing::ValuesIn(padEnds_brgemm_3d),
                                                                          ::testing::ValuesIn(dilations_brgemm_3d),
                                                                          ::testing::ValuesIn(numOutChannels_Blocked),
                                                                          ::testing::ValuesIn(numGroups_brgemm_Blocked),
                                                                          ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_brgemm_3D_FP32 = {conv_avx512_3D_nspc_brgconv};

const std::vector<CPUSpecificParams> CPUParams_brgemm_3D_BF16 = {conv_avx512_3D_nspc_brgconv,
                                                                 conv_avx512_3D_nspc_brgconv_amx};

std::vector<InputShape> inputShapes_brgemm_3d = {{{}, {{1, 64, 7, 7, 7}}},
                                                 {// dynamic shapes
                                                  {-1, 64, -1, {1, 200}, -1},
                                                  {// target static shapes
                                                   {2, 64, 7, 7, 7},
                                                   {1, 64, 9, 9, 9}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_brgemm_3D_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_3D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes_brgemm_3d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_brgemm_3D_FP32)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConv_brgemm_3D_BF16,
    GroupConvolutionLayerCPUTest,
    ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_3D,
                                          ::testing::Values(ElementType::f32),
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::ValuesIn(inputShapes_brgemm_3d),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::ValuesIn(filterCPUInfoForDeviceSupportBF16(CPUParams_brgemm_3D_BF16)),
                       ::testing::ValuesIn(fusingParamsSetBF16),
                       ::testing::Values(cpu_bf16_plugin_config)),
    GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution params (brgemm_1x1_1D) ============= */
const std::vector<std::vector<size_t>> kernels_brgemm_1x1_1d = {{1}};
const std::vector<std::vector<size_t>> strides_brgemm_1x1_1d = {{1}, {2}};
const std::vector<std::vector<ptrdiff_t>> padBegins_brgemm_1x1_1d = {{0}};
const std::vector<std::vector<ptrdiff_t>> padEnds_brgemm_1x1_1d = {{0}};
const std::vector<std::vector<size_t>> dilations_brgemm_1x1_1d = {{1}, {2}};

/* ============= GroupConvolution params (brgemm_1x1_2D) ============= */
const std::vector<std::vector<size_t>> kernels_brgemm_1x1_2d = {{1, 1}};
const std::vector<std::vector<size_t>> strides_brgemm_1x1_2d = {{1, 1}, {2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins_brgemm_1x1_2d = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds_brgemm_1x1_2d = {{0, 0}};
const std::vector<std::vector<size_t>> dilations_brgemm_1x1_2d = {{1, 1}, {2, 2}};

/* ============= GroupConvolution params (brgemm_1x1_3D) ============= */
const std::vector<std::vector<size_t>> kernels_brgemm_1x1_3d = {{1, 1, 1}};
const std::vector<std::vector<size_t>> strides_brgemm_1x1_3d = {{1, 1, 1}, {2, 2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins_brgemm_1x1_3d = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds_brgemm_1x1_3d = {{0, 0, 0}};
const std::vector<std::vector<size_t>> dilations_brgemm_1x1_3d = {{1, 1, 1}, {2, 2, 2}};
/* ============= */

const std::vector<size_t> numGroups_brgemm_1x1_Blocked = {2};

/* ============= GroupConvolution (brgemm_1x1 1D) ============= */
const auto groupConvParams_ExplicitPadding_brgemm_1x1_1D =
    ::testing::Combine(::testing::ValuesIn(kernels_brgemm_1x1_1d),
                       ::testing::ValuesIn(strides_brgemm_1x1_1d),
                       ::testing::ValuesIn(padBegins_brgemm_1x1_1d),
                       ::testing::ValuesIn(padEnds_brgemm_1x1_1d),
                       ::testing::ValuesIn(dilations_brgemm_1x1_1d),
                       ::testing::ValuesIn(numOutChannels_Blocked),
                       ::testing::ValuesIn(numGroups_brgemm_1x1_Blocked),
                       ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_brgemm_1x1_1D_BF16 = {
    conv_avx512_1D_1x1_nspc_brgconv,
};

const std::vector<CPUSpecificParams> CPUParams_brgemm_1x1_1D_FP32 = {
    conv_avx512_1D_1x1_nspc_brgconv,
};

std::vector<InputShape> inputShapes_brgemm_1x1_1d = {{{}, {{2, 64, 7}}},
                                                     {// dynamic shapes
                                                      {-1, 64, {1, 200}},
                                                      {// target static shapes
                                                       {2, 64, 7},
                                                       {1, 64, 9}}},
                                                     {// dynamic shapes
                                                      {{-1, 64, -1}},
                                                      {// target static shapes
                                                       {2, 64, 7},
                                                       {1, 64, 14}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_brgemm_1x1_1D_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_1x1_1D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes_brgemm_1x1_1d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_brgemm_1x1_1D_FP32)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_brgemm_1x1_1D_FP32_fusingBias,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_1x1_1D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes_brgemm_1x1_1d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_brgemm_1x1_1D_FP32)),
                                            ::testing::Values(fusingAddPerChannel),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConv_brgemm_1x1_1D_BF16,
    GroupConvolutionLayerCPUTest,
    ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_1x1_1D,
                                          ::testing::Values(ElementType::f32),
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::ValuesIn(inputShapes_brgemm_1x1_1d),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::ValuesIn(filterCPUInfoForDeviceSupportBF16(CPUParams_brgemm_1x1_1D_BF16)),
                       ::testing::ValuesIn(fusingParamsSetBF16),
                       ::testing::Values(cpu_bf16_plugin_config)),
    GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (brgemm_1x1_2D) ============= */
const auto groupConvParams_ExplicitPadding_brgemm_1x1_2D =
    ::testing::Combine(::testing::ValuesIn(kernels_brgemm_1x1_2d),
                       ::testing::ValuesIn(strides_brgemm_1x1_2d),
                       ::testing::ValuesIn(padBegins_brgemm_1x1_2d),
                       ::testing::ValuesIn(padEnds_brgemm_1x1_2d),
                       ::testing::ValuesIn(dilations_brgemm_1x1_2d),
                       ::testing::ValuesIn(numOutChannels_Blocked),
                       ::testing::ValuesIn(numGroups_brgemm_1x1_Blocked),
                       ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_brgemm_1x1_2D_FP32 = {conv_avx512_2D_1x1_nspc_brgconv};

const std::vector<CPUSpecificParams> CPUParams_brgemm_1x1_2D_BF16 = {conv_avx512_2D_1x1_nspc_brgconv,
                                                                     conv_avx512_2D_1x1_nspc_brgconv_amx};

std::vector<InputShape> inputShapes_brgemm_1x1_2d = {{{}, {{1, 64, 7, 7}}},
                                                     {// dynamic shapes
                                                      {-1, 64, -1, {1, 200}},
                                                      {// target static shapes
                                                       {2, 64, 7, 7},
                                                       {1, 64, 9, 9}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_brgemm_1x1_2D_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_1x1_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes_brgemm_1x1_2d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_brgemm_1x1_2D_FP32)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

std::vector<InputShape> inputShapes_brgemm_1x1_2d_dynBatch = {{// dynamic shapes
                                                               {{1, 10}, 64, {7, 9}, {7, 9}},
                                                               {// target static shapes
                                                                {2, 64, 7, 7},
                                                                {1, 64, 9, 9},
                                                                {3, 64, 9, 9}}}};

INSTANTIATE_TEST_SUITE_P(nightly_GroupConv_brgemm_1x1_2D_FP32_dynBatch,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_1x1_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes_brgemm_1x1_2d_dynBatch),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_brgemm_1x1_2D_FP32)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

std::vector<InputShape> inputShapes_brgemm_1x1_2d_cache = {{// dynamic shapes
                                                            {-1, 64, -1, {1, 200}},
                                                            {
                                                                // target static shapes
                                                                {1, 64, 7, 7},
                                                                {1, 64, 9, 9},
                                                                {1, 64, 7, 7},
                                                            }}};

INSTANTIATE_TEST_SUITE_P(nightly_GroupConv_brgemm_1x1_2D_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_1x1_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes_brgemm_1x1_2d_cache),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_brgemm_1x1_2D_FP32)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConv_brgemm_1x1_2D_BF16,
    GroupConvolutionLayerCPUTest,
    ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_1x1_2D,
                                          ::testing::Values(ElementType::f32),
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::ValuesIn(inputShapes_brgemm_1x1_2d),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::ValuesIn(filterCPUInfoForDeviceSupportBF16(CPUParams_brgemm_1x1_2D_BF16)),
                       ::testing::ValuesIn(fusingParamsSetBF16),
                       ::testing::Values(cpu_bf16_plugin_config)),
    GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (brgemm_1x1_3D) ============= */
const auto groupConvParams_ExplicitPadding_brgemm_1x1_3D =
    ::testing::Combine(::testing::ValuesIn(kernels_brgemm_1x1_3d),
                       ::testing::ValuesIn(strides_brgemm_1x1_3d),
                       ::testing::ValuesIn(padBegins_brgemm_1x1_3d),
                       ::testing::ValuesIn(padEnds_brgemm_1x1_3d),
                       ::testing::ValuesIn(dilations_brgemm_1x1_3d),
                       ::testing::ValuesIn(numOutChannels_Blocked),
                       ::testing::ValuesIn(numGroups_brgemm_1x1_Blocked),
                       ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_brgemm_1x1_3D_FP32 = {
    conv_avx512_3D_1x1_nspc_brgconv,
};

const std::vector<CPUSpecificParams> CPUParams_brgemm_1x1_3D_BF16 = {
    conv_avx512_3D_1x1_nspc_brgconv,
    conv_avx512_3D_1x1_nspc_brgconv_amx,
};

std::vector<InputShape> inputShapes_brgemm_1x1_3d = {{{}, {{1, 64, 7, 7, 7}}},
                                                     {// dynamic shapes
                                                      {-1, 64, -1, {1, 200}, -1},
                                                      {// target static shapes
                                                       {2, 64, 7, 7, 7},
                                                       {1, 64, 9, 9, 9}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_brgemm_1x1_3D_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_1x1_3D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes_brgemm_1x1_3d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_brgemm_1x1_3D_FP32)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConv_brgemm_1x1_3D_BF16,
    GroupConvolutionLayerCPUTest,
    ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_brgemm_1x1_3D,
                                          ::testing::Values(ElementType::f32),
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::ValuesIn(inputShapes_brgemm_1x1_3d),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ::testing::ValuesIn(filterCPUInfoForDeviceSupportBF16(CPUParams_brgemm_1x1_3D_BF16)),
                       ::testing::ValuesIn(fusingParamsSetBF16),
                       ::testing::Values(cpu_bf16_plugin_config)),
    GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (1D) ============= */
const auto groupConvParams_ExplicitPadding_1D = ::testing::Combine(::testing::ValuesIn(kernels1d),
                                                                   ::testing::ValuesIn(strides1d),
                                                                   ::testing::ValuesIn(padBegins1d),
                                                                   ::testing::ValuesIn(padEnds1d),
                                                                   ::testing::ValuesIn(dilations1d),
                                                                   ::testing::ValuesIn(numOutChannels_Blocked),
                                                                   ::testing::ValuesIn(numGroups_Blocked),
                                                                   ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_1D =
    {conv_sse42_1D, conv_avx2_1D, conv_avx512_1D, conv_sse42_1D_nspc, conv_avx2_1D_nspc, conv_avx512_1D_nspc};

std::vector<InputShape> inputShapes1d = {{{}, {{2, 64, 7}}},
                                         {// dynamic shapes
                                          {-1, 64, {1, 200}},
                                          {// target static shapes
                                           {2, 64, 7},
                                           {1, 64, 9}}},
                                         {// dynamic shapes
                                          {{-1, 64, -1}},
                                          {// target static shapes
                                           {2, 64, 7},
                                           {1, 64, 14}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_1D_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_1D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes1d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1D)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_1D_FP32_fusingBias,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_1D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes1d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1D)),
                                            ::testing::Values(fusingAddPerChannel),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_1D_BF16,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_1D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes1d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(
                                                {conv_avx512_1D})),  // todo: [AV] what about conv_avx512_1D_nspc?
                                            ::testing::ValuesIn(fusingParamsSetBF16),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (2D) ============= */
const auto groupConvParams_ExplicitPadding_2D = ::testing::Combine(::testing::ValuesIn(kernels2d),
                                                                   ::testing::ValuesIn(strides2d),
                                                                   ::testing::ValuesIn(padBegins2d),
                                                                   ::testing::ValuesIn(padEnds2d),
                                                                   ::testing::ValuesIn(dilations2d),
                                                                   ::testing::ValuesIn(numOutChannels_Blocked),
                                                                   ::testing::ValuesIn(numGroups_Blocked),
                                                                   ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_2D =
    {conv_sse42_2D, conv_avx2_2D, conv_avx512_2D, conv_sse42_2D_nspc, conv_avx2_2D_nspc, conv_avx512_2D_nspc};

std::vector<InputShape> inputShapes2d = {{{}, {{1, 64, 7, 7}}},
                                         {// dynamic shapes
                                          {-1, 64, -1, {1, 200}},
                                          {// target static shapes
                                           {2, 64, 7, 7},
                                           {1, 64, 9, 9}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_2D_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes2d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

std::vector<InputShape> inputShapes2d_dynBatch = {{// dynamic shapes
                                                   {{1, 10}, 64, {7, 9}, {7, 9}},
                                                   {// target static shapes
                                                    {2, 64, 7, 7},
                                                    {1, 64, 9, 9},
                                                    {3, 64, 9, 9}}}};

INSTANTIATE_TEST_SUITE_P(nightly_GroupConv_2D_FP32_dynBatch,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes2d_dynBatch),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

std::vector<InputShape> inputShapes2d_cache = {{// dynamic shapes
                                                {-1, 64, -1, {1, 200}},
                                                {
                                                    // target static shapes
                                                    {1, 64, 7, 7},
                                                    {1, 64, 9, 9},
                                                    {1, 64, 7, 7},
                                                }}};

INSTANTIATE_TEST_SUITE_P(nightly_GroupConv_2D_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes2d_cache),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_2D_BF16,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes2d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D,
                                                                                        conv_avx512_2D_nspc})),
                                            ::testing::ValuesIn(fusingParamsSetBF16),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (3D) ============= */
const auto groupConvParams_ExplicitPadding_3D = ::testing::Combine(::testing::ValuesIn(kernels3d),
                                                                   ::testing::ValuesIn(strides3d),
                                                                   ::testing::ValuesIn(padBegins3d),
                                                                   ::testing::ValuesIn(padEnds3d),
                                                                   ::testing::ValuesIn(dilations3d),
                                                                   ::testing::ValuesIn(numOutChannels_Blocked),
                                                                   ::testing::ValuesIn(numGroups_Blocked),
                                                                   ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_3D = {
    //        conv_sse42_3D, // not supported jit_sse42 for 3d
    conv_avx2_3D,
    conv_avx512_3D,
    conv_avx2_3D_nspc,
    conv_avx512_3D_nspc};

std::vector<InputShape> inputShapes3d = {{{}, {{1, 64, 7, 7, 7}}},
                                         {// dynamic shapes
                                          {-1, 64, -1, {1, 200}, -1},
                                          {// target static shapes
                                           {2, 64, 7, 7, 7},
                                           {1, 64, 9, 9, 9}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_3D_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_3D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes3d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_3D_BF16,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_3D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes3d),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D,
                                                                                        conv_avx512_3D_nspc})),
                                            ::testing::ValuesIn(fusingParamsSetBF16),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (DW 1D) ============= */
const auto groupConvParams_ExplicitPadding_DW_1D = ::testing::Combine(::testing::ValuesIn(kernels1d),
                                                                      ::testing::ValuesIn(strides1d),
                                                                      ::testing::ValuesIn(padBegins1d),
                                                                      ::testing::ValuesIn(padEnds1d),
                                                                      ::testing::ValuesIn(dilations1d),
                                                                      ::testing::ValuesIn(numOutChannels_DW),
                                                                      ::testing::ValuesIn(numGroups_DW),
                                                                      ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_DW_1D = {conv_sse42_dw_1D,
                                                        conv_avx2_dw_1D,
                                                        conv_avx512_dw_1D,
                                                        conv_sse42_dw_1D_nspc,
                                                        conv_avx2_dw_1D_nspc,
                                                        conv_avx512_dw_1D_nspc};

std::vector<InputShape> inputShapes1dDW = {{{}, {{2, 32, 7}}},
                                           {// dynamic shapes
                                            {-1, 32, {1, 200}},
                                            {// target static shapes
                                             {2, 32, 7},
                                             {1, 32, 9}}}};

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConv_1D_DW_FP32,
    GroupConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(groupConvParams_ExplicitPadding_DW_1D,
                           ::testing::Values(ElementType::f32),
                           ::testing::Values(ElementType::undefined),
                           ::testing::Values(ElementType::undefined),
                           ::testing::ValuesIn(inputShapes1dDW),
                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(
            {conv_sse42_dw_1D, conv_avx2_dw_1D, conv_avx512_dw_1D})),  // todo: [AV] what about conv_sse42_dw_1D_nspc,
                                                                       //  conv_avx2_dw_1D_nspc, conv_avx512_dw_1D_nspc?
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(empty_plugin_config)),
    GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_1D_DW_BF16,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_DW_1D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes1dDW),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(
                                                {conv_avx512_dw_1D})),  // todo: [AV] what about conv_avx512_dw_1D_nspc?
                                            ::testing::ValuesIn(fusingParamsSetBF16),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (DW 2D) ============= */
const auto groupConvParams_ExplicitPadding_DW_2D = ::testing::Combine(::testing::ValuesIn(kernels2d),
                                                                      ::testing::ValuesIn(strides2d),
                                                                      ::testing::ValuesIn(padBegins2d),
                                                                      ::testing::ValuesIn(padEnds2d),
                                                                      ::testing::ValuesIn(dilations2d),
                                                                      ::testing::ValuesIn(numOutChannels_DW),
                                                                      ::testing::ValuesIn(numGroups_DW),
                                                                      ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_DW_2D = {conv_sse42_dw_2D,
                                                        conv_avx2_dw_2D,
                                                        conv_avx512_dw_2D,
                                                        conv_sse42_dw_2D_nspc,
                                                        conv_avx2_dw_2D_nspc,
                                                        conv_avx512_dw_2D_nspc};

std::vector<InputShape> inputShapes2dDW = {{{}, {{2, 32, 7, 7}}},
                                           {// dynamic shapes
                                            {-1, 32, -1, {1, 200}},
                                            {// target static shapes
                                             {2, 32, 7, 7},
                                             {1, 32, 9, 9}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_2D_DW_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_DW_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes2dDW),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_DW_2D)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_2D_DW_BF16,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_DW_2D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes2dDW),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_dw_2D,
                                                                                        conv_avx512_dw_2D_nspc})),
                                            ::testing::ValuesIn(fusingParamsSetBF16),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (DW 3D) ============= */
const auto groupConvParams_ExplicitPadding_DW_3D = ::testing::Combine(::testing::ValuesIn(kernels3d),
                                                                      ::testing::ValuesIn(strides3d),
                                                                      ::testing::ValuesIn(padBegins3d),
                                                                      ::testing::ValuesIn(padEnds3d),
                                                                      ::testing::ValuesIn(dilations3d),
                                                                      ::testing::ValuesIn(numOutChannels_DW),
                                                                      ::testing::ValuesIn(numGroups_DW),
                                                                      ::testing::Values(ov::op::PadType::EXPLICIT));

const std::vector<CPUSpecificParams> CPUParams_DW_3D = {conv_sse42_dw_3D,
                                                        conv_avx2_dw_3D,
                                                        conv_avx512_dw_3D,
                                                        conv_sse42_dw_3D_nspc,
                                                        conv_avx2_dw_3D_nspc,
                                                        conv_avx512_dw_3D_nspc};

std::vector<InputShape> inputShapes3dDW = {{{}, {{2, 32, 7, 7, 7}}},
                                           {// dynamic shapes
                                            {-1, 32, -1, {1, 200}, -1},
                                            {// target static shapes
                                             {2, 32, 7, 7, 7},
                                             {1, 32, 9, 9, 9}}}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConv_3D_DW_FP32,
                         GroupConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(groupConvParams_ExplicitPadding_DW_3D,
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::Values(ElementType::undefined),
                                                               ::testing::ValuesIn(inputShapes3dDW),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_DW_3D)),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::Values(empty_plugin_config)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);
/* ========= */

/* ============= SINGLE TEST CASES ============= */
using VecFusingParams = std::vector<fusingSpecificParams>;
using ConfigRelatedParams = std::tuple<ov::AnyMap, VecFusingParams>;  // Plugin config FusingParamsSet
using VecConfigRelatedParams = std::vector<ConfigRelatedParams>;

std::vector<groupConvLayerCPUTestParamsSet> makeSingleGroupConvCPUTestCases(
    std::vector<size_t> kernels,
    std::vector<size_t> strides,
    std::vector<size_t> dilations,
    std::vector<ptrdiff_t> padBegins,
    std::vector<ptrdiff_t> padEnds,
    ov::op::PadType padType,
    int groups,
    int mb,
    std::vector<size_t> spDims,
    int inGroupSize,
    int outGroupSize,
    const std::vector<CPUSpecificParams>& CPUParams,
    const VecConfigRelatedParams& vecConfigRelatedParams) {
    int inChannels = groups * inGroupSize;
    int outChannels = groups * outGroupSize;

    InputShape inputShapes;
    std::vector<size_t> targetShape;
    targetShape.push_back(mb);
    targetShape.push_back(inChannels);
    targetShape.insert(targetShape.end(), spDims.begin(), spDims.end());
    inputShapes.second.push_back({targetShape});

    groupConvSpecificParams
        specificParams(kernels, strides, padBegins, padEnds, dilations, outChannels, groups, padType);
    std::vector<groupConvLayerCPUTestParamsSet> retVector;

    for (auto& configRelatedParams : vecConfigRelatedParams) {
        VecFusingParams fusingParams;
        ov::AnyMap config;
        std::tie(config, fusingParams) = configRelatedParams;

        groupConvLayerTestsParamsSet basicParamsSet(specificParams,
                                                    ElementType::f32,
                                                    ElementType::undefined,
                                                    ElementType::undefined,
                                                    inputShapes,
                                                    ov::test::utils::DEVICE_CPU);

        for (auto& item : CPUParams) {
            for (auto& fusingParam : fusingParams) {
                retVector.push_back(groupConvLayerCPUTestParamsSet(basicParamsSet, item, fusingParam, config));
            }
        }
    }
    return retVector;
}

template <typename T>
void concatTestCases(std::vector<groupConvLayerCPUTestParamsSet>& resultVec, T tesCase) {
    resultVec.insert(resultVec.begin(),
                     std::make_move_iterator(tesCase.begin()),
                     std::make_move_iterator(tesCase.end()));
}

template <typename T, typename... Args>
void concatTestCases(std::vector<groupConvLayerCPUTestParamsSet>& resultVec, T&& tesCase, Args&&... args) {
    concatTestCases(resultVec, std::forward<T>(tesCase));
    concatTestCases(resultVec, std::forward<Args>(args)...);
}

template <typename... Args>
std::vector<groupConvLayerCPUTestParamsSet> generateSingleGroupConvCPUTestCases(Args&&... args) {
    std::vector<groupConvLayerCPUTestParamsSet> retVec;
    concatTestCases(retVec, std::forward<Args>(args)...);
    return retVec;
}

/* COMMON PARAMS */

const VecConfigRelatedParams vecPrcConnectParamsFP32 = {ConfigRelatedParams{empty_plugin_config, fusingParamsSet}};
const VecConfigRelatedParams vecPrcConnectParams = {ConfigRelatedParams{empty_plugin_config, fusingParamsSet},
                                                    ConfigRelatedParams{cpu_bf16_plugin_config, fusingParamsSetBF16}};
const VecConfigRelatedParams vecPrcConnectParamsBF16 = {
    ConfigRelatedParams{cpu_bf16_plugin_config, fusingParamsSetBF16}};

const VecConfigRelatedParams vecPrcConnectParamsFP32Default = {
    ConfigRelatedParams{empty_plugin_config, VecFusingParams{emptyFusingSpec}}};
const VecConfigRelatedParams vecPrcConnectParamsDefault = {
    ConfigRelatedParams{empty_plugin_config, VecFusingParams{emptyFusingSpec}},
    ConfigRelatedParams{cpu_bf16_plugin_config, VecFusingParams{emptyFusingSpec}}};

/* ============= GEMM GroupConvolution ============= */
const std::vector<groupConvLayerCPUTestParamsSet> gemmGroupConvTestCases = generateSingleGroupConvCPUTestCases(
    //  1. is_depthwise (true, false)
    //  2. jcp.im2col_sz (=0,>0)
    //  3. is_blocking_applicable (true, false)

    //  is_depthwise == false, im2col_sz > 0
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    2,
                                    2,
                                    CPUParams_Gemm_2D,
                                    vecPrcConnectParams),
    //  is_depthwise == true
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    1,
                                    1,
                                    CPUParams_Gemm_2D,
                                    vecPrcConnectParams),
    //  im2col_sz == 0, is_blocking_applicable == true
    makeSingleGroupConvCPUTestCases({1, 1},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    2,
                                    2,
                                    CPUParams_Gemm_2D,
                                    vecPrcConnectParams),
    //  is_blocking_applicable == false ((jcp.im2col_sz == 0) && (jcp.ic / jcp.oc >= 42))
    makeSingleGroupConvCPUTestCases({1, 1},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    42,
                                    1,
                                    CPUParams_Gemm_2D,
                                    vecPrcConnectParams),

    //  "hard" cases
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {2, 2},
                                    {1, 1},
                                    {1, 1},
                                    {1, 1},
                                    ov::op::PadType::EXPLICIT,
                                    3,
                                    2,
                                    {129, 129},
                                    4,
                                    2,
                                    CPUParams_Gemm_2D,
                                    vecPrcConnectParamsDefault),
    makeSingleGroupConvCPUTestCases({2, 4},
                                    {1, 2},
                                    {3, 2},
                                    {2, 1},
                                    {1, 0},
                                    ov::op::PadType::EXPLICIT,
                                    2,
                                    1,
                                    {10, 10},
                                    3,
                                    3,
                                    CPUParams_Gemm_2D,
                                    vecPrcConnectParamsDefault),
    makeSingleGroupConvCPUTestCases({3, 3, 3},
                                    {2, 2, 2},
                                    {1, 1, 1},
                                    {1, 1, 1},
                                    {1, 1, 1},
                                    ov::op::PadType::EXPLICIT,
                                    3,
                                    2,
                                    {33, 33, 33},
                                    4,
                                    2,
                                    CPUParams_Gemm_3D,
                                    vecPrcConnectParamsDefault),
    makeSingleGroupConvCPUTestCases({2, 3, 4},
                                    {1, 2, 2},
                                    {3, 1, 2},
                                    {2, 2, 1},
                                    {1, 1, 0},
                                    ov::op::PadType::EXPLICIT,
                                    2,
                                    1,
                                    {10, 10, 10},
                                    3,
                                    3,
                                    CPUParams_Gemm_3D,
                                    vecPrcConnectParams));

INSTANTIATE_TEST_SUITE_P(smoke_GEMM_GroupConv,
                         GroupConvolutionLayerCPUTest,
                         ::testing::ValuesIn(filterParamsSetForDevice(gemmGroupConvTestCases)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT SSE42 GroupConvolution ============= */
const std::vector<CPUSpecificParams> sse42_GroupConv = {conv_sse42_2D, conv_sse42_2D_nspc};
const std::vector<groupConvLayerCPUTestParamsSet> JIT_SSE42_GroupConvTestCases = generateSingleGroupConvCPUTestCases(
    //  1. jcp.ur_w (=3,<3)
    //  2. jcp.ur_w_tail (=0,>0)
    //  3. jcp.kw (>7,<=7)
    //  4. jcp.nb_oc = jcp.oc / jcp.oc_block;
    //  5. jcp.nb_ic = jcp.ic / jcp.ic_block;
    //  6. ocb_work

    //  jcp.ur_w == 3, jcp.ur_w_tail == 2
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 10},
                                    8,
                                    8,
                                    sse42_GroupConv,
                                    vecPrcConnectParamsFP32),
    //  jcp.ur_w < 3 (jcp.ur_w == jcp.ow)
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 4},
                                    8,
                                    8,
                                    sse42_GroupConv,
                                    vecPrcConnectParamsFP32),
    //  jcp.ur_w == 3, jcp.ur_w_tail == 0
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 11},
                                    8,
                                    8,
                                    sse42_GroupConv,
                                    vecPrcConnectParamsFP32),
    //  jcp.kw > 7
    makeSingleGroupConvCPUTestCases({3, 8},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 10},
                                    8,
                                    8,
                                    sse42_GroupConv,
                                    vecPrcConnectParamsFP32),
    //  jcp.nb_oc == 2
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    8,
                                    16,
                                    sse42_GroupConv,
                                    vecPrcConnectParamsFP32),
    //  jcp.nb_ic == 2
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    16,
                                    8,
                                    sse42_GroupConv,
                                    vecPrcConnectParamsFP32),
    //  ocb_work > 1 (ocb_work == 2)
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    8,
                                    40,
                                    sse42_GroupConv,
                                    vecPrcConnectParamsFP32),
    //  jcp.nb_ic == 2, ocb_work == 2
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    16,
                                    40,
                                    sse42_GroupConv,
                                    vecPrcConnectParamsFP32),

    //  "hard" cases
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {2, 2},
                                    {1, 1},
                                    {1, 1},
                                    {1, 1},
                                    ov::op::PadType::EXPLICIT,
                                    3,
                                    2,
                                    {129, 129},
                                    8,
                                    8,
                                    sse42_GroupConv,
                                    vecPrcConnectParamsFP32Default),
    makeSingleGroupConvCPUTestCases({2, 4},
                                    {1, 2},
                                    {3, 2},
                                    {2, 1},
                                    {1, 0},
                                    ov::op::PadType::EXPLICIT,
                                    2,
                                    1,
                                    {10, 10},
                                    8,
                                    8,
                                    sse42_GroupConv,
                                    vecPrcConnectParamsFP32Default)

    //  not supported jit_sse42 for 3d
    //  makeSingleGroupConvCPUTestCases({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
    //  ov::op::PadType::EXPLICIT,
    //                              3, 2, {33, 33, 33}, 8, 8, cpuParams_sse42_3D),
    //  makeSingleGroupConvCPUTestCases({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0},
    //  ov::op::PadType::EXPLICIT,
    //                              2, 1, {10, 10, 10}, 8, 8, cpuParams_sse42_3D),
);

INSTANTIATE_TEST_SUITE_P(smoke_JIT_SSE42_GroupConv,
                         GroupConvolutionLayerCPUTest,
                         ::testing::ValuesIn(filterParamsSetForDevice(JIT_SSE42_GroupConvTestCases)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT AVX2 GroupConvolution ============= */
const std::vector<CPUSpecificParams> avx2_GroupConv_2D = {conv_avx2_2D, conv_avx2_2D_nspc};
const std::vector<CPUSpecificParams> avx2_GroupConv_3D = {conv_avx2_3D, conv_avx2_3D_nspc};
const std::vector<groupConvLayerCPUTestParamsSet> JIT_AVX2_GroupConvTestCases = generateSingleGroupConvCPUTestCases(
    //  1. jcp.ur_w (=3,<3)
    //  2. jcp.ur_w_tail (=0,>0)
    //  3. jcp.kw (>7,<=7)
    //  4. jcp.nb_oc = jcp.oc / jcp.oc_block;
    //  5. jcp.nb_ic = jcp.ic / jcp.ic_block;
    //  6. ocb_work

    //  jcp.ur_w == 3, jcp.ur_w_tail == 2
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 10},
                                    8,
                                    8,
                                    avx2_GroupConv_2D,
                                    vecPrcConnectParamsFP32),
    //  jcp.ur_w < 3 (jcp.ur_w == jcp.ow)
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 4},
                                    8,
                                    8,
                                    avx2_GroupConv_2D,
                                    vecPrcConnectParamsFP32),
    //  jcp.ur_w == 3, jcp.ur_w_tail == 0
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 11},
                                    8,
                                    8,
                                    avx2_GroupConv_2D,
                                    vecPrcConnectParamsFP32),
    //  jcp.kw > 7
    makeSingleGroupConvCPUTestCases({3, 8},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 10},
                                    8,
                                    8,
                                    avx2_GroupConv_2D,
                                    vecPrcConnectParamsFP32),
    //  jcp.nb_oc == 2
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    8,
                                    16,
                                    avx2_GroupConv_2D,
                                    vecPrcConnectParamsFP32),
    //  jcp.nb_ic == 2
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    16,
                                    8,
                                    avx2_GroupConv_2D,
                                    vecPrcConnectParamsFP32),
    //  ocb_work > 1 (ocb_work == 2)
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    8,
                                    40,
                                    avx2_GroupConv_2D,
                                    vecPrcConnectParamsFP32),
    //  jcp.nb_ic == 2, ocb_work == 2
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    16,
                                    40,
                                    avx2_GroupConv_2D,
                                    vecPrcConnectParamsFP32),

    //  "hard" cases
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {2, 2},
                                    {1, 1},
                                    {1, 1},
                                    {1, 1},
                                    ov::op::PadType::EXPLICIT,
                                    3,
                                    2,
                                    {129, 129},
                                    8,
                                    8,
                                    avx2_GroupConv_2D,
                                    vecPrcConnectParamsFP32Default),
    makeSingleGroupConvCPUTestCases({2, 4},
                                    {1, 2},
                                    {3, 2},
                                    {2, 1},
                                    {1, 0},
                                    ov::op::PadType::EXPLICIT,
                                    2,
                                    1,
                                    {10, 10},
                                    8,
                                    8,
                                    avx2_GroupConv_2D,
                                    vecPrcConnectParamsFP32Default),
    makeSingleGroupConvCPUTestCases({3, 3, 3},
                                    {2, 2, 2},
                                    {1, 1, 1},
                                    {1, 1, 1},
                                    {1, 1, 1},
                                    ov::op::PadType::EXPLICIT,
                                    3,
                                    2,
                                    {33, 33, 33},
                                    8,
                                    8,
                                    avx2_GroupConv_3D,
                                    vecPrcConnectParamsFP32Default),
    makeSingleGroupConvCPUTestCases({2, 3, 4},
                                    {1, 2, 2},
                                    {3, 1, 2},
                                    {2, 2, 1},
                                    {1, 1, 0},
                                    ov::op::PadType::EXPLICIT,
                                    2,
                                    1,
                                    {10, 10, 10},
                                    8,
                                    8,
                                    avx2_GroupConv_3D,
                                    vecPrcConnectParamsFP32));

INSTANTIATE_TEST_SUITE_P(smoke_JIT_AVX2_GroupConv,
                         GroupConvolutionLayerCPUTest,
                         ::testing::ValuesIn(filterParamsSetForDevice(JIT_AVX2_GroupConvTestCases)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT AVX512 GroupConvolution ============= */
const std::vector<CPUSpecificParams> avx512_GroupConv_2D = {conv_avx512_2D, conv_avx512_2D_nspc};
const std::vector<CPUSpecificParams> avx512_GroupConv_3D = {conv_avx512_3D, conv_avx512_3D_nspc};
const std::vector<groupConvLayerCPUTestParamsSet> JIT_AVX512_GroupConvTestCases = generateSingleGroupConvCPUTestCases(
    //  1. "blocked to blocked" or "planar to blocked"
    //  2. jcp.nb_ic, jcp.nb_oc

    //  blocked to blocked
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    16,
                                    16,
                                    avx512_GroupConv_2D,
                                    vecPrcConnectParams),
    //  jcp.nb_ic == 2
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    32,
                                    16,
                                    avx512_GroupConv_2D,
                                    vecPrcConnectParams),
    //  jcp.nb_oc == 2
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    2,
                                    1,
                                    {5, 5},
                                    16,
                                    32,
                                    avx512_GroupConv_2D,
                                    vecPrcConnectParams),

    //  "hard" cases
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {2, 2},
                                    {1, 1},
                                    {1, 1},
                                    {1, 1},
                                    ov::op::PadType::EXPLICIT,
                                    3,
                                    2,
                                    {129, 129},
                                    16,
                                    16,
                                    avx512_GroupConv_2D,
                                    vecPrcConnectParams),
    makeSingleGroupConvCPUTestCases({2, 4},
                                    {1, 2},
                                    {3, 2},
                                    {2, 1},
                                    {1, 0},
                                    ov::op::PadType::EXPLICIT,
                                    2,
                                    1,
                                    {10, 10},
                                    16,
                                    16,
                                    avx512_GroupConv_2D,
                                    vecPrcConnectParamsDefault),
    makeSingleGroupConvCPUTestCases({3, 3, 3},
                                    {2, 2, 2},
                                    {1, 1, 1},
                                    {1, 1, 1},
                                    {1, 1, 1},
                                    ov::op::PadType::EXPLICIT,
                                    3,
                                    2,
                                    {33, 33, 33},
                                    16,
                                    16,
                                    avx512_GroupConv_3D,
                                    vecPrcConnectParamsDefault),
    makeSingleGroupConvCPUTestCases({2, 3, 4},
                                    {1, 2, 2},
                                    {3, 1, 2},
                                    {2, 2, 1},
                                    {1, 1, 0},
                                    ov::op::PadType::EXPLICIT,
                                    2,
                                    1,
                                    {10, 10, 10},
                                    16,
                                    16,
                                    avx512_GroupConv_3D,
                                    vecPrcConnectParams));

INSTANTIATE_TEST_SUITE_P(smoke_JIT_AVX512_GroupConv,
                         GroupConvolutionLayerCPUTest,
                         ::testing::ValuesIn(filterParamsSetForDevice(JIT_AVX512_GroupConvTestCases)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT SSE42 DW GroupConvolution ============= */
const std::vector<CPUSpecificParams> sse42_DW_2D = {conv_sse42_dw_2D, conv_sse42_dw_2D_nspc};
const std::vector<CPUSpecificParams> sse42_DW_3D = {conv_sse42_dw_3D, conv_sse42_dw_3D_nspc};
const std::vector<groupConvLayerCPUTestParamsSet> JIT_SSE42_DW_GroupConvTestCases = generateSingleGroupConvCPUTestCases(
    //  1. jcp.ngroups % simd_w (=0,!=0)
    //  2. jcp.nb_ch
    //  3. jcp.nb_ch_blocking (=2,<2)
    //  4. jcp.ur_w == 3

    //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 1, jcp.nb_ch_blocking == 1 (jcp.ngroups == 8)
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    8,
                                    1,
                                    {5, 5},
                                    1,
                                    1,
                                    sse42_DW_2D,
                                    vecPrcConnectParamsFP32),
    //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 2, jcp.nb_ch_blocking == 2 (jcp.ngroups == 16)
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    16,
                                    1,
                                    {5, 5},
                                    1,
                                    1,
                                    sse42_DW_2D,
                                    vecPrcConnectParamsFP32),
    //  jcp.ngroups % simd_w != 0, jcp.nb_ch == 3, jcp.nb_ch_blocking == 2 (jcp.ngroups == 17) TODO: pad channels not
    //  supported for SSE42 makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0},
    //  ov::op::PadType::VALID, 17, 1, {5, 5}, 1, 1, conv_sse42_DW_2D, vecPrcConnectParamsFP32only), jcp.ow > jcp.ur_w
    //  (jcp.ow == 7)
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    8,
                                    1,
                                    {5, 9},
                                    1,
                                    1,
                                    sse42_DW_2D,
                                    vecPrcConnectParamsFP32),

    //  "hard" cases
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {2, 2},
                                    {1, 1},
                                    {1, 1},
                                    {1, 1},
                                    ov::op::PadType::EXPLICIT,
                                    8,
                                    2,
                                    {129, 129},
                                    1,
                                    1,
                                    sse42_DW_2D,
                                    vecPrcConnectParamsFP32),
    makeSingleGroupConvCPUTestCases({2, 4},
                                    {1, 2},
                                    {3, 2},
                                    {2, 1},
                                    {1, 0},
                                    ov::op::PadType::EXPLICIT,
                                    8,
                                    1,
                                    {10, 10},
                                    1,
                                    1,
                                    sse42_DW_2D,
                                    vecPrcConnectParamsFP32Default),
    makeSingleGroupConvCPUTestCases({3, 3, 3},
                                    {2, 2, 2},
                                    {1, 1, 1},
                                    {1, 1, 1},
                                    {1, 1, 1},
                                    ov::op::PadType::EXPLICIT,
                                    8,
                                    2,
                                    {33, 33, 33},
                                    1,
                                    1,
                                    sse42_DW_3D,
                                    vecPrcConnectParamsFP32Default),
    makeSingleGroupConvCPUTestCases({2, 3, 4},
                                    {1, 2, 2},
                                    {3, 1, 2},
                                    {2, 2, 1},
                                    {1, 1, 0},
                                    ov::op::PadType::EXPLICIT,
                                    8,
                                    1,
                                    {10, 10, 10},
                                    1,
                                    1,
                                    sse42_DW_3D,
                                    vecPrcConnectParamsFP32));

INSTANTIATE_TEST_SUITE_P(smoke_JIT_SSE42_DW_GroupConv,
                         GroupConvolutionLayerCPUTest,
                         ::testing::ValuesIn(filterParamsSetForDevice(JIT_SSE42_DW_GroupConvTestCases)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT AVX2 DW GroupConvolution ============= */
const std::vector<CPUSpecificParams> avx2_DW_2D = {conv_avx2_dw_2D, conv_avx2_dw_2D_nspc};
const std::vector<CPUSpecificParams> avx2_DW_3D = {conv_avx2_dw_3D, conv_avx2_dw_3D_nspc};
const std::vector<groupConvLayerCPUTestParamsSet> JIT_AVX2_DW_GroupConvTestCases = generateSingleGroupConvCPUTestCases(
    //  1. jcp.ngroups % simd_w (=0,!=0)
    //  2. jcp.nb_ch
    //  3. jcp.nb_ch_blocking (=3,<3)
    //  4. jcp.ur_w == 4

    //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 1, jcp.nb_ch_blocking == 1 (jcp.ngroups == 8)
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    8,
                                    1,
                                    {5, 5},
                                    1,
                                    1,
                                    avx2_DW_2D,
                                    vecPrcConnectParamsFP32),
    //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 3, jcp.nb_ch_blocking == 3 (jcp.ngroups == 24)
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    24,
                                    1,
                                    {5, 5},
                                    1,
                                    1,
                                    avx2_DW_2D,
                                    vecPrcConnectParamsFP32),
    //  jcp.ngroups % simd_w != 0, jcp.nb_ch == 4, jcp.nb_ch_blocking == 3 (jcp.ngroups == 25)
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    25,
                                    1,
                                    {5, 5},
                                    1,
                                    1,
                                    avx2_DW_2D,
                                    vecPrcConnectParamsFP32),
    //  jcp.ow > jcp.ur_w (jcp.ow == 7)
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {1, 1},
                                    {1, 1},
                                    {0, 0},
                                    {0, 0},
                                    ov::op::PadType::VALID,
                                    8,
                                    1,
                                    {5, 9},
                                    1,
                                    1,
                                    avx2_DW_2D,
                                    vecPrcConnectParamsFP32),

    //  "hard" cases
    makeSingleGroupConvCPUTestCases({3, 3},
                                    {2, 2},
                                    {1, 1},
                                    {1, 1},
                                    {1, 1},
                                    ov::op::PadType::EXPLICIT,
                                    8,
                                    2,
                                    {129, 129},
                                    1,
                                    1,
                                    avx2_DW_2D,
                                    vecPrcConnectParamsFP32Default),
    makeSingleGroupConvCPUTestCases({2, 4},
                                    {1, 2},
                                    {3, 2},
                                    {2, 1},
                                    {1, 0},
                                    ov::op::PadType::EXPLICIT,
                                    8,
                                    1,
                                    {10, 10},
                                    1,
                                    1,
                                    avx2_DW_2D,
                                    vecPrcConnectParamsFP32Default),
    makeSingleGroupConvCPUTestCases({3, 3, 3},
                                    {2, 2, 2},
                                    {1, 1, 1},
                                    {1, 1, 1},
                                    {1, 1, 1},
                                    ov::op::PadType::EXPLICIT,
                                    8,
                                    2,
                                    {33, 33, 33},
                                    1,
                                    1,
                                    avx2_DW_3D,
                                    vecPrcConnectParamsFP32Default),
    makeSingleGroupConvCPUTestCases({2, 3, 4},
                                    {1, 2, 2},
                                    {3, 1, 2},
                                    {2, 2, 1},
                                    {1, 1, 0},
                                    ov::op::PadType::EXPLICIT,
                                    8,
                                    1,
                                    {10, 10, 10},
                                    1,
                                    1,
                                    avx2_DW_3D,
                                    vecPrcConnectParamsFP32));

INSTANTIATE_TEST_SUITE_P(smoke_JIT_AVX2_DW_GroupConv,
                         GroupConvolutionLayerCPUTest,
                         ::testing::ValuesIn(filterParamsSetForDevice(JIT_AVX2_DW_GroupConvTestCases)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT AVX512 DW GroupConvolution ============= */
const std::vector<CPUSpecificParams> avx512_DW_2D = {conv_avx512_dw_2D, conv_avx512_dw_2D_nspc};
const std::vector<CPUSpecificParams> avx512_DW_3D = {conv_avx512_dw_3D, conv_avx512_dw_3D_nspc};
const std::vector<groupConvLayerCPUTestParamsSet> JIT_AVX512_DW_GroupConvTestCases =
    generateSingleGroupConvCPUTestCases(
        //  1. jcp.ngroups % simd_w (=0,!=0)
        //  2. jcp.nb_ch
        //  3. jcp.nb_ch_blocking (=4,<4)
        //  4. jcp.ur_w == 6

        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 1, jcp.nb_ch_blocking == 1 (jcp.ngroups == 16)
        makeSingleGroupConvCPUTestCases({3, 3},
                                        {1, 1},
                                        {1, 1},
                                        {0, 0},
                                        {0, 0},
                                        ov::op::PadType::VALID,
                                        16,
                                        1,
                                        {5, 5},
                                        1,
                                        1,
                                        avx512_DW_2D,
                                        vecPrcConnectParams),
        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 4, jcp.nb_ch_blocking == 4 (jcp.ngroups == 64)
        makeSingleGroupConvCPUTestCases({3, 3},
                                        {1, 1},
                                        {1, 1},
                                        {0, 0},
                                        {0, 0},
                                        ov::op::PadType::VALID,
                                        64,
                                        1,
                                        {5, 5},
                                        1,
                                        1,
                                        avx512_DW_2D,
                                        vecPrcConnectParams),
        //  jcp.ngroups % simd_w != 0, jcp.nb_ch == 5, jcp.nb_ch_blocking == 4 (jcp.ngroups == 65)
        makeSingleGroupConvCPUTestCases({3, 3},
                                        {1, 1},
                                        {1, 1},
                                        {0, 0},
                                        {0, 0},
                                        ov::op::PadType::VALID,
                                        65,
                                        1,
                                        {5, 5},
                                        1,
                                        1,
                                        avx512_DW_2D,
                                        vecPrcConnectParams),
        //  jcp.ow > jcp.ur_w (jcp.ow == 7)
        makeSingleGroupConvCPUTestCases({3, 3},
                                        {1, 1},
                                        {1, 1},
                                        {0, 0},
                                        {0, 0},
                                        ov::op::PadType::VALID,
                                        8,
                                        1,
                                        {5, 9},
                                        1,
                                        1,
                                        avx512_DW_2D,
                                        vecPrcConnectParams),
        //  "hard" cases
        makeSingleGroupConvCPUTestCases({3, 3},
                                        {2, 2},
                                        {1, 1},
                                        {1, 1},
                                        {1, 1},
                                        ov::op::PadType::EXPLICIT,
                                        16,
                                        2,
                                        {129, 129},
                                        1,
                                        1,
                                        avx512_DW_2D,
                                        vecPrcConnectParamsDefault),
        makeSingleGroupConvCPUTestCases({2, 4},
                                        {1, 2},
                                        {3, 2},
                                        {2, 1},
                                        {1, 0},
                                        ov::op::PadType::EXPLICIT,
                                        16,
                                        1,
                                        {10, 10},
                                        1,
                                        1,
                                        avx512_DW_2D,
                                        vecPrcConnectParamsDefault),
        makeSingleGroupConvCPUTestCases({3, 3, 3},
                                        {2, 2, 2},
                                        {1, 1, 1},
                                        {1, 1, 1},
                                        {1, 1, 1},
                                        ov::op::PadType::EXPLICIT,
                                        16,
                                        2,
                                        {33, 33, 33},
                                        1,
                                        1,
                                        avx512_DW_3D,
                                        vecPrcConnectParamsDefault),
        makeSingleGroupConvCPUTestCases({2, 3, 4},
                                        {1, 2, 2},
                                        {3, 1, 2},
                                        {2, 2, 1},
                                        {1, 1, 0},
                                        ov::op::PadType::EXPLICIT,
                                        16,
                                        1,
                                        {10, 10, 10},
                                        1,
                                        1,
                                        avx512_DW_3D,
                                        vecPrcConnectParams));

INSTANTIATE_TEST_SUITE_P(smoke_JIT_AVX512_DW_GroupConv,
                         GroupConvolutionLayerCPUTest,
                         ::testing::ValuesIn(filterParamsSetForDevice(JIT_AVX512_DW_GroupConvTestCases)),
                         GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT SSE42 1x1 Convolution (not supported with groups) ============= */
/* ============= JIT AVX2 1x1 Convolution (not supported with groups) ============= */
/* ============= JIT AVX512 1x1 Convolution (not supported with groups) ============= */
/* ============= JIT AVX2 PLANAR Convolution (not supported with groups) ============= */
/* ============= JIT AVX5122 PLANAR Convolution (not supported with groups) ============= */
/* ============================================= */

/* ============= brgemm GroupConvolution test, expect fallback to other implementation ============= */
const std::vector<CPUSpecificParams> CPUParams_Fallback_Brgemm_2D = {
    CPUSpecificParams{{nhwc}, {nhwc}, {/* non-brgconv_avx512 is expected */}, "brgconv_avx512"},
    CPUSpecificParams{{nhwc}, {nhwc}, {/* non-brgconv_avx512_amx is expected */}, "brgconv_avx512_amx"},
};
const std::vector<CPUSpecificParams> CPUParams_Fallback_Brgemm_1D_Small_Shape = {
    CPUSpecificParams{{nwc}, {nwc}, {/* non-brgconv_avx512_amx is expected */}, "brgconv_avx512_amx"}};
const std::vector<groupConvLayerCPUTestParamsSet> BRGEMM_EXPECT_FALLBACK_GroupConvTestCases =
    generateSingleGroupConvCPUTestCases(
        // channel <= 16
        // https://github.com/openvinotoolkit/oneDNN/blob/6df930dab5ab0a7dfaea6100acd03b479e2fa0a8/src/cpu/x64/jit_brgemm_conv_utils.cpp#L1712
        makeSingleGroupConvCPUTestCases({3, 3},
                                        {1, 1},
                                        {1, 1},
                                        {0, 0},
                                        {0, 0},
                                        ov::op::PadType::EXPLICIT,
                                        4,
                                        1,
                                        {5, 5},
                                        16,
                                        16,
                                        CPUParams_Fallback_Brgemm_2D,
                                        vecPrcConnectParamsFP32),
        // small shape on amx
        //  https://github.com/openvinotoolkit/oneDNN/blob/6df930dab5ab0a7dfaea6100acd03b479e2fa0a8/src/cpu/x64/jit_brgemm_conv_utils.cpp#L1719
        makeSingleGroupConvCPUTestCases({3},
                                        {1},
                                        {1},
                                        {0},
                                        {0},
                                        ov::op::PadType::EXPLICIT,
                                        4,
                                        1,
                                        {3},
                                        32,
                                        32,
                                        CPUParams_Fallback_Brgemm_1D_Small_Shape,
                                        vecPrcConnectParamsBF16));

INSTANTIATE_TEST_SUITE_P(smoke_BRGEMM_EXPECT_FALLBACK_GroupConv,
                         ExpectFallbackGroupConvolutionLayerCPUTest,
                         ::testing::ValuesIn(filterParamsSetForDevice(BRGEMM_EXPECT_FALLBACK_GroupConvTestCases)),
                         ExpectFallbackGroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= brgemm GroupConvolution test, expect fallback to other implementation, end ============= */

}  // namespace

}  // namespace test
}  // namespace ov
