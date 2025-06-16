// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/group_convolution_backprop_data.hpp"

#include "common_test_utils/node_builders/group_convolution_backprop_data.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/convolution_params.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {

using GroupDeconvSpecParams = ov::test::groupConvBackpropSpecificParams;

using DeconvInputData = std::tuple<InputShape,                          // data shape
                                   ov::test::utils::InputLayerType,     // 'output_shape' input type
                                   std::vector<std::vector<int32_t>>>;  // values for 'output_shape'

using GroupDeconvLayerCPUTestParamsSet = std::
    tuple<GroupDeconvSpecParams, DeconvInputData, ElementType, fusingSpecificParams, CPUSpecificParams, ov::AnyMap>;

class GroupDeconvolutionLayerCPUTest : public testing::WithParamInterface<GroupDeconvLayerCPUTestParamsSet>,
                                       virtual public SubgraphBaseTest,
                                       public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GroupDeconvLayerCPUTestParamsSet> obj) {
        GroupDeconvSpecParams basicParamsSet;
        DeconvInputData inputData;
        ElementType prec;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        ov::AnyMap additionalConfig;
        std::tie(basicParamsSet, inputData, prec, fusingParams, cpuParams, additionalConfig) = obj.param;

        ov::op::PadType padType;
        std::vector<size_t> kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
        size_t convOutChannels, groupNum;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, groupNum, padType, outPadding) =
            basicParamsSet;

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
        result << "PRC=" << prec << "_";
        result << "K=" << ov::test::utils::vec2str(kernel) << "_";
        result << "S=" << ov::test::utils::vec2str(stride) << "_";
        result << "PB=" << ov::test::utils::vec2str(padBegin) << "_";
        result << "PE=" << ov::test::utils::vec2str(padEnd) << "_";
        result << "D=" << ov::test::utils::vec2str(dilation) << "_";
        result << "OP=" << ov::test::utils::vec2str(outPadding) << "_";
        result << "O=" << convOutChannels << "_";
        result << "G=" << groupNum << "_";
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
                result << "_" << item.first << "=" << item.second.as<std::string>();
            }
        }

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        if (function->get_parameters().size() != 1) {
            // WA: output_shape depends on 3rd deconvolution input data
            // but the reference implementation doesn't implement shape inference
            // so we need to build a new function and replace the 3rd input parameter with a constant
            // to get valid output shapes
            functionRefs = createGraph({targetInputStaticShapes[0]}, ov::test::utils::InputLayerType::CONSTANT);
        }
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 1) {
                tensor = ov::Tensor(funcInput.get_element_type(),
                                    targetInputStaticShapes[i],
                                    outShapeData[inferRequestNum].data());
            } else {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = 2560;
                in_data.resolution = 256;

                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                 targetInputStaticShapes[i],
                                                                 in_data);
            }

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
        inferRequestNum++;
    }

    void validate() override {
        auto actualOutputs = get_plugin_outputs();
        if (function->get_parameters().size() == 2) {
            auto pos = std::find_if(inputs.begin(),
                                    inputs.end(),
                                    [](const std::pair<std::shared_ptr<ov::Node>, ov::Tensor>& params) {
                                        return params.first->get_friendly_name() == "param_1";
                                    });
            OPENVINO_ASSERT(pos != inputs.end());
            inputs.erase(pos);
        }
        auto expectedOutputs = calculate_refs();
        if (expectedOutputs.empty()) {
            return;
        }
        ASSERT_EQ(actualOutputs.size(), expectedOutputs.size())
            << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while OV " << actualOutputs.size();

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
                if (inType != ov::element::Type_t::dynamic) {
                    p.input(i).tensor().set_element_type(inType);
                }
            }
        }
        {
            auto results = function->get_results();
            for (size_t i = 0; i < results.size(); i++) {
                if (outType != ov::element::Type_t::dynamic) {
                    p.output(i).tensor().set_element_type(outType);
                }
            }
        }
        function = p.build();
    }

    std::shared_ptr<ov::Model> createGraph(const std::vector<ov::PartialShape>& inShapes,
                                           ov::test::utils::InputLayerType outShapeType) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(prec, inShapes.front())};
        std::shared_ptr<ov::Node> outShapeNode;
        if (!outShapeData.empty()) {
            if (outShapeType == ov::test::utils::InputLayerType::PARAMETER) {
                OPENVINO_ASSERT(inputDynamicShapes.size() == 2);
                auto outShapeParam =
                    std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes.back());
                params.push_back(outShapeParam);
                outShapeNode = outShapeParam;
            } else {
                outShapeNode = ov::op::v0::Constant::create(ov::element::i32,
                                                            {outShapeData[inferRequestNum].size()},
                                                            outShapeData[inferRequestNum]);
            }
        }

        for (size_t i = 0; i < params.size(); i++) {
            params[i]->set_friendly_name(std::string("param_") + std::to_string(i));
        }

        std::shared_ptr<ov::Node> deconv;
        if (!outShapeData.empty()) {
            OPENVINO_ASSERT(outShapeNode != nullptr);
            deconv = ov::test::utils::make_group_convolution_backprop_data(params[0],
                                                                           outShapeNode,
                                                                           prec,
                                                                           kernel,
                                                                           stride,
                                                                           padBegin,
                                                                           padEnd,
                                                                           dilation,
                                                                           padType,
                                                                           convOutChannels,
                                                                           groupNum);
        } else {
            deconv = ov::test::utils::make_group_convolution_backprop_data(params[0],
                                                                           prec,
                                                                           kernel,
                                                                           stride,
                                                                           padBegin,
                                                                           padEnd,
                                                                           dilation,
                                                                           padType,
                                                                           convOutChannels,
                                                                           groupNum,
                                                                           false,
                                                                           outPadding);
        }

        return makeNgraphFunction(prec, params, deconv, "GroupDeconvCPU");
    }

protected:
    std::vector<size_t> kernel, stride;

    void SetUp() override {
        rel_threshold = 1e-4f;

        targetDevice = ov::test::utils::DEVICE_CPU;

        GroupDeconvSpecParams basicParamsSet;
        DeconvInputData inputData;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        ov::AnyMap additionalConfig;
        std::tie(basicParamsSet, inputData, prec, fusingParams, cpuParams, additionalConfig) = this->GetParam();

        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, groupNum, padType, outPadding) =
            basicParamsSet;

        InputShape inputShape;
        ov::test::utils::InputLayerType outShapeType;
        std::tie(inputShape, outShapeType, outShapeData) = inputData;

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        if (additionalConfig[ov::hint::inference_precision.name()] == ov::element::bf16) {
            inType = outType = prec = ElementType::bf16;
            rel_threshold = 1e-2f;
        } else {
            inType = outType = prec;
        }

        selectedType = makeSelectedTypeStr(selectedType, prec);

        std::vector<InputShape> paramsShapes;
        paramsShapes.push_back(inputShape);
        if (!outShapeData.empty() && outShapeType == ov::test::utils::InputLayerType::PARAMETER) {
            const auto outShapeDims = ov::Shape{outShapeData.front().size()};
            paramsShapes.push_back(
                InputShape{outShapeDims, std::vector<ov::Shape>(inputShape.second.size(), outShapeDims)});
        }

        init_input_shapes(paramsShapes);

        function = createGraph(inputDynamicShapes, outShapeType);
    }

private:
    ElementType prec;
    ov::op::PadType padType;
    std::vector<size_t> dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels, groupNum;
    std::vector<std::vector<int32_t>> outShapeData;
    size_t inferRequestNum = 0;
};

TEST_P(GroupDeconvolutionLayerCPUTest, CompareWithRefs) {
    if (!fusedOps.empty()) {
        bool isSupportedParams = stride[stride.size() - 1] <= kernel[kernel.size() - 1];
        if (stride.size() > 1)
            isSupportedParams &= stride[stride.size() - 2] <= kernel[kernel.size() - 2];
        if (stride.size() > 2)
            isSupportedParams &= stride[stride.size() - 3] <= kernel[kernel.size() - 3];
        if (!isSupportedParams) {
            GTEST_SKIP() << "Fusing with strides more than kernel size was disabled, because oneDNN deconvolution "
                            "doesn't support it"
                         << std::endl;
        }
    }

    run();
    CheckPluginRelatedResults(compiledModel, "Deconvolution");
}

namespace {
/* COMMON PARAMS */
std::vector<fusingSpecificParams> fusingParamsSet{
    emptyFusingSpec,
    //bias fusing
    fusingAddPerChannel,
};

std::vector<fusingSpecificParams> fusingParamsSetBrg{
    emptyFusingSpec,
    // Bias fusing
    fusingAddPerChannel,
    fusingMultiplyPerChannel
};

const std::vector<std::vector<size_t>> emptyOutputShape = {{}};
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = {{}};

/* ============= GroupConvolution params (planar layout) ============= */
const std::vector<size_t> numOutChannels_Planar = {6};
const std::vector<size_t> numGroups_Planar = {2, 3};

/* ============= GroupConvolution params (blocked layout) ============= */
const std::vector<size_t> numOutChannels_Blocked = {64};
const std::vector<size_t> numGroups_Blocked = {2, 4};

/* ============= GroupConvolution params (nspc layout) ============= */
const std::vector<size_t> numOutChannels_nspc = {64};
const std::vector<size_t> numGroups_nspc = {2};

/* ============= GroupConvolution params (DW) ============= */
const std::vector<size_t> numOutChannels_DW = {32};
const std::vector<size_t> numGroups_DW = {32};

/* ============= GroupConvolution params (2D) ============= */
const std::vector<std::vector<size_t>> kernels2d = {{3, 3}, {1, 1}};
const std::vector<std::vector<size_t>> brgKernels2d = {{3, 3}, {2, 2}};
const std::vector<std::vector<size_t>> strides2d = {{1, 1}, {2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins2d = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2d = {{0, 0}};
const std::vector<std::vector<size_t>> dilations2d = {{1, 1}};

/* ============= GroupConvolution params (3D) ============= */
const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}, {1, 1, 1}};
const std::vector<std::vector<size_t>> brgKernels3d = {{3, 3, 3}, {2, 2, 2}};
const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}, {2, 2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins3d = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3d = {{0, 0, 0}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}};
/* ============= */

/* INSTANCES */
/* ============= GroupConvolution (Planar 2D) ============= */
const std::vector<DeconvInputData> Planar_2D_inputs_nightly = {
    DeconvInputData{InputShape{{-1, 12, -1, -1}, {{2, 12, 7, 7}, {2, 12, 5, 7}, {1, 12, 9, 4}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {}},
    DeconvInputData{InputShape{{-1, 12, -1, -1}, {{2, 12, 7, 7}, {2, 12, 5, 7}, {1, 12, 9, 4}, {2, 12, 5, 7}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {{15, 15}}},
    DeconvInputData{InputShape{{{1, 10}, 12, 7, 7}, {{1, 12, 7, 7}, {3, 12, 7, 7}, {2, 12, 7, 7}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {{15, 15}}},
    DeconvInputData{InputShape{{}, {{2, 12, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
    DeconvInputData{InputShape{{-1, 12, -1, -1}, {{1, 12, 7, 7}, {2, 12, 5, 7}, {1, 12, 7, 7}}},
                    ov::test::utils::InputLayerType::PARAMETER,
                    {{15, 15}, {9, 10}, {15, 15}}}};

const auto groupConvParams_ExplicitPadding_Planar_2D = ::testing::Combine(::testing::ValuesIn(kernels2d),
                                                                          ::testing::ValuesIn(strides2d),
                                                                          ::testing::ValuesIn(padBegins2d),
                                                                          ::testing::ValuesIn(padEnds2d),
                                                                          ::testing::ValuesIn(dilations2d),
                                                                          ::testing::ValuesIn(numOutChannels_Planar),
                                                                          ::testing::ValuesIn(numGroups_Planar),
                                                                          ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                          ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(nightly_GroupDeconv_2D_Planar_FP32,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_Planar_2D,
                                            ::testing::ValuesIn(Planar_2D_inputs_nightly),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({planar_2D})),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_GroupDeconv_2D_Planar_BF16,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_Planar_2D,
                                            ::testing::ValuesIn(Planar_2D_inputs_nightly),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({planar_2D})),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);

const std::vector<DeconvInputData> Planar_3D_inputs_nightly = {
    DeconvInputData{
        InputShape{{-1, 12, -1, -1, -1}, {{2, 12, 7, 7, 7}, {2, 12, 5, 7, 7}, {1, 12, 9, 4, 9}, {2, 12, 5, 7, 7}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {}},
    DeconvInputData{InputShape{{-1, 12, -1, -1, -1}, {{2, 12, 7, 7, 7}, {2, 12, 5, 7, 7}, {1, 12, 9, 4, 9}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {{15, 15, 15}}},
    DeconvInputData{InputShape{{{1, 10}, 12, 7, 7, 7}, {{3, 12, 7, 7, 7}, {2, 12, 7, 7, 7}, {1, 12, 7, 7, 7}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {{15, 15, 15}}},
    DeconvInputData{InputShape{{}, {{2, 12, 7, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
    DeconvInputData{InputShape{{-1, 12, -1, -1, -1}, {{2, 12, 7, 7, 7}, {2, 12, 5, 7, 7}, {1, 12, 9, 4, 9}}},
                    ov::test::utils::InputLayerType::PARAMETER,
                    {{15, 15, 15}, {9, 10, 10}, {9, 9, 9}}}};

const auto groupConvParams_ExplicitPadding_Planar_3D = ::testing::Combine(::testing::ValuesIn(kernels3d),
                                                                          ::testing::ValuesIn(strides3d),
                                                                          ::testing::ValuesIn(padBegins3d),
                                                                          ::testing::ValuesIn(padEnds3d),
                                                                          ::testing::ValuesIn(dilations3d),
                                                                          ::testing::ValuesIn(numOutChannels_Planar),
                                                                          ::testing::ValuesIn(numGroups_Planar),
                                                                          ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                          ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(nightly_GroupDeconv_3D_Planar_FP32,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_Planar_3D,
                                            ::testing::ValuesIn(Planar_3D_inputs_nightly),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({planar_3D})),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_GroupDeconv_3D_Planar_BF16,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_Planar_3D,
                                            ::testing::ValuesIn(Planar_3D_inputs_nightly),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({planar_3D})),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);

const auto groupConvParams_ExplicitPadding_Blocked_2D_nightly =
    ::testing::Combine(::testing::ValuesIn(kernels2d),
                       ::testing::ValuesIn({strides2d[1]}),
                       ::testing::ValuesIn(padBegins2d),
                       ::testing::ValuesIn(padEnds2d),
                       ::testing::ValuesIn(dilations2d),
                       ::testing::ValuesIn(numOutChannels_Blocked),
                       ::testing::ValuesIn(numGroups_Blocked),
                       ::testing::Values(ov::op::PadType::EXPLICIT),
                       ::testing::ValuesIn(emptyOutputPadding));

const std::vector<DeconvInputData> Blocked_2D_inputs_nightly = {
    DeconvInputData{InputShape{{-1, 64, -1, -1}, {{2, 64, 7, 7}, {2, 64, 5, 7}, {1, 64, 9, 4}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {}},
    DeconvInputData{InputShape{{-1, 64, -1, -1}, {{2, 64, 7, 7}, {2, 64, 5, 7}, {1, 64, 9, 4}, {2, 64, 7, 7}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {{15, 15}}},
    DeconvInputData{InputShape{{{1, 10}, 64, 7, 7}, {{2, 64, 7, 7}, {3, 64, 7, 7}, {1, 64, 7, 7}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {{15, 15}}},
    DeconvInputData{InputShape{{}, {{2, 64, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
    DeconvInputData{InputShape{{-1, 64, -1, -1}, {{2, 64, 7, 7}, {2, 64, 5, 7}, {1, 64, 9, 5}}},
                    ov::test::utils::InputLayerType::PARAMETER,
                    {{15, 15}, {9, 10}, {19, 9}}}};

const auto groupConvParams_ExplicitPadding_Blocked_2D = ::testing::Combine(::testing::ValuesIn(kernels2d),
                                                                           ::testing::ValuesIn(strides2d),
                                                                           ::testing::ValuesIn(padBegins2d),
                                                                           ::testing::ValuesIn(padEnds2d),
                                                                           ::testing::ValuesIn(dilations2d),
                                                                           ::testing::ValuesIn(numOutChannels_Blocked),
                                                                           ::testing::ValuesIn(numGroups_Blocked),
                                                                           ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                           ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(nightly_GroupDeconv_2D_Blocked_FP32,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_Blocked_2D,
                                            ::testing::ValuesIn(Blocked_2D_inputs_nightly),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({block16c_2D, block8c_2D})),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_GroupDeconv_2D_Blocked_BF16,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_Blocked_2D,
                                            ::testing::ValuesIn(Blocked_2D_inputs_nightly),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({block16c_2D})),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (nspc 2D) ============= */
const std::vector<DeconvInputData> nspc_2D_inputs_smoke = {
    DeconvInputData{InputShape{{}, {{2, 64, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
    DeconvInputData{InputShape{{-1, 64, -1, -1}, {{2, 64, 7, 7}, {2, 64, 5, 7}, {1, 64, 9, 5}}},
                    ov::test::utils::InputLayerType::PARAMETER,
                    {{15, 15}, {9, 10}, {19, 9}}}};

const auto groupConvParams_ExplicitPadding_nspc_2D = ::testing::Combine(::testing::ValuesIn(brgKernels2d),
                                                                        ::testing::ValuesIn({strides2d[0]}),
                                                                        ::testing::ValuesIn(padBegins2d),
                                                                        ::testing::ValuesIn(padEnds2d),
                                                                        ::testing::ValuesIn(dilations2d),
                                                                        ::testing::ValuesIn(numOutChannels_nspc),
                                                                        ::testing::ValuesIn(numGroups_nspc),
                                                                        ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                        ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_AMX_nspc_BF16,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_nspc_2D,
                                            ::testing::ValuesIn(nspc_2D_inputs_smoke),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSetBrg),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_nspc_brgconv_amx})),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_nspc_brg,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_nspc_2D,
                                            ::testing::ValuesIn(nspc_2D_inputs_smoke),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSetBrg),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_nspc_brgconv, conv_avx2_2D_nspc_brgconv})),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (Blocked 3D) ============= */
const std::vector<DeconvInputData> Blocked_3D_inputs_nightly = {
    DeconvInputData{InputShape{{-1, 64, -1, -1, -1}, {{1, 64, 5, 5, 5}, {2, 64, 5, 7, 5}, {1, 64, 5, 5, 5}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {}},
    DeconvInputData{InputShape{{-1, 64, -1, -1, -1}, {{1, 64, 5, 5, 5}, {2, 64, 5, 7, 5}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {{7, 7, 7}}},
    DeconvInputData{InputShape{{{1, 10}, 64, -1, -1, -1}, {{1, 64, 5, 5, 5}, {2, 64, 5, 5, 5}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {{7, 7, 7}}},
    DeconvInputData{InputShape{{}, {{2, 64, 7, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
    DeconvInputData{InputShape{{-1, 64, -1, -1, -1}, {{1, 64, 5, 5, 5}, {2, 64, 5, 7, 5}}},
                    ov::test::utils::InputLayerType::PARAMETER,
                    {{7, 7, 7}, {7, 9, 7}}}};

const auto groupConvParams_ExplicitPadding_Blocked_3D = ::testing::Combine(::testing::ValuesIn(kernels3d),
                                                                           ::testing::ValuesIn(strides3d),
                                                                           ::testing::ValuesIn(padBegins3d),
                                                                           ::testing::ValuesIn(padEnds3d),
                                                                           ::testing::ValuesIn(dilations3d),
                                                                           ::testing::ValuesIn(numOutChannels_Blocked),
                                                                           ::testing::ValuesIn(numGroups_Blocked),
                                                                           ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                           ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(nightly_GroupDeconv_3D_Blocked_FP32,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_Blocked_3D,
                                            ::testing::ValuesIn(Blocked_3D_inputs_nightly),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({block16c_3D})),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_GroupDeconv_3D_Blocked_BF16,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_Blocked_3D,
                                            ::testing::ValuesIn(Blocked_3D_inputs_nightly),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({block16c_3D})),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (nspc 3D) ============= */
const std::vector<DeconvInputData> nspc_3D_inputs_smoke = {
    DeconvInputData{InputShape{{}, {{2, 64, 7, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
    DeconvInputData{InputShape{{-1, 64, -1, -1, -1}, {{1, 64, 5, 5, 5}, {2, 64, 5, 7, 5}}},
                    ov::test::utils::InputLayerType::PARAMETER,
                    {{7, 7, 7}, {7, 9, 7}}}};

const auto groupConvParams_ExplicitPadding_nspc_3D = ::testing::Combine(::testing::ValuesIn(brgKernels3d),
                                                                        ::testing::ValuesIn({strides3d[0]}),
                                                                        ::testing::ValuesIn(padBegins3d),
                                                                        ::testing::ValuesIn(padEnds3d),
                                                                        ::testing::ValuesIn(dilations3d),
                                                                        ::testing::ValuesIn(numOutChannels_nspc),
                                                                        ::testing::ValuesIn(numGroups_nspc),
                                                                        ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                        ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_3D_nspc_BF16,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_nspc_3D,
                                            ::testing::ValuesIn(nspc_3D_inputs_smoke),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSetBrg),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D_nspc_brgconv_amx})),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_3D_nspc_brg,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_nspc_3D,
                                            ::testing::ValuesIn(nspc_3D_inputs_smoke),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSetBrg),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D_nspc_brgconv, conv_avx2_3D_nspc_brgconv})),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);
/* ============= GroupConvolution (DW 2D) ============= */
const std::vector<DeconvInputData> dw_2D_inputs_nightly = {
    DeconvInputData{InputShape{{-1, 32, -1, -1}, {{1, 32, 5, 5}, {2, 32, 5, 7}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {}},
    DeconvInputData{InputShape{{-1, 32, -1, -1}, {{1, 32, 5, 5}, {2, 32, 5, 7}, {1, 32, 5, 5}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {{7, 7}}},
    DeconvInputData{InputShape{{{1, 10}, 32, 5, 5}, {{2, 32, 5, 5}, {1, 32, 5, 5}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {{7, 7}}},
    DeconvInputData{InputShape{{}, {{2, 32, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
    DeconvInputData{InputShape{{-1, 32, -1, -1}, {{1, 32, 5, 5}, {2, 32, 5, 7}}},
                    ov::test::utils::InputLayerType::PARAMETER,
                    {{7, 7}, {7, 9}}}};

const auto groupConvParams_ExplicitPadding_DW_2D = ::testing::Combine(::testing::ValuesIn(kernels2d),
                                                                      ::testing::ValuesIn(strides2d),
                                                                      ::testing::ValuesIn(padBegins2d),
                                                                      ::testing::ValuesIn(padEnds2d),
                                                                      ::testing::ValuesIn(dilations2d),
                                                                      ::testing::ValuesIn(numOutChannels_DW),
                                                                      ::testing::ValuesIn(numGroups_DW),
                                                                      ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                      ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(nightly_GroupDeconv_2D_DW_FP32,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_DW_2D,
                                            ::testing::ValuesIn(dw_2D_inputs_nightly),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({block16c_2D,
                                                                                        block8c_2D})),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_GroupDeconv_2D_DW_BF16,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupConvParams_ExplicitPadding_DW_2D,
                                            ::testing::ValuesIn(dw_2D_inputs_nightly),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({block16c_2D})),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Reorder + GroupDeconvolution ============= */
INSTANTIATE_TEST_SUITE_P(
    smoke_reorder_GroupDeconv_2D,
    GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(kernels2d),
                                          ::testing::Values(std::vector<size_t>{1, 1}),
                                          ::testing::ValuesIn(padBegins2d),
                                          ::testing::ValuesIn(padEnds2d),
                                          ::testing::ValuesIn(dilations2d),
                                          ::testing::ValuesIn(numOutChannels_Blocked),
                                          ::testing::ValuesIn(numGroups_Blocked),
                                          ::testing::Values(ov::op::PadType::EXPLICIT),
                                          ::testing::ValuesIn(emptyOutputPadding)),
                       ::testing::Values(DeconvInputData{
                           InputShape{{-1, 64, -1, -1}, {{1, 64, 7, 7}, {2, 64, 5, 7}, {1, 64, 9, 4}, {1, 64, 7, 7}}},
                           ov::test::utils::InputLayerType::PARAMETER,
                           {{15, 15}, {9, 10}, {9, 9}, {15, 15}}}),
                       ::testing::Values(ElementType::f32),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::ValuesIn(filterCPUInfoForDevice({block16c_2D})),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupDeconvolution auto padding tests ============= */
const std::vector<DeconvInputData> inputs_2D_AutoPadding = {
    DeconvInputData{InputShape{{}, {{2, 64, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
    DeconvInputData{InputShape{{-1, 64, -1, -1}, {{2, 64, 7, 7}, {2, 64, 5, 7}, {1, 64, 9, 4}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {}},
    DeconvInputData{InputShape{{-1, 64, -1, -1}, {{1, 64, 7, 7}, {2, 64, 5, 7}, {1, 64, 7, 7}}},
                    ov::test::utils::InputLayerType::CONSTANT,
                    {{15, 15}}},
    DeconvInputData{InputShape{{-1, 64, -1, -1}, {{2, 64, 7, 7}, {2, 64, 5, 7}, {1, 64, 9, 5}}},
                    ov::test::utils::InputLayerType::PARAMETER,
                    {{15, 15}, {9, 10}, {19, 9}}}};

const auto groupDeconvParams_AutoPadding_2D =
    ::testing::Combine(::testing::ValuesIn(kernels2d),
                       ::testing::ValuesIn(strides2d),
                       ::testing::ValuesIn(padBegins2d),
                       ::testing::ValuesIn(padEnds2d),
                       ::testing::ValuesIn(dilations2d),
                       ::testing::ValuesIn(numOutChannels_Blocked),
                       ::testing::ValuesIn(numGroups_Blocked),
                       ::testing::Values(ov::op::PadType::SAME_UPPER, ov::op::PadType::SAME_LOWER),
                       ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(nightly_GroupDeconv_2D_AutoPadding_FP32,
                         GroupDeconvolutionLayerCPUTest,
                         ::testing::Combine(groupDeconvParams_AutoPadding_2D,
                                            ::testing::ValuesIn(inputs_2D_AutoPadding),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::ValuesIn(filterCPUInfoForDevice({planar_2D, block16c_2D})),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         GroupDeconvolutionLayerCPUTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
