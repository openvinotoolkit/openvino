// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "openvino/core/preprocess/pre_post_process.hpp"
#include <transformations/op_conversions/convert_interpolate11_downgrade.hpp>

using namespace ov::test;
using namespace CPUTestUtils;
using ngraph::helpers::operator<<;

namespace CPULayerTestsDefinitions {

using InterpolateSpecificParams = std::tuple<ov::op::v11::Interpolate::InterpolateMode,          // InterpolateMode
                                             ov::op::v11::Interpolate::CoordinateTransformMode,  // CoordinateTransformMode
                                             ov::op::v11::Interpolate::NearestMode,              // NearestMode
                                             bool,                                                  // AntiAlias
                                             std::vector<size_t>,                                   // PadBegin
                                             std::vector<size_t>,                                   // PadEnd
                                             double>;                                               // Cube coef

using ShapeParams = std::tuple<ov::op::v11::Interpolate::ShapeCalcMode, // ShapeCalculationMode
                               InputShape,                                 // Input shapes
                               // params describing input, choice of which depends on ShapeCalcMode
                               ngraph::helpers::InputLayerType,            // input type
                               std::vector<std::vector<float>>,            // scales or sizes values
                               std::vector<int64_t>>;                      // axes

using InterpolateLayerCPUTestParamsSet = std::tuple<InterpolateSpecificParams,
                                                    ShapeParams,
                                                    ElementType,
                                                    CPUSpecificParams,
                                                    fusingSpecificParams,
                                                    std::map<std::string, std::string>>;

class InterpolateLayerCPUTest : public testing::WithParamInterface<InterpolateLayerCPUTestParamsSet>,
                                virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateLayerCPUTestParamsSet> obj) {
        InterpolateSpecificParams specificParams;
        ShapeParams shapeParams;
        ElementType prec;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(specificParams, shapeParams, prec, cpuParams, fusingParams, additionalConfig) = obj.param;

        ov::op::v11::Interpolate::InterpolateMode mode;
        ov::op::v11::Interpolate::CoordinateTransformMode transfMode;
        ov::op::v11::Interpolate::NearestMode nearMode;
        bool antiAlias;
        std::vector<size_t> padBegin;
        std::vector<size_t> padEnd;
        double cubeCoef;
        std::tie(mode, transfMode, nearMode, antiAlias, padBegin, padEnd, cubeCoef) = specificParams;

        ov::op::v11::Interpolate::ShapeCalcMode shapeCalcMode;
        InputShape inputShapes;
        ngraph::helpers::InputLayerType shapeInputType;
        std::vector<std::vector<float>> shapeDataForInput;
        std::vector<int64_t> axes;
        std::tie(shapeCalcMode, inputShapes, shapeInputType, shapeDataForInput, axes) = shapeParams;

        std::ostringstream result;
        result << "ShapeCalcMode=" << shapeCalcMode << "_";
        result << "IS=";
        result << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
        result << "TS=";
        for (const auto& shape : inputShapes.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        if (shapeCalcMode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES) {
            result << "Scales=";
        } else {
            result << "Sizes=";
        }
        for (const auto &data : shapeDataForInput) {
            result << ov::test::utils::vec2str(data) << "_";
        }
        result << shapeInputType << "_";
        result << "InterpolateMode=" << mode << "_";
        result << "CoordinateTransformMode=" << transfMode << "_";
        result << "NearestMode=" << nearMode << "_";
        result << "CubeCoef=" << cubeCoef << "_";
        result << "Antialias=" << antiAlias << "_";
        result << "PB=" << ov::test::utils::vec2str(padBegin) << "_";
        result << "PE=" << ov::test::utils::vec2str(padEnd) << "_";
        result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
        result << "PRC=" << prec << "_";

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

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 1) {
                if (shapeCalcMode == ov::op::v4::Interpolate::ShapeCalcMode::SIZES) {
                    tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], sizes[inferRequestNum].data());
                } else {
                    tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], scales[inferRequestNum].data());
                }
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 2560, 0, 256);
            }

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
        inferRequestNum++;
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

protected:
    std::vector<std::vector<float>> scales;
    std::vector<std::vector<int32_t>> sizes;
    ov::op::v4::Interpolate::ShapeCalcMode shapeCalcMode;
    size_t inferRequestNum = 0;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        InterpolateSpecificParams specificParams;
        ShapeParams shapeParams;
        ElementType ngPrc;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(specificParams, shapeParams, ngPrc, cpuParams, fusingParams, additionalConfig) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        ov::op::v11::Interpolate::InterpolateMode mode;
        ov::op::v11::Interpolate::CoordinateTransformMode transfMode;
        ov::op::v11::Interpolate::NearestMode nearMode;
        bool antiAlias;
        std::vector<size_t> padBegin;
        std::vector<size_t> padEnd;
        double cubeCoef;
        std::tie(mode, transfMode, nearMode, antiAlias, padBegin, padEnd, cubeCoef) = specificParams;

        InputShape dataShape;
        ngraph::helpers::InputLayerType shapeInputType;
        std::vector<std::vector<float>> shapeDataForInput;
        std::vector<int64_t> axes;
        std::tie(shapeCalcMode, dataShape, shapeInputType, shapeDataForInput, axes) = shapeParams;

        if (shapeCalcMode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES) {
            scales = shapeDataForInput;
        } else {
            sizes.resize(shapeDataForInput.size());
            for (size_t i = 0; i < shapeDataForInput.size(); i++) {
                for (size_t j = 0; j < shapeDataForInput[i].size(); j++) {
                    sizes[i].push_back(shapeDataForInput[i][j]);
                }
            }
        }

        std::vector<InputShape> inputShapes;
        inputShapes.push_back(dataShape);
        if (shapeInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            inputShapes.push_back(InputShape({static_cast<int64_t>(axes.size())}, std::vector<ov::Shape>(dataShape.second.size(), {axes.size()})));
        }

        if (additionalConfig[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] == InferenceEngine::PluginConfigParams::YES) {
            inType = outType = ngPrc = ElementType::bf16;
            rel_threshold = 1e-2f;
        } else {
            inType = outType = ngPrc;
        }

        init_input_shapes(inputShapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, inputDynamicShapes.front())};
        std::shared_ptr<ov::Node> sizesInput, scalesInput;
        if (shapeCalcMode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES) {
            if (shapeInputType == ngraph::helpers::InputLayerType::PARAMETER) {
                auto paramNode = std::make_shared<ov::op::v0::Parameter>(ElementType::f32, ov::Shape{scales.front().size()});
                params.push_back(paramNode);
                scalesInput = paramNode;
            } else {
                scalesInput = std::make_shared<ov::op::v0::Constant>(ElementType::f32, ov::Shape{scales.front().size()}, scales.front());
            }
        } else {
            if (shapeInputType == ngraph::helpers::InputLayerType::PARAMETER) {
                auto paramNode = std::make_shared<ov::op::v0::Parameter>(ElementType::i32, ov::Shape{sizes.front().size()});
                params.push_back(paramNode);
                sizesInput = paramNode;
            } else {
                sizesInput = std::make_shared<ov::op::v0::Constant>(ElementType::i32, ov::Shape{sizes.front().size()}, sizes.front());
            }
        }
        auto axesInput = std::make_shared<ov::op::v0::Constant>(ElementType::i64, ov::Shape{axes.size()}, axes);

        for (size_t i = 0; i < params.size(); i++) {
            params[i]->set_friendly_name(std::string("param_") + std::to_string(i));
        }

        ov::op::v11::Interpolate::InterpolateAttrs interpAttr{mode, shapeCalcMode, padBegin, padEnd, transfMode, nearMode,
                                                                            antiAlias, cubeCoef};

        std::shared_ptr<ov::op::v11::Interpolate> interp = nullptr;
        if (shapeCalcMode == ov::op::v11::Interpolate::ShapeCalcMode::SCALES) {
            interp = std::make_shared<ov::op::v11::Interpolate>(params[0], scalesInput, axesInput, interpAttr);
        } else {
            interp = std::make_shared<ov::op::v11::Interpolate>(params[0], sizesInput, axesInput, interpAttr);
        }

        function = makeNgraphFunction(ngPrc, params, interp, "InterpolateCPU");

        ov::pass::Manager m;
        m.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
        m.run_passes(function);

        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }
        selectedType = makeSelectedTypeStr(selectedType, ngPrc);
    }
};

TEST_P(InterpolateLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Interpolate");
}

namespace {
const std::vector<ov::op::v11::Interpolate::CoordinateTransformMode> coordinateTransformModes_Smoke = {
        ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC,
};

const std::vector<ov::op::v11::Interpolate::CoordinateTransformMode> coordinateTransformModes_Full = {
        ov::op::v11::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
        ov::op::v11::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC,
        ov::op::v11::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

const std::vector<ov::op::v11::Interpolate::NearestMode> nearestModes_Smoke = {
        ov::op::v11::Interpolate::NearestMode::SIMPLE,
        ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ov::op::v11::Interpolate::NearestMode::FLOOR,
};

const std::vector<ov::op::v11::Interpolate::NearestMode> nearestModes_Full = {
        ov::op::v11::Interpolate::NearestMode::SIMPLE,
        ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ov::op::v11::Interpolate::NearestMode::FLOOR,
        ov::op::v11::Interpolate::NearestMode::CEIL,
        ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_CEIL,
};

const std::vector<ov::op::v11::Interpolate::NearestMode> defNearestModes = {
        ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
};

const std::vector<bool> antialias = {
        false,
};

const std::vector<double> cubeCoefs = {
        -0.75f,
};

const std::vector<fusingSpecificParams> interpolateFusingParamsSet{
        emptyFusingSpec,
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
        fusingSwish,
        fusingFakeQuantizePerTensorRelu,
#endif
};

std::vector<std::map<std::string, std::string>> filterAdditionalConfig() {
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        return {
            {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES}}
        };
    } else {
        return {
            // default config as an stub for target without avx512, otherwise all tests with BF16 in its name are skipped
            {{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO}}
        };
    }
}

// 3D
std::vector<CPUSpecificParams> filterCPUInfoForDevice3D() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (InferenceEngine::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{ncw, x, x, x}, {ncw}, {"jit_avx2"}, "jit_avx2"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{ncw, x, x, x}, {ncw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

std::vector<std::map<std::string, std::string>> filterAdditionalConfig3D() {
    return {
        // default config as an stub
        {{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO}}
    };
}

const std::vector<std::vector<size_t>> pads3D = {
    {0, 0, 0},
    {0, 0, 1},
};

const std::vector<std::vector<int64_t>> defaultAxes3D = {
    {0, 1, 2}
};

const std::vector<ShapeParams> shapeParams3D = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 3, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f}},
        defaultAxes3D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{}, {{1, 3, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 3, 5}},
        defaultAxes3D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1}, {{1, 3, 4}, {2, 4, 6}, {1, 3, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f}, {1.f, 1.f, 1.25f}, {1.f, 1.f, 1.5f}},
        defaultAxes3D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1}, {{1, 3, 4}, {2, 4, 6}, {1, 3, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1, 3, 6}, {2, 4, 8}, {1, 3, 6}},
        defaultAxes3D.front()
    }
};

const auto interpolateCasesNN_Smoke_3D = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads3D),
        ::testing::ValuesIn(pads3D),
        ::testing::ValuesIn(cubeCoefs));
INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_Test_3D, InterpolateLayerCPUTest,
        ::testing::Combine(
             interpolateCasesNN_Smoke_3D,
            ::testing::ValuesIn(shapeParams3D),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesNN_Full_3D = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(nearestModes_Full),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads3D),
        ::testing::ValuesIn(pads3D),
        ::testing::ValuesIn(cubeCoefs));
INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_Test_3D, InterpolateLayerCPUTest,
         ::testing::Combine(
             interpolateCasesNN_Full_3D,
             ::testing::ValuesIn(shapeParams3D),
             ::testing::Values(ElementType::f32),
             ::testing::ValuesIn(filterCPUInfoForDevice3D()),
             ::testing::ValuesIn(interpolateFusingParamsSet),
             ::testing::ValuesIn(filterAdditionalConfig3D())),
     InterpolateLayerCPUTest::getTestCaseName);

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
const std::vector<fusingSpecificParams> interpolateFusingParamsSet3D_fixed_C() {
    std::vector<fusingSpecificParams> fuseParams;
    if (InferenceEngine::with_cpu_x86_avx2()) {
        fuseParams.push_back(fusingFakeQuantizePerChannelRelu);
        fuseParams.push_back(fusingMultiplyPerChannel);
    }
    fuseParams.push_back(emptyFusingSpec);
    return fuseParams;
}

const std::vector<ShapeParams> shapeParams3D_fixed_C = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 3, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f}},
        defaultAxes3D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, 3, -1}, {{1, 3, 4}, {1, 3, 6}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 3, 8}},
        defaultAxes3D.front()
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_PerChannelFuse3D_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN_Smoke_3D,
            ::testing::ValuesIn(shapeParams3D_fixed_C),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet3D_fixed_C()),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_PerChannelFuse3D_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN_Smoke_3D,
            ::testing::ValuesIn(shapeParams3D_fixed_C),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet3D_fixed_C()),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);
#endif

const auto interpolateCasesLinear3D_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads3D),
        ::testing::ValuesIn(pads3D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesLinear3D_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads3D),
        ::testing::ValuesIn(pads3D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinear_Layout3D_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear3D_Smoke,
            ::testing::ValuesIn(shapeParams3D),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinear_Layout3D_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear3D_Full,
            ::testing::ValuesIn(shapeParams3D),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesCubic3D_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::CUBIC),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads3D),
        ::testing::ValuesIn(pads3D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesCubic3D_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::CUBIC),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads3D),
        ::testing::ValuesIn(pads3D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateCubic_Layout3D_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesCubic3D_Smoke,
            ::testing::ValuesIn(shapeParams3D),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateCubic_Layout3D_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesCubic3D_Full,
            ::testing::ValuesIn(shapeParams3D),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);

// 4D
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, x, x, x}, {nChw16c}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_avx512"}, "jit_avx512"});
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x, x}, {nChw8c}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"jit_avx2"}, "jit_avx2"});
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x, x}, {nChw8c}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_sse42"}, "jit_sse42"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

const std::vector<std::vector<size_t>> pads4D = {
        {0, 0, 0, 0},
        {0, 0, 1, 1},
};

const std::vector<std::vector<int64_t>> defaultAxes4D = {
    {0, 1, 2, 3}
};

const std::vector<ShapeParams> shapeParams4D_Smoke = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{}, {{1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 11, 5, 6}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f}, {1.f, 1.f, 1.25f, 1.25f}, {1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1, 11, 6, 7}, {2, 7, 8, 7}, {1, 11, 6, 7}},
        defaultAxes4D.front()
    }
};

const std::vector<ShapeParams> shapeParams4D_Full = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {1, 11, 5, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 11, 5, 6}},
        defaultAxes4D.front()
    }
};

const auto interpolateCasesNN_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesNN_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(nearestModes_Full),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
             interpolateCasesNN_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_Test, InterpolateLayerCPUTest,
         ::testing::Combine(
             interpolateCasesNN_Full,
             ::testing::ValuesIn(shapeParams4D_Full),
             ::testing::Values(ElementType::f32),
             ::testing::ValuesIn(filterCPUInfoForDevice()),
             ::testing::ValuesIn(interpolateFusingParamsSet),
             ::testing::ValuesIn(filterAdditionalConfig())),
     InterpolateLayerCPUTest::getTestCaseName);

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
const std::vector<fusingSpecificParams> interpolateFusingParamsSet_fixed_C{
        fusingFakeQuantizePerChannelRelu,
        fusingMultiplyPerChannel,
};

const std::vector<ShapeParams> shapeParams4D_fixed_C = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, 16, -1, -1}, {{1, 16, 4, 4}, {1, 16, 6, 5}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 16, 6, 7}},
        defaultAxes4D.front()
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_PerChannelFuse_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN_Smoke,
            ::testing::ValuesIn(shapeParams4D_fixed_C),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet_fixed_C),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_PerChannelFuse_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN_Full,
            ::testing::ValuesIn(shapeParams4D_fixed_C),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet_fixed_C),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);
#endif

const auto interpolateCasesLinearOnnx_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesLinearOnnx_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinearOnnx_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesLinear_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesLinear_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinear_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinear_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesCubic_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::CUBIC),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesCubic_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::CUBIC),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateCubic_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesCubic_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateCubic_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesCubic_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

////////////////////////5D/////////////////////////////
std::vector<CPUSpecificParams> filterCPUInfoForDevice5D() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw16c, x, x, x}, {nCdhw16c}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc, x, x, x}, {ndhwc}, {"jit_avx512"}, "jit_avx512"});
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw8c, x, x, x}, {nCdhw8c}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc, x, x, x}, {ndhwc}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{ncdhw, x, x, x}, {ncdhw}, {"jit_avx2"}, "jit_avx2"});
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw8c, x, x, x}, {nCdhw8c}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc, x, x, x}, {ndhwc}, {"jit_sse42"}, "jit_sse42"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{ncdhw, x, x, x}, {ncdhw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

const std::vector<std::vector<size_t>> pads5D = {
        {0, 0, 0, 0, 0}
};

const std::vector<std::vector<int64_t>> defaultAxes5D = {
    {0, 1, 2, 3, 4}
};

const std::vector<ShapeParams> shapeParams5D_Smoke = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{}, {{1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 11, 5, 6, 2}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f, 0.5f}, {1.f, 1.f, 1.25f, 1.25f, 1.25f}, {1.f, 1.f, 1.25f, 1.5f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1, 11, 6, 7, 2}, {2, 7, 8, 7, 4}, {1, 11, 6, 7, 2}},
        defaultAxes5D.front()
    },
};

const std::vector<ShapeParams> shapeParams5D_Full = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {1, 11, 5, 5, 8}, {1, 11, 4, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1, 11, 5, 6, 4}},
        defaultAxes5D.front()
    }
};

const auto interpolateCasesLinearOnnx5D_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));
const auto interpolateCasesLinearOnnx5D_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinearOnnx5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx5D_Smoke,
            ::testing::ValuesIn(shapeParams5D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx5D_Full,
            ::testing::ValuesIn(shapeParams5D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesNN5D_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCasesNN5D_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(nearestModes_Full),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN5D_Smoke,
            ::testing::ValuesIn(shapeParams5D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN5D_Full,
            ::testing::ValuesIn(shapeParams5D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

// corner cases
const std::vector<ShapeParams> shapeParams4D_corner = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{1, 11, 4, 4}, {{1, 11, 4, 4}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f}, {1.f, 1.f, 1.25f, 1.25f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{1, 11, 4, 4}, {{1, 11, 4, 4}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1, 11, 6, 7}, {1, 11, 8, 7}},
        defaultAxes4D.front()
    }
};

const auto interpolateCornerCases = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::Values(ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC),
        ::testing::Values(ov::op::v11::Interpolate::NearestMode::SIMPLE),
        ::testing::ValuesIn(antialias),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_corner_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCornerCases,
            ::testing::ValuesIn(shapeParams4D_corner),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

// bilinear pillow and bicubic pillow test case supported in spec(ov ref)
const std::vector<ov::op::v11::Interpolate::CoordinateTransformMode> coordinateTransformModesPillow_Smoke = {
    ov::op::v11::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
};

const std::vector<double> cubeCoefsPillow = {
    -0.5f,
};

const std::vector<fusingSpecificParams> interpolateFusingPillowParamsSet{
    emptyFusingSpec
};

const std::vector<std::vector<int64_t>> defaultAxes4D_pillow = {
    {2, 3}
};

const std::vector<ShapeParams> shapeParams4D_Pillow_Smoke = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 3, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{2.0f, 4.0f}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{2, 4, 16, 16}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{0.25f, 0.5f}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{}, {{1, 3, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{5, 6}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{}, {{2, 4, 16, 16}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{2, 8}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{1.25f, 1.5f}, {0.5f, 0.75f}, {1.25f, 1.5f}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.25f, 0.75f}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 17, 4, 4}, {2, 3, 10, 12}, {1, 17, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{6, 8}, {5, 4}, {6, 8}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 17, 4, 4}, {2, 3, 10, 12}, {1, 17, 4, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{6, 8}},
        defaultAxes4D_pillow.front()
    },
    // test for only one pass or just copy
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 17, 4, 4}, {2, 3, 10, 12}, {1, 17, 4, 4}}},
        ngraph::helpers::InputLayerType::PARAMETER,
        {{4, 4}, {10, 20}, {10, 4}},
        defaultAxes4D_pillow.front()
    }
};

std::vector<CPUSpecificParams> filterCPUInfoForDevice_pillow() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x}, {nhwc}, {"jit_avx512"}, "jit_avx512"});
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x}, {nhwc}, {"jit_avx2"}, "jit_avx2"});
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x}, {nhwc}, {"jit_sse42"}, "jit_sse42"});
    }
    resCPUParams.push_back(CPUSpecificParams{{nchw, x, x}, {nchw}, {"ref"}, "ref"});
    return resCPUParams;
}
std::vector<std::map<std::string, std::string>> filterPillowAdditionalConfig() {
    return {
        {{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO}}
    };
}

const auto interpolateCasesBilinearPillow_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::BILINEAR_PILLOW),
        ::testing::ValuesIn(coordinateTransformModesPillow_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefsPillow));

const auto interpolateCasesBicubicPillow_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW),
        ::testing::ValuesIn(coordinateTransformModesPillow_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefsPillow));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateBilinearPillow_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBilinearPillow_Smoke,
            ::testing::ValuesIn(shapeParams4D_Pillow_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice_pillow()),
            ::testing::ValuesIn(interpolateFusingPillowParamsSet),
            ::testing::ValuesIn(filterPillowAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateBicubicPillow_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBicubicPillow_Smoke,
            ::testing::ValuesIn(shapeParams4D_Pillow_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice_pillow()),
            ::testing::ValuesIn(interpolateFusingPillowParamsSet),
            ::testing::ValuesIn(filterPillowAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

// pillow modes: planar layout with axis[1,2] executed as nhwc layout case
const std::vector<std::vector<int64_t>> defaultAxes4D_pillow_nchw_as_nhwc = {
    {1, 2}
};

const std::vector<ShapeParams> shapeParams4D_Pillow_Smoke_nchw_as_nhwc = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 4, 4, 3}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{2.0f, 4.0f}},
        defaultAxes4D_pillow_nchw_as_nhwc.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{}, {{2, 16, 16, 4}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{2, 8}},
        defaultAxes4D_pillow_nchw_as_nhwc.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, -1, -1, {2, 20}}, {{1, 4, 4, 11}, {2, 6, 5, 7}, {1,  4, 4, 11}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{1.25f, 0.75f}},
        defaultAxes4D_pillow_nchw_as_nhwc.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, -1, -1, {2, 20}}, {{1, 4, 4, 17}, {2, 10, 12, 3}, {1, 4, 4, 17}}},
        ngraph::helpers::InputLayerType::CONSTANT,
        {{6, 8}},
        defaultAxes4D_pillow_nchw_as_nhwc.front()
    }
};

const std::vector<std::vector<size_t>> pads4D_nchw_as_nhwc = {
        {0, 0, 0, 0}
};

std::vector<CPUSpecificParams> filterCPUInfoForDevice_pillow_nchw_as_nhwc() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x}, {nchw}, {"jit_avx512"}, "jit_avx512"});
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x}, {nchw}, {"jit_avx2"}, "jit_avx2"});
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x}, {nchw}, {"jit_sse42"}, "jit_sse42"});
    }
    return resCPUParams;
}

const auto interpolateCasesBilinearPillow_Smoke_nchw_as_nhwc = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::BILINEAR_PILLOW),
        ::testing::ValuesIn(coordinateTransformModesPillow_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D_nchw_as_nhwc),
        ::testing::ValuesIn(pads4D_nchw_as_nhwc),
        ::testing::ValuesIn(cubeCoefsPillow));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateBilinearPillow_LayoutAlign_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBilinearPillow_Smoke_nchw_as_nhwc,
            ::testing::ValuesIn(shapeParams4D_Pillow_Smoke_nchw_as_nhwc),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice_pillow_nchw_as_nhwc()),
            ::testing::ValuesIn(interpolateFusingPillowParamsSet),
            ::testing::ValuesIn(filterPillowAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesBicubicPillow_Smoke_nchw_as_nhwc = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW),
        ::testing::ValuesIn(coordinateTransformModesPillow_Smoke),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads4D_nchw_as_nhwc),
        ::testing::ValuesIn(pads4D_nchw_as_nhwc),
        ::testing::ValuesIn(cubeCoefsPillow));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateBicubicPillow_LayoutAlign_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBicubicPillow_Smoke_nchw_as_nhwc,
            ::testing::ValuesIn(shapeParams4D_Pillow_Smoke_nchw_as_nhwc),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice_pillow_nchw_as_nhwc()),
            ::testing::ValuesIn(interpolateFusingPillowParamsSet),
            ::testing::ValuesIn(filterPillowAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
