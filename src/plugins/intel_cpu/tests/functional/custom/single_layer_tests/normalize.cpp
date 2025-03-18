// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/fusing_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using NormalizeL2LayerCPUTestParamSet = std::tuple<InputShape,            // input shape
                                                   ElementType,           // input element type
                                                   std::vector<int64_t>,  // axes
                                                   float,                 // eps
                                                   ov::op::EpsMode,       // eps_mode
                                                   CPUSpecificParams,
                                                   fusingSpecificParams>;

class NormalizeL2LayerCPUTest : public testing::WithParamInterface<NormalizeL2LayerCPUTestParamSet>,
                                virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<NormalizeL2LayerCPUTestParamSet> obj) {
        InputShape shapes;
        ElementType inType;
        std::vector<int64_t> axes;
        float eps;
        ov::op::EpsMode epsMode;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(shapes, inType, axes, eps, epsMode, cpuParams, fusingParams) = obj.param;

        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "Prc=" << inType << "_";
        results << "axes=" << ov::test::utils::vec2str(axes) << "_";
        results << "eps=" << eps << "_";
        results << "epsMode=" << epsMode << "_";
        results << CPUTestsBase::getTestCaseName(cpuParams);
        results << CpuTestWithFusing::getTestCaseName(fusingParams);

        return results.str();
    }

protected:
    void SetUp() override {
        InputShape shapes;
        ElementType inType;
        std::vector<int64_t> axes;
        float eps;
        ov::op::EpsMode epsMode;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(shapes, inType, axes, eps, epsMode, cpuParams, fusingParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;
        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }

        if (!with_cpu_x86_avx512_core() && inType == ElementType::bf16) {
            selectedType = makeSelectedTypeStr("unknown", ElementType::f32);
        } else {
            selectedType = makeSelectedTypeStr("unknown", inType);
        }

        targetDevice = ov::test::utils::DEVICE_CPU;
        init_input_shapes({shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        auto normAxes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);
        auto normalize = std::make_shared<ov::op::v0::NormalizeL2>(params[0], normAxes, eps, epsMode);

        function = makeNgraphFunction(inType, params, normalize, "Normalize");

        if (inType == ov::element::bf16) {
            abs_threshold = 1e-1f;
        }
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (funcInput.get_element_type().is_real()) {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = -5;
                in_data.range = 10;
                in_data.resolution = 7;
                in_data.seed = 222;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(NormalizeL2LayerCPUTest, CompareWithRefs) {
    run();

    CheckPluginRelatedResults(compiledModel, "NormalizeL2");
}

namespace {

/* ============= Common params ============= */
std::vector<fusingSpecificParams> fusingParamsSet {
        emptyFusingSpec,
#if defined(OPENVINO_ARCH_X86_64)
        fusingMultiplyPerTensor,
        fusingRelu,
        fusingPReluPerChannel
#endif
};

std::vector<fusingSpecificParams> fusingParamsSetDynamic {
    emptyFusingSpec,
#if defined(OPENVINO_ARCH_X86_64)
    fusingMultiplyPerTensor,
    fusingRelu,
    fusingFakeQuantizePerTensor
#endif
};

std::vector<fusingSpecificParams> fusingParamsSetPerChannel {
#if defined(OPENVINO_ARCH_X86_64)
    fusingPReluPerChannel,
    fusingFakeQuantizePerChannel
#endif
};

const float epsilon = 1e-4f;
const op::EpsMode epsMode = op::EpsMode::ADD;
const std::vector<ElementType> netPrecisions = {
        ElementType::f32,
        ElementType::bf16
};

/* ============= 2D ============= */
const std::vector<ov::Shape> inputShapeStatic_2D = {
        {2, 3},
        {2, 16},
        {3, 20}
};

const std::vector<InputShape> inputShapeDynamic_2D = {
        {{-1, -1},
        {{2, 3}, {2, 3}, {5, 5}, {2, 3}}},

        {{-1, 5},
        {{5, 5}, {5, 5}, {12, 5}, {5, 5}}},

        {{{1, 5}, {8, 16}},
        {{3, 8}, {5, 16}, {3, 10}}}
};

const std::vector<std::vector<int64_t>> axes_2D = {
    {1}
};

INSTANTIATE_TEST_SUITE_P(smoke_Static_2D, NormalizeL2LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_shapes_to_test_representation(inputShapeStatic_2D)),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::ValuesIn(axes_2D),
                                 ::testing::Values(epsilon),
                                 ::testing::Values(epsMode),
                                 ::testing::Values(CPUSpecificParams{}),
                                 ::testing::ValuesIn(fusingParamsSet)),
                         NormalizeL2LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Dynamic_2D, NormalizeL2LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapeDynamic_2D),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::ValuesIn(axes_2D),
                                 ::testing::Values(epsilon),
                                 ::testing::Values(epsMode),
                                 ::testing::Values(CPUSpecificParams{}),
                                 ::testing::ValuesIn(fusingParamsSetDynamic)),
                         NormalizeL2LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Dynamic_2D_FusingPerChannel, NormalizeL2LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Values(inputShapeDynamic_2D[1]),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::ValuesIn(axes_2D),
                                 ::testing::Values(epsilon),
                                 ::testing::Values(epsMode),
                                 ::testing::Values(CPUSpecificParams{}),
                                 ::testing::ValuesIn(fusingParamsSetPerChannel)),
                         NormalizeL2LayerCPUTest::getTestCaseName);

/* ============= 3D ============= */
const std::vector<ov::Shape> inputShapeStatic_3D = {
        {2, 3, 4},
        {2, 16, 6},
        {3, 20, 10}
};

const std::vector<InputShape> inputShapeDynamic_3D = {
        {{-1, -1, -1},
         {{2, 3, 4}, {2, 5, 5}, {1, 10, 2}, {2, 3, 4}}},

        {{-1, 5, -1},
         {{1, 5, 5}, {2, 5, 3}, {5, 5, 5}, {1, 5, 5}}},

        {{{1, 5}, {5, 10}, {5, 10}},
         {{3, 8, 8}, {5, 5, 10}, {5, 5, 10}, {5, 10, 10}}}
};

const std::vector<std::vector<int64_t>> axes_3D = {
    {1, 2},
    {2, 1},
    {1}
};

INSTANTIATE_TEST_SUITE_P(smoke_Static_3D, NormalizeL2LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_shapes_to_test_representation(inputShapeStatic_3D)),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::ValuesIn(axes_3D),
                                 ::testing::Values(epsilon),
                                 ::testing::Values(epsMode),
                                 ::testing::Values(CPUSpecificParams{}),
                                 ::testing::ValuesIn(fusingParamsSet)),
                         NormalizeL2LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Dynamic_3D, NormalizeL2LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapeDynamic_3D),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::ValuesIn(axes_3D),
                                 ::testing::Values(epsilon),
                                 ::testing::Values(epsMode),
                                 ::testing::Values(CPUSpecificParams{}),
                                 ::testing::ValuesIn(fusingParamsSetDynamic)),
                         NormalizeL2LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Dynamic_3D_FusingPerChannel, NormalizeL2LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Values(inputShapeDynamic_3D[1]),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::ValuesIn(axes_3D),
                                 ::testing::Values(epsilon),
                                 ::testing::Values(epsMode),
                                 ::testing::Values(CPUSpecificParams{}),
                                 ::testing::ValuesIn(fusingParamsSetPerChannel)),
                         NormalizeL2LayerCPUTest::getTestCaseName);

/* ============= 4D ============= */
const std::vector<ov::Shape> inputShapeStatic_4D = {
        {2, 3, 4, 4},
        {2, 16, 7, 6},
        {3, 20, 2, 10}
};

const std::vector<InputShape> inputShapeDynamic_4D = {
        {{-1, -1, -1, -1},
         {{2, 3, 4, 5}, {2, 5, 5, 5}, {1, 16, 2, 4}, {2, 3, 4, 5}}},

        {{-1, 5, -1, -1},
        {{1, 5, 5, 8}, {1, 5, 5, 8}, {3, 5, 8, 8}, {1, 5, 5, 8}}},

        {{{1, 5}, {5, 16}, {5, 10}, {5, 10}},
        {{3, 8, 8, 8}, {5, 7, 10, 10}, {1, 16, 7, 9}, {5, 9, 10, 5}}}
};

const std::vector<std::vector<int64_t>> axes_4D = {
    {1, 2, 3},
    {3, 1, 2},
    {1}
};

std::vector<CPUSpecificParams> getCPUSpecificParams() {
    std::vector<CPUSpecificParams> result;
    result.push_back(CPUSpecificParams({nchw}, {nchw}, {}, {}));
    if (with_cpu_x86_sse42()) {
        result.push_back(CPUSpecificParams({nhwc}, {nhwc}, {}, {}));
        if (with_cpu_x86_avx512f()) {
            result.push_back(CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}));
        } else {
            result.push_back(CPUSpecificParams({nChw8c}, {nChw8c}, {}, {}));
        }
    }
    return result;
}

INSTANTIATE_TEST_SUITE_P(smoke_Static_4D, NormalizeL2LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_shapes_to_test_representation(inputShapeStatic_4D)),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::ValuesIn(axes_4D),
                                 ::testing::Values(epsilon),
                                 ::testing::Values(epsMode),
                                 ::testing::ValuesIn(getCPUSpecificParams()),
                                 ::testing::ValuesIn(fusingParamsSet)),
                         NormalizeL2LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Dynamic_4D, NormalizeL2LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapeDynamic_4D),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::ValuesIn(axes_4D),
                                 ::testing::Values(epsilon),
                                 ::testing::Values(epsMode),
                                 ::testing::ValuesIn(getCPUSpecificParams()),
                                 ::testing::ValuesIn(fusingParamsSetDynamic)),
                         NormalizeL2LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Dynamic_4D_FusingPerChannel, NormalizeL2LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Values(inputShapeDynamic_4D[1]),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::ValuesIn(axes_4D),
                                 ::testing::Values(epsilon),
                                 ::testing::Values(epsMode),
                                 ::testing::ValuesIn(getCPUSpecificParams()),
                                 ::testing::ValuesIn(fusingParamsSetPerChannel)),
                         NormalizeL2LayerCPUTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
