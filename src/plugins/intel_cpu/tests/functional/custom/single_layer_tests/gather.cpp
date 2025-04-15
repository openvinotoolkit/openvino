// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {

typedef std::tuple<std::vector<InputShape>,  // Input shapes
                   std::tuple<int, int>,     // Axis and Batch dim
                   ElementType,              // Network precision
                   bool,                     // Is const Axis
                   CPUSpecificParams,        // CPU specific params
                   ov::AnyMap                // Additional config
                   >
    GatherLayerTestCPUParams;

class GatherLayerTestCPU : public testing::WithParamInterface<GatherLayerTestCPUParams>,
                           virtual public ov::test::SubgraphBaseTest,
                           public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherLayerTestCPUParams> obj) {
        std::vector<InputShape> inputShapes;
        std::tuple<int, int> axisAndBatchDims;
        ElementType netPrecision;
        bool isAxisConstant;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;

        std::tie(inputShapes, axisAndBatchDims, netPrecision, isAxisConstant, cpuParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (size_t i = 0lu; i < inputShapes.size(); i++) {
            result << ov::test::utils::partialShape2str({inputShapes[i].first})
                   << (i < inputShapes.size() - 1lu ? "_" : "");
        }
        result << ")_TS=";
        for (size_t i = 0lu; i < inputShapes.front().second.size(); i++) {
            result << "{";
            for (size_t j = 0lu; j < inputShapes.size(); j++) {
                result << ov::test::utils::vec2str(inputShapes[j].second[i])
                       << (j < inputShapes.size() - 1lu ? "_" : "");
            }
            result << "}_";
        }
        result << "axis=" << std::get<0>(axisAndBatchDims) << "_";
        result << "batchDims=" << std::get<1>(axisAndBatchDims) << "_";
        result << "netPrc=" << netPrecision << "_";
        result << "constAx=" << (isAxisConstant ? "True" : "False") << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                if (item.second == ov::element::bf16)
                    result << "_" << item.first << "=" << item.second.as<std::string>();
            }
        }

        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<InputShape> inputShapes;
        std::tuple<int, int> axisAndBatchDims;
        ElementType netPrecision;
        bool isAxisConstant;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;
        const ElementType intInputsPrecision = ElementType::i64;

        std::tie(inputShapes, axisAndBatchDims, netPrecision, isAxisConstant, cpuParams, additionalConfig) =
            this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        axis = std::get<0>(axisAndBatchDims);
        const int batchDims = std::get<1>(axisAndBatchDims);
        targetDevice = ov::test::utils::DEVICE_CPU;
        init_input_shapes(inputShapes);
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        if (additionalConfig[ov::hint::inference_precision.name()] == ov::element::bf16) {
            selectedType = makeSelectedTypeStr(selectedType, ElementType::bf16);
        } else {
            selectedType = makeSelectedTypeStr(selectedType, netPrecision);
        }

        if (!isAxisConstant) {
            inputDynamicShapes.push_back({1});
            for (size_t i = 0lu; i < targetStaticShapes.size(); i++) {
                targetStaticShapes[i].push_back({1});
            }
        }

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(intInputsPrecision, inputDynamicShapes[1])};
        params[0]->set_friendly_name("data");
        params[1]->set_friendly_name("indices");
        if (!isAxisConstant) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(intInputsPrecision, inputDynamicShapes[2]));
            params[2]->set_friendly_name("axis");
        }
        std::shared_ptr<ov::Node> gatherNode;
        if (isAxisConstant) {
            gatherNode = std::make_shared<ov::op::v8::Gather>(
                params[0],
                params[1],
                ov::op::v0::Constant::create(intInputsPrecision, ov::Shape({1}), {axis}),
                batchDims);
        } else {
            gatherNode = std::make_shared<ov::op::v8::Gather>(params[0], params[1], params[2], batchDims);
        }

        function = makeNgraphFunction(netPrecision, params, gatherNode, "GatherCPU");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        const auto& funcInputs = function->inputs();
        inputs.clear();

        const size_t normAxis = axis < 0 ? axis + targetInputStaticShapes[0].size() : axis;
        const int32_t axisDim = targetInputStaticShapes[0][normAxis];

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            ov::test::utils::InputGenerateData in_data;

            if (funcInput.get_node()->get_friendly_name() == "data") {
                const auto dataTypeSize = funcInput.get_element_type().size();
                in_data.start_from = 0;
                in_data.range = dataTypeSize == 4 ? 0x7FFFFFFF : dataTypeSize == 2 ? 0xFFFF : 0xFF;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[0], in_data);
            } else if (funcInput.get_node()->get_friendly_name() == "indices") {
                in_data.start_from = -axisDim;
                in_data.range = axisDim * 2;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[1], in_data);
            } else if (funcInput.get_node()->get_friendly_name() == "axis") {
                in_data.start_from = axis;
                in_data.range = 1;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), {1}, in_data);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    int64_t axis = 0;
};

typedef std::tuple<InputShape,            // Input shapes
                   std::vector<int64_t>,  // Indices
                   int,                   // Axis
                   ElementType,           // Network precision
                   CPUSpecificParams      // CPU specific params
                   >
    GatherInPlaceLayerTestCPUParams;

class GatherInPlaceLayerTestCPU : public testing::WithParamInterface<GatherInPlaceLayerTestCPUParams>,
                                  virtual public ov::test::SubgraphBaseTest,
                                  public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherInPlaceLayerTestCPUParams> obj) {
        InputShape inputShapes;
        std::vector<int64_t> indices;
        int axis;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;

        std::tie(inputShapes, indices, axis, netPrecision, cpuParams) = obj.param;

        std::ostringstream result;
        result << "IS=(";

        result << ov::test::utils::partialShape2str({inputShapes.first}) << ")_TS=";

        result << "{";
        for (size_t i = 0lu; i < inputShapes.second.size(); i++) {
            result << ov::test::utils::vec2str(inputShapes.second[i])
                   << (i < inputShapes.second.size() - 1lu ? "_" : "");
        }
        result << "}_";
        result << "axis=" << axis << "_";
        result << "indices=" << ov::test::utils::vec2str(indices) << "_";
        result << "netPrc=" << netPrecision << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        InputShape inputShapes;
        std::vector<int64_t> indices;
        int axis;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        constexpr ElementType intInputsPrecision = ElementType::i64;
        constexpr int batchDims = 0;

        std::tie(inputShapes, indices, axis, netPrecision, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        targetDevice = ov::test::utils::DEVICE_CPU;
        init_input_shapes({inputShapes});

        selectedType = makeSelectedTypeStr(selectedType, netPrecision);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0])};
        params[0]->set_friendly_name("data");
        std::shared_ptr<ov::Node> gatherNode = std::make_shared<ov::op::v8::Gather>(
            params[0],
            ov::op::v0::Constant::create(intInputsPrecision, ov::Shape({indices.size()}), indices),
            ov::op::v0::Constant::create(intInputsPrecision, ov::Shape({1}), {axis}),
            batchDims);

        function = makeNgraphFunction(netPrecision, params, gatherNode, "GatherCPU");
    }
};

TEST_P(GatherLayerTestCPU, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Gather");
}

TEST_P(GatherInPlaceLayerTestCPU, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Gather");
}

namespace {
const std::vector<ElementType> netPrecisions = {ElementType::f32, ElementType::bf16, ElementType::i8};

std::vector<ov::AnyMap> additionalConfig = {{{ov::hint::inference_precision(ov::element::f32)}},
                                            {{ov::hint::inference_precision(ov::element::bf16)}}};

std::vector<bool> isAxisConst{true, false};
const CPUSpecificParams cpuParamsRef{{}, {}, {"ref_any"}, "ref_any"};

std::vector<CPUSpecificParams> getCPUInfo() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (ov::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx512"}, "jit_avx512"});
    } else if (ov::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx2"}, "jit_avx2"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

///// 1D /////
const std::vector<std::vector<ov::test::InputShape>> staticInputShapes1D = {
    {{{}, {{1}}}, {{}, {{1}}}},   {{{}, {{2}}}, {{}, {{2}}}},   {{{}, {{3}}}, {{}, {{3}}}},
    {{{}, {{4}}}, {{}, {{4}}}},   {{{}, {{5}}}, {{}, {{5}}}},   {{{}, {{6}}}, {{}, {{6}}}},
    {{{}, {{7}}}, {{}, {{7}}}},   {{{}, {{8}}}, {{}, {{8}}}},   {{{}, {{9}}}, {{}, {{9}}}},
    {{{}, {{11}}}, {{}, {{11}}}}, {{{}, {{13}}}, {{}, {{13}}}}, {{{}, {{15}}}, {{}, {{15}}}},
    {{{}, {{16}}}, {{}, {{16}}}}, {{{}, {{17}}}, {{}, {{17}}}}, {{{}, {{19}}}, {{}, {{19}}}},
    {{{}, {{23}}}, {{}, {{23}}}}, {{{}, {{24}}}, {{}, {{24}}}}, {{{}, {{32}}}, {{}, {{32}}}},
    {{{}, {{33}}}, {{}, {{33}}}}, {{{}, {{37}}}, {{}, {{37}}}}, {{{}, {{41}}}, {{}, {{41}}}},
    {{{}, {{48}}}, {{}, {{48}}}}, {{{}, {{51}}}, {{}, {{51}}}}, {{{}, {{63}}}, {{}, {{63}}}},
    {{{}, {{64}}}, {{}, {{64}}}}, {{{}, {{65}}}, {{}, {{65}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_static_1D,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(staticInputShapes1D),
                                            ::testing::Values(std::tuple<int, int>{0, 0}),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> staticInputShapes1DI32 = {{{{}, {{1}}}, {{}, {{1}}}},
                                                                               {{{}, {{15}}}, {{}, {{15}}}},
                                                                               {{{}, {{64}}}, {{}, {{64}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_static_1D_I32,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(staticInputShapes1DI32),
                                            ::testing::Values(std::tuple<int, int>{0, 0}),
                                            ::testing::Values(ElementType::i32),
                                            ::testing::Values(true),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"}),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes1D = {
    {{{ov::Dimension{1, 70}},  // Dynamic shape 0
      {{1},  {2},  {3},  {4},  {5},  {6},  {7},  {8},  {9},  {11}, {13},
       {15}, {16}, {17}, {19}, {23}, {24}, {32}, {55}, {63}, {64}, {65}}},  // Target shapes
     {{-1},                                                                 // Dynamic shape 1
      {{1},  {2},  {3},  {4},  {5},  {6},  {7},  {8},  {9},  {11}, {13},
       {15}, {16}, {17}, {19}, {23}, {24}, {32}, {55}, {63}, {64}, {65}}}}  // Target shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_1D,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(dynamicInputShapes1D),
                                            ::testing::Values(std::tuple<int, int>{0, 0}),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(true, false),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

const std::vector<std::vector<ov::test::InputShape>> dynamicInputShapes1DI32 = {
    {{{ov::Dimension{1, 70}},  // Dynamic shape 0
      {{1}, {15}, {64}}},      // Target shapes
     {{-1},                    // Dynamic shape 1
      {{1}, {15}, {64}}}}      // Target shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_1D_I32,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(dynamicInputShapes1DI32),
                                            ::testing::Values(std::tuple<int, int>{0, 0}),
                                            ::testing::Values(ElementType::i32),
                                            ::testing::Values(true, false),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"}),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

///// 4D JIT /////
std::vector<std::vector<ov::test::InputShape>> get4DShapesJitStat(int maxBatchDims) {
    std::vector<std::vector<ov::test::InputShape>> result = {};
    if (ov::with_cpu_x86_avx2()) {
        if (maxBatchDims == 2) {
            result = {{{{}, {{18, 2, 2, 1}}},  // Static shapes
                       {{}, {{18, 2, 8}}}},
                      {{{}, {{17, 2, 2, 2}}},  // Static shapes
                       {{}, {{17, 2, 7}}}},
                      {{{}, {{16, 2, 2, 3}}},  // Static shapes
                       {{}, {{16, 2, 6}}}},
                      {{{}, {{15, 2, 2, 4}}},  // Static shapes
                       {{}, {{15, 2, 5}}}},
                      {{{}, {{14, 2, 2, 5}}},  // Static shapes
                       {{}, {{14, 2, 4}}}},
                      {{{}, {{13, 2, 2, 6}}},  // Static shapes
                       {{}, {{13, 2, 3}}}},
                      {{{}, {{12, 2, 2, 7}}},  // Static shapes
                       {{}, {{12, 2, 2}}}},
                      {{{}, {{11, 2, 2, 8}}},  // Static shapes
                       {{}, {{11, 2, 1}}}}};
        } else if (maxBatchDims == 3) {
            result = {{{{}, {{18, 2, 8, 1}}},  // Static shapes
                       {{}, {{18, 2, 8}}}},
                      {{{}, {{17, 2, 7, 2}}},  // Static shapes
                       {{}, {{17, 2, 7}}}},
                      {{{}, {{16, 2, 6, 3}}},  // Static shapes
                       {{}, {{16, 2, 6}}}},
                      {{{}, {{15, 2, 5, 4}}},  // Static shapes
                       {{}, {{15, 2, 5}}}},
                      {{{}, {{14, 2, 4, 5}}},  // Static shapes
                       {{}, {{14, 2, 4}}}},
                      {{{}, {{13, 2, 3, 6}}},  // Static shapes
                       {{}, {{13, 2, 3}}}},
                      {{{}, {{12, 2, 2, 7}}},  // Static shapes
                       {{}, {{12, 2, 2}}}},
                      {{{}, {{11, 2, 1, 8}}},  // Static shapes
                       {{}, {{11, 2, 1}}}}};
        } else {
            throw std::invalid_argument("Invalid test case. Not valid batch dims.");
        }
    }  // AVX2
    if (ov::with_cpu_x86_avx512f()) {
        std::vector<std::vector<ov::test::InputShape>> tmp;
        if (maxBatchDims == 2) {
            tmp = {{{{}, {{19, 4, 2, 9}}},  // Static shapes
                    {{}, {{19, 4, 16}}}},
                   {
                       {{}, {{20, 4, 2, 10}}},  // Static shapes
                       {{}, {{20, 4, 15}}},
                   },
                   {{{}, {{21, 4, 2, 11}}},  // Static shapes
                    {{}, {{21, 4, 14}}}},
                   {
                       {{}, {{22, 4, 2, 12}}},  // Static shapes
                       {{}, {{22, 4, 13}}},
                   },
                   {
                       {{}, {{23, 4, 2, 13}}},  // Static shapes
                       {{}, {{23, 4, 12}}},
                   },
                   {
                       {{}, {{24, 4, 2, 14}}},  // Static shapes
                       {{}, {{24, 4, 11}}},
                   },
                   {
                       {{}, {{25, 4, 2, 15}}},  // Static shapes
                       {{}, {{25, 4, 10}}},
                   },
                   {
                       {{}, {{26, 4, 2, 16}}},  // Static shapes
                       {{}, {{26, 4, 9}}},
                   }};
        } else if (maxBatchDims == 3) {
            tmp = {{{{}, {{19, 4, 16, 9}}},  // Static shapes
                    {{}, {{19, 4, 16}}}},
                   {
                       {{}, {{20, 4, 15, 10}}},  // Static shapes
                       {{}, {{20, 4, 15}}},
                   },
                   {{{}, {{21, 4, 14, 11}}},  // Static shapes
                    {{}, {{21, 4, 14}}}},
                   {
                       {{}, {{22, 4, 13, 12}}},  // Static shapes
                       {{}, {{22, 4, 13}}},
                   },
                   {
                       {{}, {{23, 4, 12, 13}}},  // Static shapes
                       {{}, {{23, 4, 12}}},
                   },
                   {
                       {{}, {{24, 4, 11, 14}}},  // Static shapes
                       {{}, {{24, 4, 11}}},
                   },
                   {
                       {{}, {{25, 4, 10, 15}}},  // Static shapes
                       {{}, {{25, 4, 10}}},
                   },
                   {
                       {{}, {{26, 4, 9, 16}}},  // Static shapes
                       {{}, {{26, 4, 9}}},
                   }};
        } else {
            throw std::invalid_argument("Invalid test case. Not valid batch dims.");
        }
        result.insert(result.end(), tmp.begin(), tmp.end());
    }  // AVX5

    return result;
}

std::vector<std::tuple<int, int>> get4DAxisBatchJitStat(ov::element::Type type, int maxBatchDims) {
    std::vector<std::tuple<int, int>> result = {};
    if (ov::with_cpu_x86_avx512f()) {
        if (type.size() == 4 || type.size() == 2 || type.size() == 1) {
            if (maxBatchDims == 2)
                return std::vector<std::tuple<int, int>>{{3, 0}, {3, 1}, {3, 2}, {2, 0}, {2, 1}, {2, 2}};
            else if (maxBatchDims == 3)
                return std::vector<std::tuple<int, int>>{{3, 3}};
            else
                throw std::invalid_argument("Invalid test case. Not valid batch dims.");
        }
    } else if (ov::with_cpu_x86_avx2()) {
        if (type.size() == 4) {
            if (maxBatchDims == 2)
                return std::vector<std::tuple<int, int>>{{3, 0}, {3, 1}, {3, 2}, {2, 0}, {2, 1}, {2, 2}};
            else if (maxBatchDims == 3)
                return std::vector<std::tuple<int, int>>{{3, 3}};
            else
                throw std::invalid_argument("Invalid test case. Not valid batch dims.");
        } else if (type.size() == 2 || type.size() == 1) {
            if (maxBatchDims == 2)
                return std::vector<std::tuple<int, int>>{{3, 0}, {3, 1}, {3, 2}};
            else if (maxBatchDims == 3)
                return std::vector<std::tuple<int, int>>{{3, 3}};
            else
                throw std::invalid_argument("Invalid test case. Not valid batch dims.");
        }
    }
    return {};
}

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit32,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesJitStat(2)),
                                            ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::f32, 2)),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::ValuesIn(additionalConfig)),
                         GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit16,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesJitStat(2)),
                                            ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::bf16, 2)),
                                            ::testing::Values(ElementType::bf16),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit8,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesJitStat(2)),
                                            ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::i8, 2)),
                                            ::testing::Values(ElementType::i8),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

// batchDims == indicesRank
INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit32_Bmax,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesJitStat(3)),
                                            ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::f32, 3)),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::ValuesIn(additionalConfig)),
                         GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit16_Bmax,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesJitStat(3)),
                                            ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::bf16, 3)),
                                            ::testing::Values(ElementType::bf16),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit8_Bmax,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesJitStat(3)),
                                            ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::i8, 3)),
                                            ::testing::Values(ElementType::i8),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> get4DShapesJitDyn(int maxBatchDims) {
    std::vector<std::vector<ov::test::InputShape>> result = {};
    if (ov::with_cpu_x86_avx2()) {
        if (maxBatchDims == 2) {
            result = {
                {{{ov::Dimension(5, 15), -1, -1, -1},                             // Dynamic shape 0
                  {{8, 2, 2, 1}, {10, 2, 2, 2}, {8, 2, 2, 3}, {10, 2, 2, 4}}},    // Target shapes
                 {{ov::Dimension(4, 16), -1, -1},                                 // Dynamic shape 1
                  {{8, 2, 8}, {10, 2, 7}, {8, 2, 6}, {10, 2, 5}}}},               // Target shapes
                {{{-1, -1, -1, -1},                                               // Dynamic shape 0
                  {{8, 2, 2, 5}, {10, 2, 2, 6}, {8, 2, 2, 7}, {10, 2, 2, 8}}},    // Target shapes
                 {{-1, -1, -1},                                                   // Dynamic shape 1
                  {{8, 2, 4}, {10, 2, 3}, {8, 2, 2}, {10, 2, 1}}}},               // Target shapes
                {{{ov::Dimension(5, 15), -1, -1, -1},                             // Dynamic shape 0
                  {{10, 2, 2, 1}, {10, 2, 2, 2}, {10, 2, 2, 3}, {10, 2, 2, 4}}},  // Target shapes
                 {{10, 2, 5},                                                     // Dynamic shape 1
                  {{10, 2, 5}, {10, 2, 5}, {10, 2, 5}, {10, 2, 5}}}},             // Target shapes
                {{{8, 2, 2, 5},                                                   // Dynamic shape 0
                  {{8, 2, 2, 5}, {8, 2, 2, 5}, {8, 2, 2, 5}, {8, 2, 2, 5}}},      // Target shapes
                 {{-1, -1, -1},                                                   // Dynamic shape 1
                  {{8, 2, 4}, {8, 2, 3}, {8, 2, 2}, {8, 2, 1}}}}                  // Target shapes
            };
        } else if (maxBatchDims == 3) {
            result = {
                {{{ov::Dimension(5, 15), -1, -1, -1},                             // Dynamic shape 0
                  {{8, 2, 8, 1}, {10, 2, 8, 2}, {8, 2, 8, 3}, {10, 2, 5, 4}}},    // Target shapes
                 {{ov::Dimension(4, 16), -1, -1},                                 // Dynamic shape 1
                  {{8, 2, 8}, {10, 2, 8}, {8, 2, 8}, {10, 2, 5}}}},               // Target shapes
                {{{-1, -1, -1, -1},                                               // Dynamic shape 0
                  {{8, 2, 4, 5}, {10, 2, 3, 6}, {8, 2, 2, 7}, {10, 2, 1, 8}}},    // Target shapes
                 {{-1, -1, -1},                                                   // Dynamic shape 1
                  {{8, 2, 4}, {10, 2, 3}, {8, 2, 2}, {10, 2, 1}}}},               // Target shapes
                {{{ov::Dimension(5, 15), -1, -1, -1},                             // Dynamic shape 0
                  {{10, 2, 5, 1}, {10, 2, 5, 2}, {10, 2, 5, 3}, {10, 2, 5, 4}}},  // Target shapes
                 {{10, 2, 5},                                                     // Dynamic shape 1
                  {{10, 2, 5}, {10, 2, 5}, {10, 2, 5}, {10, 2, 5}}}},             // Target shapes
                {{{8, 2, 3, 5},                                                   // Dynamic shape 0
                  {{8, 2, 3, 5}, {8, 2, 3, 5}, {8, 2, 3, 5}, {8, 2, 3, 5}}},      // Target shapes
                 {{-1, -1, -1},                                                   // Dynamic shape 1
                  {{8, 2, 3}, {8, 2, 3}, {8, 2, 3}, {8, 2, 3}}}}                  // Target shapes
            };
        } else {
            throw std::invalid_argument("Invalid test case. Not valid batch dims.");
        }
    }
    if (ov::with_cpu_x86_avx512f()) {
        std::vector<std::vector<ov::test::InputShape>> tmp;
        if (maxBatchDims == 2) {
            tmp = {
                {{{ov::Dimension(5, 15), -1, -1, -1},                                // Dynamic shape 0
                  {{8, 2, 2, 9}, {10, 2, 2, 10}, {8, 2, 2, 11}, {10, 2, 2, 12}}},    // Target shapes
                 {{ov::Dimension(4, 16), -1, -1},                                    // Dynamic shape 1
                  {{8, 2, 16}, {10, 2, 15}, {8, 2, 14}, {10, 2, 13}}}},              // Target shapes
                {{{-1, -1, -1, -1},                                                  // Dynamic shape 0
                  {{8, 2, 2, 13}, {10, 2, 2, 14}, {8, 2, 2, 15}, {10, 2, 2, 16}}},   // Target shapes
                 {{-1, -1, -1},                                                      // Dynamic shape 1
                  {{8, 2, 12}, {10, 2, 11}, {8, 2, 10}, {10, 2, 9}}}},               // Target shapes
                {{{ov::Dimension(5, 15), -1, -1, -1},                                // Dynamic shape 0
                  {{10, 2, 2, 9}, {10, 2, 2, 10}, {10, 2, 2, 11}, {10, 2, 2, 12}}},  // Target shapes
                 {{10, 2, 16},                                                       // Dynamic shape 1
                  {{10, 2, 16}, {10, 2, 16}, {10, 2, 16}, {10, 2, 16}}}},            // Target shapes
                {{{8, 2, 2, 15},                                                     // Dynamic shape 0
                  {{8, 2, 2, 15}, {8, 2, 2, 15}, {8, 2, 2, 15}, {8, 2, 2, 15}}},     // Target shapes
                 {{-1, -1, -1},                                                      // Dynamic shape 1
                  {{8, 2, 12}, {8, 2, 11}, {8, 2, 10}, {8, 2, 9}}}}                  // Target shapes
            };
        } else if (maxBatchDims == 3) {
            tmp = {
                {{{ov::Dimension(5, 15), -1, -1, -1},                                    // Dynamic shape 0
                  {{8, 2, 16, 9}, {10, 2, 15, 10}, {8, 2, 14, 11}, {10, 2, 13, 12}}},    // Target shapes
                 {{ov::Dimension(4, 16), -1, -1},                                        // Dynamic shape 1
                  {{8, 2, 16}, {10, 2, 15}, {8, 2, 14}, {10, 2, 13}}}},                  // Target shapes
                {{{-1, -1, -1, -1},                                                      // Dynamic shape 0
                  {{8, 2, 12, 13}, {10, 2, 11, 14}, {8, 2, 10, 15}, {10, 2, 9, 16}}},    // Target shapes
                 {{-1, -1, -1},                                                          // Dynamic shape 1
                  {{8, 2, 12}, {10, 2, 11}, {8, 2, 10}, {10, 2, 9}}}},                   // Target shapes
                {{{ov::Dimension(5, 15), -1, -1, -1},                                    // Dynamic shape 0
                  {{10, 2, 16, 9}, {10, 2, 16, 10}, {10, 2, 16, 11}, {10, 2, 16, 12}}},  // Target shapes
                 {{10, 2, 16},                                                           // Dynamic shape 1
                  {{10, 2, 16}, {10, 2, 16}, {10, 2, 16}, {10, 2, 16}}}},                // Target shapes
                {{{8, 2, 11, 15},                                                        // Dynamic shape 0
                  {{8, 2, 11, 15}, {8, 2, 11, 15}, {8, 2, 11, 15}, {8, 2, 11, 15}}},     // Target shapes
                 {{-1, -1, -1},                                                          // Dynamic shape 1
                  {{8, 2, 11}, {8, 2, 11}, {8, 2, 11}, {8, 2, 11}}}}                     // Target shapes
            };
        } else {
            throw std::invalid_argument("Invalid test case. Not valid batch dims.");
        }
        result.insert(result.end(), tmp.begin(), tmp.end());
    }

    return result;
}

std::vector<std::tuple<int, int>> get4DAxisBatchJitDyn(ov::element::Type type, int maxBatchDims) {
    std::vector<std::tuple<int, int>> result = {};
    if (ov::with_cpu_x86_avx512f()) {
        if (type.size() == 4 || type.size() == 2 || type.size() == 1) {
            if (maxBatchDims == 2)
                return std::vector<std::tuple<int, int>>{{3, 0}, {3, 1}, {3, 2}};
            else if (maxBatchDims == 3)
                return std::vector<std::tuple<int, int>>{{3, 3}};
            else
                throw std::invalid_argument("Invalid test case. Not valid batch dims.");
        }
    } else if (ov::with_cpu_x86_avx2()) {
        if (type.size() == 4 || type.size() == 2 || type.size() == 1) {
            if (maxBatchDims == 2)
                return std::vector<std::tuple<int, int>>{{3, 0}, {3, 1}, {3, 2}};
            else if (maxBatchDims == 3)
                return std::vector<std::tuple<int, int>>{{3, 3}};
            else
                throw std::invalid_argument("Invalid test case. Not valid batch dims.");
        }
    }
    return {};
}

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit32,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesJitDyn(2)),
                                            ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::f32, 2)),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(isAxisConst),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::ValuesIn(additionalConfig)),
                         GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit16,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesJitDyn(2)),
                                            ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::bf16, 2)),
                                            ::testing::Values(ElementType::bf16),
                                            ::testing::ValuesIn(isAxisConst),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit8,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesJitDyn(2)),
                                            ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::i8, 2)),
                                            ::testing::Values(ElementType::i8),
                                            ::testing::ValuesIn(isAxisConst),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

// batchDims == indicesRank
INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit32_Bmax,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesJitDyn(3)),
                                            ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::f32, 3)),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(isAxisConst),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::ValuesIn(additionalConfig)),
                         GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit16_Bmax,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesJitDyn(3)),
                                            ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::bf16, 3)),
                                            ::testing::Values(ElementType::bf16),
                                            ::testing::ValuesIn(isAxisConst),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit8_Bmax,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesJitDyn(3)),
                                            ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::i8, 3)),
                                            ::testing::Values(ElementType::i8),
                                            ::testing::ValuesIn(isAxisConst),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

///// 4D REFERENCE /////
std::vector<std::vector<ov::test::InputShape>> get4DShapesRefStat(bool maxBatchDims) {
    std::vector<std::vector<ov::test::InputShape>> result = {};
    if (ov::with_cpu_x86_avx2()) {
        if (!maxBatchDims) {
            result = {{{{}, {{10, 2, 9, 9}}},  // Static shapes
                       {{}, {{10, 2, 8}}}},
                      {{{}, {{11, 2, 9, 2}}},  // Static shapes
                       {{}, {{11, 2, 7}}}},
                      {{{}, {{12, 2, 9, 3}}},  // Static shapes
                       {{}, {{12, 2, 6}}}},
                      {{{}, {{13, 2, 9, 4}}},  // Static shapes
                       {{}, {{13, 2, 5}}}},
                      {{{}, {{14, 2, 9, 5}}},  // Static shapes
                       {{}, {{14, 2, 4}}}},
                      {{{}, {{15, 2, 9, 6}}},  // Static shapes
                       {{}, {{15, 2, 3}}}},
                      {{{}, {{16, 2, 9, 7}}},  // Static shapes
                       {{}, {{16, 2, 2}}}},
                      {{{}, {{17, 2, 9, 8}}},  // Static shapes
                       {{}, {{17, 2, 1}}}}};
        } else {
            result = {{{{}, {{10, 8, 2, 39}}},  // Static shapes
                       {{}, {{10, 8}}}},
                      {{{}, {{11, 7, 2, 42}}},  // Static shapes
                       {{}, {{11, 7}}}},
                      {{{}, {{12, 6, 2, 43}}},  // Static shapes
                       {{}, {{12, 6}}}},
                      {{{}, {{13, 5, 2, 44}}},  // Static shapes
                       {{}, {{13, 5}}}},
                      {{{}, {{14, 4, 2, 45}}},  // Static shapes
                       {{}, {{14, 4}}}},
                      {{{}, {{15, 3, 2, 46}}},  // Static shapes
                       {{}, {{15, 3}}}},
                      {{{}, {{16, 2, 2, 47}}},  // Static shapes
                       {{}, {{16, 2}}}},
                      {{{}, {{17, 1, 2, 38}}},  // Static shapes
                       {{}, {{17, 1}}}}};
        }
    }
    if (ov::with_cpu_x86_avx512f()) {
        std::vector<std::vector<ov::test::InputShape>> tmp;
        if (!maxBatchDims) {
            tmp = {{{{}, {{25, 4, 4, 17}}},  // Static shapes
                    {{}, {{25, 4, 16}}}},
                   {
                       {{}, {{24, 4, 4, 18}}},  // Static shapes
                       {{}, {{24, 4, 15}}},
                   },
                   {{{}, {{23, 4, 4, 19}}},  // Static shapes
                    {{}, {{23, 4, 14}}}},
                   {
                       {{}, {{22, 4, 4, 20}}},  // Static shapes
                       {{}, {{22, 4, 13}}},
                   },
                   {
                       {{}, {{21, 4, 4, 21}}},  // Static shapes
                       {{}, {{21, 4, 12}}},
                   },
                   {
                       {{}, {{20, 4, 4, 22}}},  // Static shapes
                       {{}, {{20, 4, 11}}},
                   },
                   {
                       {{}, {{19, 4, 4, 23}}},  // Static shapes
                       {{}, {{19, 4, 10}}},
                   },
                   {
                       {{}, {{18, 4, 4, 24}}},  // Static shapes
                       {{}, {{18, 4, 9}}},
                   }};
        } else {
            tmp = {{{{}, {{25, 16, 4, 65}}},  // Static shapes
                    {{}, {{25, 16}}}},
                   {
                       {{}, {{24, 15, 4, 66}}},  // Static shapes
                       {{}, {{24, 15}}},
                   },
                   {{{}, {{23, 14, 4, 67}}},  // Static shapes
                    {{}, {{23, 14}}}},
                   {
                       {{}, {{22, 13, 4, 68}}},  // Static shapes
                       {{}, {{22, 13}}},
                   },
                   {
                       {{}, {{21, 12, 4, 69}}},  // Static shapes
                       {{}, {{21, 12}}},
                   },
                   {
                       {{}, {{20, 11, 4, 70}}},  // Static shapes
                       {{}, {{20, 11}}},
                   },
                   {
                       {{}, {{19, 10, 4, 71}}},  // Static shapes
                       {{}, {{19, 10}}},
                   },
                   {
                       {{}, {{18, 9, 4, 72}}},  // Static shapes
                       {{}, {{18, 9}}},
                   }};
        }
        result.insert(result.end(), tmp.begin(), tmp.end());
    }

    return result;
}

std::vector<std::tuple<int, int>> get4DAxisBatchRefStat(ov::element::Type type, bool maxBatchDims) {
    std::vector<std::tuple<int, int>> result = {};
    if (ov::with_cpu_x86_avx512f()) {
        if (type.size() == 4) {
            if (!maxBatchDims)
                return std::vector<std::tuple<int, int>>{{1, 0}, {1, 1}, {0, 0}};
            else
                return std::vector<std::tuple<int, int>>{{2, 2}};
        } else if (type.size() == 2 || type.size() == 1) {
            if (!maxBatchDims)
                return std::vector<std::tuple<int, int>>{{0, 0}};
            else
                return std::vector<std::tuple<int, int>>{{2, 2}};
        }
    } else if (ov::with_cpu_x86_avx2()) {
        if (type.size() == 4) {
            if (!maxBatchDims)
                return std::vector<std::tuple<int, int>>{{1, 0}, {1, 1}, {0, 0}};
            else
                return std::vector<std::tuple<int, int>>{{2, 2}};
        } else if (type.size() == 2 || type.size() == 1) {
            if (!maxBatchDims)
                return std::vector<std::tuple<int, int>>{{2, 0}, {2, 1}, {2, 2}, {1, 0}, {1, 1}, {0, 0}};
            else
                return std::vector<std::tuple<int, int>>{{2, 2}};
        }
    }
    return {};
}

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref32,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesRefStat(false)),
                                            ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::f32, false)),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(true),
                                            ::testing::Values(cpuParamsRef),
                                            ::testing::ValuesIn(additionalConfig)),
                         GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref16,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesRefStat(false)),
                                            ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::bf16, false)),
                                            ::testing::Values(ElementType::bf16),
                                            ::testing::Values(true),
                                            ::testing::Values(cpuParamsRef),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref8,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesRefStat(false)),
                                            ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::i8, false)),
                                            ::testing::Values(ElementType::i8),
                                            ::testing::Values(true),
                                            ::testing::Values(cpuParamsRef),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

// batchDims == indicesRank
INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref32_Bmax,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesRefStat(true)),
                                            ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::f32, true)),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(true),
                                            ::testing::Values(cpuParamsRef),
                                            ::testing::ValuesIn(additionalConfig)),
                         GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref16_Bmax,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesRefStat(true)),
                                            ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::bf16, true)),
                                            ::testing::Values(ElementType::bf16),
                                            ::testing::Values(true),
                                            ::testing::Values(cpuParamsRef),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref8_Bmax,
                         GatherLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(get4DShapesRefStat(true)),
                                            ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::i8, true)),
                                            ::testing::Values(ElementType::i8),
                                            ::testing::Values(true),
                                            ::testing::Values(cpuParamsRef),
                                            ::testing::Values(additionalConfig[0])),
                         GatherLayerTestCPU::getTestCaseName);

// InPlace

const std::vector<ov::test::InputShape> shapesInPlace4D_0 = {
    {{}, {{5, 4, 4, 19}}},
    {{5, 4, -1, -1}, {{5, 4, 4, 19}, {5, 4, 4, 25}, {5, 4, 2, 19}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_inplace_4D_0,
                         GatherInPlaceLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(shapesInPlace4D_0),
                                            ::testing::Values(std::vector<int64_t>{2}),
                                            ::testing::Values(0),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"})),
                         GatherInPlaceLayerTestCPU::getTestCaseName);

const std::vector<ov::test::InputShape> shapesInPlace4D_1 = {
    {{}, {{1, 9, 4, 19}}},
    {{1, 9, -1, -1}, {{1, 9, 4, 19}, {1, 9, 4, 25}, {1, 9, 2, 19}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_inplace_4D_1,
                         GatherInPlaceLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(shapesInPlace4D_1),
                                            ::testing::Values(std::vector<int64_t>{-4}, std::vector<int64_t>{5}),
                                            ::testing::Values(1),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"})),
                         GatherInPlaceLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_4D_out_of_range,
                         GatherInPlaceLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(shapesInPlace4D_1),
                                            ::testing::Values(std::vector<int64_t>{10}, std::vector<int64_t>{-15}),
                                            ::testing::Values(1),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_any"})),
                         GatherInPlaceLayerTestCPU::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
