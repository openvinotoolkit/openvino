// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

typedef std::tuple<std::vector<InputShape>,  // Input shapes
                   std::tuple<bool, int>,    // Is flattened and axis
                   bool,                     // Sorted
                   ElementType,              // Data precision
                   CPUSpecificParams,        // CPU specific params
                   ov::AnyMap                // Additional config
                   >
    UniqueLayerTestCPUParams;

class UniqueLayerTestCPU : public testing::WithParamInterface<UniqueLayerTestCPUParams>,
                           virtual public SubgraphBaseTest,
                           public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<UniqueLayerTestCPUParams> obj) {
        std::vector<InputShape> inputShapes;
        std::tuple<bool, int> flatOrAxis;
        bool sorted;
        ElementType dataPrecision;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;

        std::tie(inputShapes, flatOrAxis, sorted, dataPrecision, cpuParams, additionalConfig) = obj.param;

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

        if (!std::get<0>(flatOrAxis)) {
            result << "axis=" << std::get<1>(flatOrAxis) << "_";
        } else {
            result << "flattened"
                   << "_";
        }
        result << "sorted=" << (sorted ? "True" : "False") << "_";
        result << "dataPrc=" << dataPrecision;
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
        std::tuple<bool, int> flatOrAxis;
        bool sorted, flattened;
        int axis;
        ElementType dataPrecision;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;

        std::tie(inputShapes, flatOrAxis, sorted, dataPrecision, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        targetDevice = ov::test::utils::DEVICE_CPU;
        init_input_shapes(inputShapes);
        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        flattened = std::get<0>(flatOrAxis);

        if (additionalConfig[ov::hint::inference_precision.name()] == ov::element::bf16) {
            selectedType = makeSelectedTypeStr(selectedType, ElementType::bf16);
        } else {
            if (dataPrecision == ElementType::bf16) {
                dataPrecision = ElementType::f32;
            }
            selectedType = makeSelectedTypeStr(selectedType, dataPrecision);
        }

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(dataPrecision, shape));
        }
        params[0]->set_friendly_name("data");
        std::shared_ptr<ov::Node> uniqueNode;
        if (flattened) {
            uniqueNode = std::make_shared<ov::op::v10::Unique>(params[0], sorted);
        } else {
            axis = std::get<1>(flatOrAxis);
            uniqueNode = std::make_shared<ov::op::v10::Unique>(
                params[0],
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape({1}), {axis}),
                sorted);
        }

        function = makeNgraphFunction(dataPrecision, params, uniqueNode, "UniqueCPU");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (funcInput.get_node()->get_friendly_name() == "data") {
                int32_t range = std::accumulate(targetInputStaticShapes[0].begin(), targetInputStaticShapes[0].end(), 1, std::multiplies<>());
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = -range / 2;
                in_data.range = range;
                tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[0], in_data);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(UniqueLayerTestCPU, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CheckPluginRelatedResults(compiledModel, "Unique");
}

namespace {

const std::vector<ElementType> dataPrecisionSmoke = {ElementType::f32, ElementType::i32};
const std::vector<ElementType> dataPrecisionNightly = {ElementType::bf16, ElementType::i8};

std::vector<std::tuple<bool, int>> flatOrAxis{{true, 0}, {false, 0}, {false, 1}, {false, -1}};

std::vector<bool> sorted{true, false};

std::vector<ov::AnyMap> additionalConfig = {{{ov::hint::inference_precision(ov::element::f32)}},
                                            {{ov::hint::inference_precision(ov::element::bf16)}}};

std::vector<CPUSpecificParams> getCPUInfo() {
    std::vector<CPUSpecificParams> resCPUParams;
    resCPUParams.push_back(CPUSpecificParams{{}, {}, {"ref"}, "ref"});
    return resCPUParams;
}

std::vector<std::vector<InputShape>> statShapes1D = {
    {{{}, {{1}}}},   // Static shapes
    {{{}, {{5}}}},   // Static shapes
    {{{}, {{8}}}},   // Static shapes
    {{{}, {{16}}}},  // Static shapes
    {{{}, {{32}}}},  // Static shapes
    {{{}, {{64}}}},  // Static shapes
    {{{}, {{99}}}},  // Static shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_static_1D,
                         UniqueLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(statShapes1D),
                                            ::testing::ValuesIn(std::vector<std::tuple<bool, int>>{{true, 0},
                                                                                                   {false, 0}}),
                                            ::testing::ValuesIn(sorted),
                                            ::testing::ValuesIn(dataPrecisionSmoke),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         UniqueLayerTestCPU::getTestCaseName);

std::vector<std::vector<InputShape>> getStaticShapes() {
    std::vector<std::vector<InputShape>> result = {
        {{{}, {{1, 1, 1}}}},     // Static shapes
        {{{}, {{1, 2, 1}}}},     // Static shapes
        {{{}, {{1, 1, 3}}}},     // Static shapes
        {{{}, {{2, 2, 1}}}},     // Static shapes
        {{{}, {{1, 4, 1}}}},     // Static shapes
        {{{}, {{1, 5, 1}}}},     // Static shapes
        {{{}, {{3, 2, 1}}}},     // Static shapes
        {{{}, {{1, 1, 7}}}},     // Static shapes
        {{{}, {{2, 2, 2}}}},     // Static shapes
        {{{}, {{1, 8, 1}}}},     // Static shapes
        {{{}, {{3, 3, 1, 1}}}},  // Static shapes
        {{{}, {{1, 5, 2, 1}}}},  // Static shapes
        {{{}, {{1, 1, 11}}}},    // Static shapes
        {{{}, {{32, 35, 37}}}},  // Static shapes
        {{{}, {{2, 3, 2}}}},     // Static shapes
        {{{}, {{1, 1, 13}}}},    // Static shapes
        {{{}, {{7, 1, 2}}}},     // Static shapes
        {{{}, {{3, 5, 1}}}},     // Static shapes
        {{{}, {{4, 2, 2}}}},     // Static shapes
        {{{}, {{1, 17, 1}}}},    // Static shapes
        {{{}, {{3, 2, 3, 1}}}},  // Static shapes
        {{{}, {{8, 16, 32}}}},   // Static shapes
        {{{}, {{37, 19, 11}}}},  // Static shapes
        {{{}, {{1, 19, 1}}}},    // Static shapes
        {{{}, {{2, 5, 2}}}},     // Static shapes
        {{{}, {{1, 3, 7}}}},     // Static shapes
        {{{}, {{11, 1, 2}}}},    // Static shapes
        {{{}, {{1, 1, 23}}}},    // Static shapes
        {{{}, {{4, 3, 2}}}},     // Static shapes
        {{{}, {{5, 1, 5}}}},     // Static shapes
        {{{}, {{100, 1, 1}}}},   // Static shapes
        {{{}, {{5, 5, 5}}}}      // Static shapes
    };

    return result;
}

INSTANTIATE_TEST_SUITE_P(smoke_static,
                         UniqueLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(getStaticShapes()),
                                            ::testing::ValuesIn(flatOrAxis),
                                            ::testing::ValuesIn(sorted),
                                            ::testing::ValuesIn(dataPrecisionSmoke),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         UniqueLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_static,
                         UniqueLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(getStaticShapes()),
                                            ::testing::ValuesIn(flatOrAxis),
                                            ::testing::ValuesIn(sorted),
                                            ::testing::ValuesIn(dataPrecisionNightly),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         UniqueLayerTestCPU::getTestCaseName);

const std::vector<std::vector<InputShape>> dynamicInSapes = {
    {{{ov::Dimension(1, 15), -1, -1, -1},                             // Dynamic shape
      {{1, 1, 1, 1}, {6, 3, 1, 2}, {4, 5, 3, 1}, {2, 7, 2, 2}}}},     // Target shapes
    {{{-1, -1, -1, -1},                                               // Dynamic shape
      {{1, 2, 1, 5}, {3, 4, 2, 3}, {5, 6, 7, 1}, {7, 8, 2, 4}}}},     // Target shapes
    {{{ov::Dimension(2, 15), -1, -1, -1},                             // Dynamic shape
      {{8, 3, 3, 3}, {6, 5, 2, 5}, {4, 7, 1, 11}, {2, 9, 3, 4}}}},    // Target shapes
    {{{3, 4, 4, 5},                                                   // Dynamic shape
      {{3, 4, 4, 5}, {3, 4, 4, 5}, {3, 4, 4, 5}, {3, 4, 4, 5}}}},     // Target shapes
    {{{-1, -1, -1, -1},                                               // Dynamic shape
      {{1, 2, 1, 13}, {3, 4, 7, 2}, {5, 6, 3, 5}, {7, 8, 4, 4}}}},    // Target shapes
    {{{-1, -1, -1, -1},                                               // Dynamic shape
      {{2, 11, 1, 17}, {4, 9, 6, 3}, {6, 7, 7, 3}, {8, 3, 2, 11}}}},  // Target shapes
    {{{3, -1, -1, -1},                                                // Dynamic shape
      {{3, 2, 1, 23}, {3, 4, 3, 8}, {3, 6, 5, 5}, {3, 8, 31, 1}}}},   // Target shapes
    {{{-1, 3, -1, -1},                                                // Dynamic shape
      {{8, 3, 8, 4}, {6, 3, 33, 1}, {4, 3, 8, 6}, {2, 3, 8, 8}}}}     // Target shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic,
                         UniqueLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(dynamicInSapes),
                                            ::testing::ValuesIn(flatOrAxis),
                                            ::testing::ValuesIn(sorted),
                                            ::testing::ValuesIn(dataPrecisionSmoke),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         UniqueLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_dynamic,
                         UniqueLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(dynamicInSapes),
                                            ::testing::ValuesIn(flatOrAxis),
                                            ::testing::ValuesIn(sorted),
                                            ::testing::ValuesIn(dataPrecisionNightly),
                                            ::testing::ValuesIn(getCPUInfo()),
                                            ::testing::Values(additionalConfig[0])),
                         UniqueLayerTestCPU::getTestCaseName);
}  // namespace