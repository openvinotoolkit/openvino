// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "grid_sample.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"
#include "openvino/op/grid_sample.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string GridSampleLayerTestCPU::getTestCaseName(testing::TestParamInfo<GridSampleLayerTestCPUParams> obj) {
    const auto &[inputShapes, interpolateMode, paddingMode, alignCorners, dataPrecision, gridPrecision, cpuParams,
                 additionalConfig] = obj.param;
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

    result << "interpMode="
           << (interpolateMode == ov::op::v9::GridSample::InterpolationMode::BILINEAR  ? "BILINEAR"
               : interpolateMode == ov::op::v9::GridSample::InterpolationMode::BICUBIC ? "BICUBIC"
                                                                           : "NEAREST")
           << "_";
    result << "padMode="
           << (paddingMode == ov::op::v9::GridSample::PaddingMode::ZEROS    ? "ZEROS"
               : paddingMode == ov::op::v9::GridSample::PaddingMode::BORDER ? "BORDER"
                                                                : "REFLECTION")
           << "_";
    result << "alignCorners=" << (alignCorners ? "True" : "False") << "_";
    result << "dataPrc=" << dataPrecision << "_";
    result << "gridPrc=" << gridPrecision;
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

void GridSampleLayerTestCPU::SetUp() {
    abs_threshold = 0.0005;
    const auto &[inputShapes, interpolateMode, paddingMode, alignCorners, dataPrecision, gridPrecision, cpuParams,
                 additionalConfig] = this->GetParam();
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    targetDevice = ov::test::utils::DEVICE_CPU;
    init_input_shapes(inputShapes);
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    if (intel_cpu::contains_key_value(additionalConfig, {ov::hint::inference_precision.name(), ov::element::bf16})) {
        selectedType = makeSelectedTypeStr(selectedType, ElementType::bf16);
    } else {
        auto execType = dataPrecision == ov::element::i32 ? ov::element::i32 : ov::element::f32;
        selectedType = makeSelectedTypeStr(selectedType, execType);
    }
    if (gridPrecision == ov::element::bf16) {
        rel_threshold = 0.01f;
    }

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(dataPrecision, inputDynamicShapes[0]),
                               std::make_shared<ov::op::v0::Parameter>(gridPrecision, inputDynamicShapes[1])};
    params[0]->set_friendly_name("data");
    params[1]->set_friendly_name("grid");
    ov::op::v9::GridSample::Attributes attributes = {alignCorners, interpolateMode, paddingMode};
    auto gridSample = std::make_shared<ov::op::v9::GridSample>(params[0], params[1], attributes);
    function = makeNgraphFunction(dataPrecision, params, gridSample, "GridSampleCPU");
}

TEST_P(GridSampleLayerTestCPU, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "GridSample");
}

namespace GridSample {

const std::vector<ov::op::v9::GridSample::InterpolationMode>& allInterpolationModes() {
    static const std::vector<ov::op::v9::GridSample::InterpolationMode> modes = {
        ov::op::v9::GridSample::InterpolationMode::BILINEAR,
        ov::op::v9::GridSample::InterpolationMode::BICUBIC,
        ov::op::v9::GridSample::InterpolationMode::NEAREST
    };
    return modes;
}

const std::vector<ov::op::v9::GridSample::InterpolationMode>& armInterpolationModes() {
    static const std::vector<ov::op::v9::GridSample::InterpolationMode> modes = {
        ov::op::v9::GridSample::InterpolationMode::BILINEAR,
        ov::op::v9::GridSample::InterpolationMode::NEAREST
    };
    return modes;
}

const std::vector<ov::op::v9::GridSample::PaddingMode>& allPaddingModes() {
    static const std::vector<ov::op::v9::GridSample::PaddingMode> modes = {
        ov::op::v9::GridSample::PaddingMode::ZEROS,
        ov::op::v9::GridSample::PaddingMode::BORDER,
        ov::op::v9::GridSample::PaddingMode::REFLECTION
    };
    return modes;
}

const std::vector<bool>& alignCornersValues() {
    static const std::vector<bool> values = {true, false};
    return values;
}

const std::vector<std::vector<InputShape>>& getStaticShapes() {
    static const std::vector<std::vector<InputShape>> shapes = {
        {{{}, {{1, 5, 1, 1}}}, {{}, {{1, 1, 1, 2}}}},
        {{{}, {{2, 4, 7, 1}}}, {{}, {{2, 1, 2, 2}}}},
        {{{}, {{3, 3, 3, 3}}}, {{}, {{3, 3, 1, 2}}}},
        {{{}, {{4, 2, 5, 4}}}, {{}, {{4, 2, 2, 2}}}},
        {{{}, {{5, 1, 5, 5}}}, {{}, {{5, 1, 5, 2}}}},
        {{{}, {{4, 2, 4, 6}}}, {{}, {{4, 2, 3, 2}}}},
        {{{}, {{3, 3, 5, 7}}}, {{}, {{3, 7, 1, 2}}}},
        {{{}, {{2, 4, 7, 7}}}, {{}, {{2, 2, 4, 2}}}},
        {{{}, {{2, 5, 8, 8}}}, {{}, {{2, 3, 3, 2}}}},
        {{{}, {{2, 6, 9, 8}}}, {{}, {{2, 2, 5, 2}}}}
    };
    return shapes;
}

const std::vector<std::vector<InputShape>>& getDynamicShapes() {
    static const std::vector<std::vector<InputShape>> shapes = {
        {{{ov::Dimension(1, 15), -1, -1, -1},
          {{1, 1, 1, 1}, {6, 3, 1, 2}, {4, 5, 3, 1}, {2, 7, 2, 2}}},
         {{ov::Dimension(1, 16), -1, -1, -1},
          {{1, 1, 1, 2}, {6, 2, 2, 2}, {4, 1, 3, 2}, {2, 1, 2, 2}}}},
        {{{-1, -1, -1, -1},
          {{1, 2, 1, 5}, {3, 4, 2, 3}, {5, 6, 7, 1}, {7, 8, 2, 4}}},
         {{-1, -1, -1, 2},
          {{1, 2, 4, 2}, {3, 1, 7, 2}, {5, 2, 3, 2}, {7, 1, 5, 2}}}}
    };
    return shapes;
}

const std::vector<CPUSpecificParams>& getCPUInfoForCommon() {
    static const std::vector<CPUSpecificParams> params = {CPUSpecificParams{{}, {}, {"ref"}, "ref"}};
    return params;
}

const std::vector<ov::AnyMap>& additionalConfigs() {
    static const std::vector<ov::AnyMap> configs = {
        {},
        {{ov::hint::inference_precision.name(), ov::element::bf16}}
    };
    return configs;
}

}  // namespace GridSample
}  // namespace test
}  // namespace ov