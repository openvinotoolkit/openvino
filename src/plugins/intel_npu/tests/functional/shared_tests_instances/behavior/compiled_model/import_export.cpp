// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/import_export.hpp"

#include <common_test_utils/test_constants.hpp>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"

using namespace ov::test::behavior;
namespace {
const std::vector<ov::element::Type> modelTypes = {
    ov::element::f16,
    ov::element::f32,
};

const std::vector<ov::Shape> modelShapes = {{1, 2, 5, 5}};

const std::vector<ov::AnyMap> compiledModelConfigs = {{}};

std::vector<ov::element::Type_t> convertModelTypes = []() -> std::vector<ov::element::Type_t> {
    std::vector<ov::element::Type_t> convertedModelTypes;
    for (auto&& modelType : modelTypes) {
        convertedModelTypes.push_back(modelType);
    }
    return convertedModelTypes;
}();

auto heteroCompiledModelConfigs = []() -> std::vector<ov::AnyMap> {
    std::vector<ov::AnyMap> heteroPluginConfigs(compiledModelConfigs.size());
    for (auto it = compiledModelConfigs.cbegin(); it != compiledModelConfigs.cend(); ++it) {
        auto&& distance = it - compiledModelConfigs.cbegin();
        heteroPluginConfigs.at(distance) = {ov::device::priorities(ov::test::utils::DEVICE_NPU),
                                            ov::device::properties(ov::test::utils::DEVICE_NPU, *it)};
    }
    return heteroPluginConfigs;
}();

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVCompiledGraphImportExportTest,
                         ::testing::Combine(::testing::ValuesIn(convertModelTypes),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompiledGraphImportExportTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVCompiledGraphImportExportTest,
                         ::testing::Combine(::testing::ValuesIn(convertModelTypes),
                                            ::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                                              ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(heteroCompiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompiledGraphImportExportTest>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVClassCompiledModelImportExportTestP,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelImportExportTestP>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelImportExportTestP,
                         ::testing::Values(std::string(ov::test::utils::DEVICE_NPU),
                                           "HETERO:" + std::string(ov::test::utils::DEVICE_NPU)),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelImportExportTestP>);

#if defined(ENABLE_INTEL_CPU) && ENABLE_INTEL_CPU

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_HETERO_CPU,
                         OVClassCompiledModelImportExportTestP,
                         ::testing::Values("HETERO:" + std::string(ov::test::utils::DEVICE_NPU) + ",CPU"));
#endif

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelGraphUniqueNodeNamesTest,
                         ::testing::Combine(::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(modelShapes),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompiledModelGraphUniqueNodeNamesTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVCompiledModelGraphUniqueNodeNamesTest,
                         ::testing::Combine(::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(modelShapes),
                                            ::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                                              ov::test::utils::DEVICE_NPU)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompiledModelGraphUniqueNodeNamesTest>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVExecGraphSerializationTest,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVExecGraphSerializationTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVExecGraphSerializationTest,
                         ::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                           ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVExecGraphSerializationTest>);

}  // namespace
