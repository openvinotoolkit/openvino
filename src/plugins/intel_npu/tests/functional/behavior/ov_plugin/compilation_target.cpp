// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

// Matches the [ INFO ] convention used elsewhere in these tests (see core_integration.cpp).
#define NPU_TEST_LOG std::cout << "[          ] [ LOG ] "

namespace ov {
namespace test {
namespace behavior {

// End-to-end coverage for ov::offline_compilation_targets / ov::compilation_target through the real
// ov::Core::compile_model path (as opposed to the PluginPropertyManager-level unit tests in
// tests/functional/internal/compiler_adapter/compile_impl.cpp).
class CompilationTargetNPU : public OVPluginTestBase, public testing::WithParamInterface<std::string> {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        target_device = GetParam();
        model = ov::test::utils::make_multi_single_conv();
        APIBaseTest::SetUp();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<std::string>& obj) {
        return "targetDevice=" + obj.param;
    }

protected:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
};

// Every known platform must be enumerable through a real ov::Core, each with a non-empty device_ids.
TEST_P(CompilationTargetNPU, EnumeratesOfflineCompilationTargets) {
    std::vector<ov::CompilationTarget> targets;
    OV_ASSERT_NO_THROW(targets = core.get_property(target_device, ov::offline_compilation_targets));

    NPU_TEST_LOG << "offline_compilation_targets returned " << targets.size() << " target(s):" << std::endl;
    std::vector<std::string> platforms;
    for (const auto& target : targets) {
        NPU_TEST_LOG << "  " << target << std::endl;
        EXPECT_FALSE(target.device_ids.empty())
            << "Target for platform '" << target.platform << "' has no device IDs";
        platforms.push_back(target.platform);
    }
    for (const auto& expectedPlatform : {ov::intel_npu::Platform::NPU3720,
                                         ov::intel_npu::Platform::NPU4000,
                                         ov::intel_npu::Platform::NPU5010,
                                         ov::intel_npu::Platform::NPU5020,
                                         ov::intel_npu::Platform::NPU6010}) {
        EXPECT_NE(std::find(platforms.begin(), platforms.end(), std::string(expectedPlatform)), platforms.end())
            << "Platform '" << expectedPlatform << "' is missing from the offline compilation targets";
    }
}

// A platform query argument must narrow the enumeration to just that platform's target.
TEST_P(CompilationTargetNPU, EnumeratesOfflineCompilationTargetsFilteredByPlatform) {
    std::vector<ov::CompilationTarget> targets;
    OV_ASSERT_NO_THROW(
        targets = core.get_property(target_device,
                                    ov::offline_compilation_targets,
                                    {{ov::intel_npu::platform.name(), std::string(ov::intel_npu::Platform::NPU6010)}}));

    NPU_TEST_LOG << "offline_compilation_targets filtered by NPU_PLATFORM=6010 returned " << targets.size()
                 << " target(s)" << std::endl;
    ASSERT_EQ(targets.size(), 1u);
    ASSERT_EQ(targets.front().platform, std::string(ov::intel_npu::Platform::NPU6010));
}

// A single-SKU target compiles through the real plugin regardless of the physical device present -
// same PLUGIN-compiler cross-compilation path exercised by CrossCompilationNPU.
TEST_P(CompilationTargetNPU, CompilesModelForASingleSkuTarget) {
    std::vector<ov::CompilationTarget> targets;
    OV_ASSERT_NO_THROW(targets = core.get_property(target_device, ov::offline_compilation_targets));
    auto targetIt = std::find_if(targets.begin(), targets.end(), [](const ov::CompilationTarget& t) {
        return t.platform == std::string(ov::intel_npu::Platform::NPU4000);
    });
    ASSERT_NE(targetIt, targets.end());

    NPU_TEST_LOG << "Compiling for compilation_target = " << *targetIt << std::endl;
    OV_ASSERT_NO_THROW(core.compile_model(model, target_device, {ov::compilation_target(*targetIt)}));
}

// ov::compilation_target may be combined with an explicit platform that agrees with it.
TEST_P(CompilationTargetNPU, AcceptsMatchingExplicitPlatform) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU4000), {0x643E}};

    NPU_TEST_LOG << "Compiling for compilation_target = " << target << " with matching explicit platform"
                 << std::endl;
    OV_ASSERT_NO_THROW(
        core.compile_model(model, target_device, {ov::compilation_target(target), ov::intel_npu::platform(target.platform)}));
}

// A conflicting explicit platform must be rejected rather than silently overriding the target.
TEST_P(CompilationTargetNPU, ThrowsOnConflictingExplicitPlatform) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU4000), {0x643E}};

    NPU_TEST_LOG << "Compiling for compilation_target = " << target
                 << " with conflicting explicit platform = " << ov::intel_npu::Platform::NPU6010 << std::endl;
    OV_EXPECT_THROW_HAS_SUBSTRING(
        core.compile_model(model,
                           target_device,
                           {ov::compilation_target(target),
                            ov::intel_npu::platform(std::string(ov::intel_npu::Platform::NPU6010))}),
        ov::Exception,
        "conflicts with");
}

// NPU6010 ships as 3-tile and 4-tile SKUs. Resolving that is now delegated to the compiler (see
// ICompilerAdapter::resolve_compilation_target_bundles) rather than guessed by the plugin; a VCL
// build that resolves it into more than one bundle must still be rejected here, since multi-blob
// packaging is not implemented yet. (Verified against a real VCL build with resolution support,
// which does report 2 bundles for NPU6010 and correctly gets rejected below.)
TEST_P(CompilationTargetNPU, ThrowsForAKnownMultiSkuPlatformUntilMultiBlobPackagingShips) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU6010), {0xD71D}};

    NPU_TEST_LOG << "Compiling for multi-SKU compilation_target = " << target << std::endl;
    OV_EXPECT_THROW_HAS_SUBSTRING(core.compile_model(model, target_device, {ov::compilation_target(target)}),
                                  ov::Exception,
                                  "not yet supported");
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         CompilationTargetNPU,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         CompilationTargetNPU::getTestCaseName);

}  // namespace behavior
}  // namespace test
}  // namespace ov
