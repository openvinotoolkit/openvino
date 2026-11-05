// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/common/npu.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "zero_backend.hpp"

namespace ov {
namespace test {
namespace behavior {

// NPU_DISABLE_IDLE_MEMORY_PRUNING switches a driver-wide context option on and off, it is not a per-compiled-model
// setting. Every test which enables it must therefore disable it again before returning, otherwise the NPU keeps its
// idle memory optimizations turned off for every test running afterwards in the same process.
class DisableIdleMemoryPruningNPU : public OVPluginTestBase,
                                    public testing::WithParamInterface<std::string /*target device*/> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::string>& obj) {
        std::ostringstream result;
        result << "targetDevice=" << obj.param;
        return result.str();
    }

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        target_device = this->GetParam();
        APIBaseTest::SetUp();

        model = ov::test::utils::make_conv_pool_relu();

        auto backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
        _supported = backend->isContextExtSupported();

        // ov::Core::set_property only reaches the plugin once the plugin has been instantiated, otherwise the value is
        // merely cached inside the core. Query a property first so that every set_property below is really forwarded to
        // the NPU plugin.
        OV_ASSERT_NO_THROW(core.get_property(target_device, ov::supported_properties));
    }

    void TearDown() override {
        if (_supported) {
            // Always leave the driver context with the idle memory optimizations enabled.
            core.set_property(target_device, ov::intel_npu::disable_idle_memory_prunning(false));
            EXPECT_FALSE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));
        }
        APIBaseTest::TearDown();
    }

protected:
    // Enables the pruning-disable switch and immediately turns it back off, running `body` in between while the
    // switch is active.
    template <typename Body>
    void withIdleMemoryPruningDisabled(Body&& body) {
        core.set_property(target_device, ov::intel_npu::disable_idle_memory_prunning(true));
        ASSERT_TRUE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));

        body();

        core.set_property(target_device, ov::intel_npu::disable_idle_memory_prunning(false));
        ASSERT_FALSE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));
    }

    void inferOnce(ov::CompiledModel& compiledModel) {
        ov::InferRequest inferRequest;
        OV_ASSERT_NO_THROW(inferRequest = compiledModel.create_infer_request());
        OV_ASSERT_NO_THROW(inferRequest.infer());
    }

    ov::Core core;
    std::shared_ptr<ov::Model> model;
    bool _supported = false;
};

TEST_P(DisableIdleMemoryPruningNPU, SupportMatchesContextExtensionAvailability) {
    std::vector<ov::PropertyName> supportedProperties;
    OV_ASSERT_NO_THROW(supportedProperties = core.get_property(target_device, ov::supported_properties));

    const auto propertyIt = std::find(supportedProperties.cbegin(),
                                      supportedProperties.cend(),
                                      ov::intel_npu::disable_idle_memory_prunning.name());

    if (!_supported) {
        ASSERT_EQ(propertyIt, supportedProperties.cend());
        return;
    }

    ASSERT_NE(propertyIt, supportedProperties.cend());
    ASSERT_TRUE(propertyIt->is_mutable());
}

TEST_P(DisableIdleMemoryPruningNPU, SetPropertyIsRejectedWhenContextExtensionIsMissing) {
    if (_supported) {
        GTEST_SKIP() << "The context extension is available, the property is expected to be usable.";
    }

    OV_EXPECT_THROW_HAS_SUBSTRING(core.set_property(target_device, ov::intel_npu::disable_idle_memory_prunning(true)),
                                  ov::Exception,
                                  "Unsupported configuration key");
    OV_EXPECT_THROW_HAS_SUBSTRING(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning),
                                  ov::Exception,
                                  "Unsupported configuration key");
    OV_EXPECT_THROW_HAS_SUBSTRING(
        core.compile_model(model, target_device, {ov::intel_npu::disable_idle_memory_prunning(true)}),
        ov::Exception,
        "Unsupported configuration key");
}

TEST_P(DisableIdleMemoryPruningNPU, SetPropertyEnablesAndDisablesTheSwitch) {
    if (!_supported) {
        GTEST_SKIP() << "The driver does not expose the context extension.";
    }

    withIdleMemoryPruningDisabled([] {});

    // Setting the same value twice in a row must stay a no-op instead of failing.
    OV_ASSERT_NO_THROW(core.set_property(target_device, ov::intel_npu::disable_idle_memory_prunning(false)));
    ASSERT_FALSE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));
}

TEST_P(DisableIdleMemoryPruningNPU, SetPropertyAcceptsStringValues) {
    if (!_supported) {
        GTEST_SKIP() << "The driver does not expose the context extension.";
    }

    OV_ASSERT_NO_THROW(core.set_property(target_device, {{ov::intel_npu::disable_idle_memory_prunning.name(), "YES"}}));
    ASSERT_TRUE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));

    OV_ASSERT_NO_THROW(core.set_property(target_device, {{ov::intel_npu::disable_idle_memory_prunning.name(), "NO"}}));
    ASSERT_FALSE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));
}

TEST_P(DisableIdleMemoryPruningNPU, CompileAndInferWhilePruningIsDisabledThroughSetProperty) {
    if (!_supported) {
        GTEST_SKIP() << "The driver does not expose the context extension.";
    }

    withIdleMemoryPruningDisabled([this] {
        ov::CompiledModel compiledModel;
        OV_ASSERT_NO_THROW(compiledModel = core.compile_model(model, target_device));
        inferOnce(compiledModel);
    });
}

TEST_P(DisableIdleMemoryPruningNPU, CompileModelWithThePropertyUpdatesThePluginGlobally) {
    if (!_supported) {
        GTEST_SKIP() << "The driver does not expose the context extension.";
    }

    ov::CompiledModel compiledModel;
    OV_ASSERT_NO_THROW(
        compiledModel = core.compile_model(model, target_device, {ov::intel_npu::disable_idle_memory_prunning(true)}));

    // The property is applied on the plugin itself, not only for the duration of the compilation call.
    ASSERT_TRUE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));

    // Turn it back off right away and check that the already compiled model stays usable.
    OV_ASSERT_NO_THROW(core.set_property(target_device, ov::intel_npu::disable_idle_memory_prunning(false)));
    ASSERT_FALSE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));

    inferOnce(compiledModel);

    // Compiling with the property explicitly disabled must keep the plugin value at false.
    OV_ASSERT_NO_THROW(
        compiledModel = core.compile_model(model, target_device, {ov::intel_npu::disable_idle_memory_prunning(false)}));
    ASSERT_FALSE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));
    inferOnce(compiledModel);
}

TEST_P(DisableIdleMemoryPruningNPU, IsNotExposedByTheCompiledModel) {
    if (!_supported) {
        GTEST_SKIP() << "The driver does not expose the context extension.";
    }

    ov::CompiledModel compiledModel;
    OV_ASSERT_NO_THROW(
        compiledModel = core.compile_model(model, target_device, {ov::intel_npu::disable_idle_memory_prunning(true)}));
    OV_ASSERT_NO_THROW(core.set_property(target_device, ov::intel_npu::disable_idle_memory_prunning(false)));

    // The switch belongs to the plugin/driver context, a compiled model does not carry it.
    std::vector<ov::PropertyName> supportedProperties;
    OV_ASSERT_NO_THROW(supportedProperties = compiledModel.get_property(ov::supported_properties));
    ASSERT_EQ(std::find(supportedProperties.cbegin(),
                        supportedProperties.cend(),
                        ov::intel_npu::disable_idle_memory_prunning.name()),
              supportedProperties.cend());
}

TEST_P(DisableIdleMemoryPruningNPU, ImportModelWithThePropertyUpdatesThePluginGlobally) {
    if (!_supported) {
        GTEST_SKIP() << "The driver does not expose the context extension.";
    }

    std::stringstream blobStream;
    OV_ASSERT_NO_THROW(core.compile_model(model, target_device).export_model(blobStream));

    // Import from a stream.
    {
        ov::CompiledModel importedModel;
        OV_ASSERT_NO_THROW(
            importedModel =
                core.import_model(blobStream, target_device, {ov::intel_npu::disable_idle_memory_prunning(true)}));
        ASSERT_TRUE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));

        OV_ASSERT_NO_THROW(core.set_property(target_device, ov::intel_npu::disable_idle_memory_prunning(false)));
        ASSERT_FALSE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));

        inferOnce(importedModel);
    }

    // Import from a tensor.
    {
        auto blob = blobStream.str();
        const ov::Tensor blobTensor(ov::element::u8, ov::Shape{blob.size()}, blob.data());

        ov::CompiledModel importedModel;
        OV_ASSERT_NO_THROW(
            importedModel =
                core.import_model(blobTensor, target_device, {ov::intel_npu::disable_idle_memory_prunning(true)}));
        ASSERT_TRUE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));

        OV_ASSERT_NO_THROW(core.set_property(target_device, ov::intel_npu::disable_idle_memory_prunning(false)));
        ASSERT_FALSE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));

        inferOnce(importedModel);
    }
}

TEST_P(DisableIdleMemoryPruningNPU, ImportAndInferWhilePruningIsDisabledThroughSetProperty) {
    if (!_supported) {
        GTEST_SKIP() << "The driver does not expose the context extension.";
    }

    std::stringstream blobStream;
    OV_ASSERT_NO_THROW(core.compile_model(model, target_device).export_model(blobStream));

    withIdleMemoryPruningDisabled([this, &blobStream] {
        ov::CompiledModel importedModel;
        OV_ASSERT_NO_THROW(importedModel = core.import_model(blobStream, target_device, {}));
        inferOnce(importedModel);
    });
}

TEST_P(DisableIdleMemoryPruningNPU, QueryModelWithThePropertyUpdatesThePluginGlobally) {
    if (!_supported) {
        GTEST_SKIP() << "The driver does not expose the context extension.";
    }

    OV_ASSERT_NO_THROW(core.query_model(model, target_device, {ov::intel_npu::disable_idle_memory_prunning(true)}));
    ASSERT_TRUE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));

    OV_ASSERT_NO_THROW(core.set_property(target_device, ov::intel_npu::disable_idle_memory_prunning(false)));
    ASSERT_FALSE(core.get_property(target_device, ov::intel_npu::disable_idle_memory_prunning));
}

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         DisableIdleMemoryPruningNPU,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<DisableIdleMemoryPruningNPU>);

}  // namespace behavior
}  // namespace test
}  // namespace ov
