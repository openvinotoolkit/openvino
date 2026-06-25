// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <sstream>

#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace {

class CompatibilityStringGPU : public ::testing::Test {
public:
    std::shared_ptr<ov::Model> model;

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        model = ov::test::utils::make_conv_pool_relu();
    }
};

// The compiled model exposes a non-empty runtime requirements descriptor.
TEST_F(CompatibilityStringGPU, RuntimeRequirementsIsSupportedAndNonEmpty) {
    ov::Core core;
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU));

    auto supported = compiled_model.get_property(ov::supported_properties);
    ASSERT_NE(std::find(supported.begin(), supported.end(), ov::runtime_requirements.name()), supported.end());

    std::string requirements;
    OV_ASSERT_NO_THROW(requirements = compiled_model.get_property(ov::runtime_requirements));
    ASSERT_FALSE(requirements.empty());
}

// The plugin advertises compatibility_check among its supported properties.
TEST_F(CompatibilityStringGPU, CompatibilityCheckListedInSupportedProperties) {
    ov::Core core;
    auto supported = core.get_property(ov::test::utils::DEVICE_GPU, ov::supported_properties);
    ASSERT_NE(std::find(supported.begin(), supported.end(), ov::compatibility_check.name()), supported.end());
}

// A descriptor generated on this device is reported as SUPPORTED.
TEST_F(CompatibilityStringGPU, GenerateAndCheckSupported) {
    ov::Core core;
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU));

    const auto requirements = compiled_model.get_property(ov::runtime_requirements);

    ov::CompatibilityCheck result = ov::CompatibilityCheck::NOT_APPLICABLE;
    OV_ASSERT_NO_THROW(result = core.get_property(ov::test::utils::DEVICE_GPU,
                                                  ov::compatibility_check,
                                                  {{ov::runtime_requirements.name(), requirements}}));
    ASSERT_EQ(result, ov::CompatibilityCheck::SUPPORTED);
}

// A tampered descriptor is reported as UNSUPPORTED.
TEST_F(CompatibilityStringGPU, TamperedRequirementsUnsupported) {
    ov::Core core;
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU));

    auto requirements = compiled_model.get_property(ov::runtime_requirements);
    requirements += ";tampered=1";

    ov::CompatibilityCheck result = ov::CompatibilityCheck::NOT_APPLICABLE;
    OV_ASSERT_NO_THROW(result = core.get_property(ov::test::utils::DEVICE_GPU,
                                                  ov::compatibility_check,
                                                  {{ov::runtime_requirements.name(), requirements}}));
    ASSERT_EQ(result, ov::CompatibilityCheck::UNSUPPORTED);
}

// An empty or missing requirements argument yields NOT_APPLICABLE.
TEST_F(CompatibilityStringGPU, EmptyAndNoArgumentNotApplicable) {
    ov::Core core;

    ov::CompatibilityCheck empty_result = ov::CompatibilityCheck::SUPPORTED;
    OV_ASSERT_NO_THROW(empty_result = core.get_property(ov::test::utils::DEVICE_GPU,
                                                        ov::compatibility_check,
                                                        {{ov::runtime_requirements.name(), std::string{}}}));
    ASSERT_EQ(empty_result, ov::CompatibilityCheck::NOT_APPLICABLE);

    ov::CompatibilityCheck missing_result = ov::CompatibilityCheck::SUPPORTED;
    OV_ASSERT_NO_THROW(missing_result = core.get_property(ov::test::utils::DEVICE_GPU, ov::compatibility_check));
    ASSERT_EQ(missing_result, ov::CompatibilityCheck::NOT_APPLICABLE);
}

// The descriptor survives an export/import round-trip of the compiled blob.
TEST_F(CompatibilityStringGPU, ExportImportPreservesRequirements) {
    ov::Core core;
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU));
    const auto requirements = compiled_model.get_property(ov::runtime_requirements);

    std::stringstream blob;
    OV_ASSERT_NO_THROW(compiled_model.export_model(blob));

    ov::CompiledModel imported;
    OV_ASSERT_NO_THROW(imported = core.import_model(blob, ov::test::utils::DEVICE_GPU));

    std::string imported_requirements;
    OV_ASSERT_NO_THROW(imported_requirements = imported.get_property(ov::runtime_requirements));
    ASSERT_EQ(requirements, imported_requirements);
}

// Mirror of the ONNX Runtime OrtCompiledModelCompatibility states that the GPU plugin
// can actually produce. OpenVINO's ov::CompatibilityCheck has 3 states and the GPU
// plugin never reports a "runs but suboptimal" outcome, so the ORT
// EP_SUPPORTED_PREFER_RECOMPILATION state is intentionally not mirrored here.
enum class OrtCompatibility {
    EP_NOT_APPLICABLE = 0,
    EP_SUPPORTED_OPTIMAL,
    EP_UNSUPPORTED,
};

// The mapping an OpenVINO EP would apply when answering ORT's
// ValidateCompiledModelCompatibilityInfo / GetModelCompatibilityForEpDevices.
OrtCompatibility to_ort_compatibility(ov::CompatibilityCheck check) {
    switch (check) {
    case ov::CompatibilityCheck::SUPPORTED:
        return OrtCompatibility::EP_SUPPORTED_OPTIMAL;
    case ov::CompatibilityCheck::UNSUPPORTED:
        return OrtCompatibility::EP_UNSUPPORTED;
    case ov::CompatibilityCheck::NOT_APPLICABLE:
    default:
        return OrtCompatibility::EP_NOT_APPLICABLE;
    }
}

// End-to-end simulation of the ONNX Runtime EP-compatibility handshake:
//   1. Producer side (OrtEp::GetCompiledModelCompatibilityInfo): compile a model and
//      read ov::runtime_requirements to obtain the opaque descriptor string.
//   2. The descriptor crosses an opaque serialization boundary (stand-in for the
//      EPContext node attribute). It must survive byte-for-byte.
//   3. Validator side (OrtEpFactory::ValidateCompiledModelCompatibilityInfo): on a
//      fresh Core, WITHOUT importing or compiling anything, query
//      ov::compatibility_check and map the result to ORT's 4-state enum.
TEST_F(CompatibilityStringGPU, OrtFactoryValidationFlow) {
    std::string descriptor;
    {
        // Producer side: a separate Core, as if running in the compiling process.
        ov::Core producer_core;
        ov::CompiledModel compiled_model;
        OV_ASSERT_NO_THROW(compiled_model = producer_core.compile_model(model, ov::test::utils::DEVICE_GPU));
        OV_ASSERT_NO_THROW(descriptor = compiled_model.get_property(ov::runtime_requirements));
        ASSERT_FALSE(descriptor.empty());
    }

    // Opaque transfer of the descriptor (e.g. embedded in an EPContext node).
    std::stringstream channel;
    channel << descriptor;
    const std::string received = channel.str();
    ASSERT_EQ(descriptor, received) << "descriptor must round-trip byte-for-byte";

    // Validator side: a fresh Core with no session/import, mimicking the EP factory.
    ov::Core validator_core;

    ov::CompatibilityCheck matched = ov::CompatibilityCheck::NOT_APPLICABLE;
    OV_ASSERT_NO_THROW(matched = validator_core.get_property(ov::test::utils::DEVICE_GPU,
                                                             ov::compatibility_check,
                                                             {{ov::runtime_requirements.name(), received}}));
    EXPECT_EQ(to_ort_compatibility(matched), OrtCompatibility::EP_SUPPORTED_OPTIMAL);

    ov::CompatibilityCheck tampered = ov::CompatibilityCheck::NOT_APPLICABLE;
    OV_ASSERT_NO_THROW(tampered = validator_core.get_property(ov::test::utils::DEVICE_GPU,
                                                             ov::compatibility_check,
                                                             {{ov::runtime_requirements.name(), received + ";tampered=1"}}));
    EXPECT_EQ(to_ort_compatibility(tampered), OrtCompatibility::EP_UNSUPPORTED);

    ov::CompatibilityCheck empty = ov::CompatibilityCheck::SUPPORTED;
    OV_ASSERT_NO_THROW(empty = validator_core.get_property(ov::test::utils::DEVICE_GPU,
                                                           ov::compatibility_check,
                                                           {{ov::runtime_requirements.name(), std::string{}}}));
    EXPECT_EQ(to_ort_compatibility(empty), OrtCompatibility::EP_NOT_APPLICABLE);
}

}  // namespace
