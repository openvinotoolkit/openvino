// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace {

class CompatibilityStringCPU : public ::testing::Test {
public:
    std::shared_ptr<ov::Model> model;

    void SetUp() override {
        model = ov::test::utils::make_conv_pool_relu();
    }
};

TEST_F(CompatibilityStringCPU, RuntimeRequirementsIsSupportedAndNonEmpty) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_CPU));

    auto supported = compiled_model.get_property(ov::supported_properties);
    ASSERT_NE(std::find(supported.begin(), supported.end(), ov::runtime_requirements.name()), supported.end());

    std::string requirements;
    OV_ASSERT_NO_THROW(requirements = compiled_model.get_property(ov::runtime_requirements));
    ASSERT_FALSE(requirements.empty());
    std::cout << "[ INFO     ] CPU ov::runtime_requirements = " << requirements << std::endl;
}

TEST_F(CompatibilityStringCPU, CompatibilityCheckListedInSupportedProperties) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    auto supported = core.get_property(ov::test::utils::DEVICE_CPU, ov::supported_properties);
    ASSERT_NE(std::find(supported.begin(), supported.end(), ov::compatibility_check.name()), supported.end());
}

TEST_F(CompatibilityStringCPU, GenerateAndCheckSupported) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_CPU));

    const auto requirements = compiled_model.get_property(ov::runtime_requirements);

    ov::CompatibilityCheck result = ov::CompatibilityCheck::NOT_APPLICABLE;
    OV_ASSERT_NO_THROW(result = core.get_property(ov::test::utils::DEVICE_CPU,
                                                  ov::compatibility_check,
                                                  {{ov::runtime_requirements.name(), requirements}}));
    ASSERT_EQ(result, ov::CompatibilityCheck::SUPPORTED);
}

TEST_F(CompatibilityStringCPU, TamperedRequirementsUnsupported) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_CPU));

    auto requirements = compiled_model.get_property(ov::runtime_requirements);
    requirements += ";tampered=1";

    ov::CompatibilityCheck result = ov::CompatibilityCheck::NOT_APPLICABLE;
    OV_ASSERT_NO_THROW(result = core.get_property(ov::test::utils::DEVICE_CPU,
                                                  ov::compatibility_check,
                                                  {{ov::runtime_requirements.name(), requirements}}));
    ASSERT_EQ(result, ov::CompatibilityCheck::UNSUPPORTED);
}

TEST_F(CompatibilityStringCPU, EmptyAndNoArgumentNotApplicable) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;

    ov::CompatibilityCheck empty_result = ov::CompatibilityCheck::SUPPORTED;
    OV_ASSERT_NO_THROW(empty_result = core.get_property(ov::test::utils::DEVICE_CPU,
                                                        ov::compatibility_check,
                                                        {{ov::runtime_requirements.name(), std::string{}}}));
    ASSERT_EQ(empty_result, ov::CompatibilityCheck::NOT_APPLICABLE);

    ov::CompatibilityCheck missing_result = ov::CompatibilityCheck::SUPPORTED;
    OV_ASSERT_NO_THROW(missing_result = core.get_property(ov::test::utils::DEVICE_CPU, ov::compatibility_check));
    ASSERT_EQ(missing_result, ov::CompatibilityCheck::NOT_APPLICABLE);
}

TEST_F(CompatibilityStringCPU, ExportImportPreservesRequirements) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_CPU));
    const auto requirements = compiled_model.get_property(ov::runtime_requirements);

    std::stringstream blob;
    OV_ASSERT_NO_THROW(compiled_model.export_model(blob));

    ov::CompiledModel imported;
    OV_ASSERT_NO_THROW(imported = core.import_model(blob, ov::test::utils::DEVICE_CPU));

    std::string imported_requirements;
    OV_ASSERT_NO_THROW(imported_requirements = imported.get_property(ov::runtime_requirements));
    ASSERT_EQ(requirements, imported_requirements);
}

TEST_F(CompatibilityStringCPU, DescriptorBlockIsMagicGuardedInBlob) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_CPU));

    std::stringstream blob;
    OV_ASSERT_NO_THROW(compiled_model.export_model(blob));
    const std::string data = blob.str();

    constexpr uint64_t expected_magic = 0x4F564350555F5252ULL;  // "OVCPU_RR" in ASCII
    ASSERT_GE(data.size(), sizeof(expected_magic));

    uint64_t magic = 0;
    std::memcpy(&magic, data.data(), sizeof(magic));
    ASSERT_EQ(magic, expected_magic);
}

TEST_F(CompatibilityStringCPU, ImportRejectsCorruptedDescriptorHeader) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_CPU));

    std::stringstream good_blob;
    OV_ASSERT_NO_THROW(compiled_model.export_model(good_blob));
    const std::string original = good_blob.str();

    constexpr size_t magic_offset = 0;
    constexpr size_t version_offset = magic_offset + sizeof(uint64_t);
    ASSERT_GE(original.size(), version_offset + sizeof(uint32_t));

    std::stringstream blob(original);
    ov::CompiledModel imported;
    OV_ASSERT_NO_THROW(imported = core.import_model(blob, ov::test::utils::DEVICE_CPU));

    std::string magic_corrupted = original;
    magic_corrupted[magic_offset] ^= 0xFF;
    std::stringstream magic_corrupted_blob(magic_corrupted);
    EXPECT_THROW((void)core.import_model(magic_corrupted_blob, ov::test::utils::DEVICE_CPU), ov::Exception);

    std::string version_corrupted = original;
    const uint32_t bad_version = 0xFFFFFFFFu;
    std::memcpy(&version_corrupted[version_offset], &bad_version, sizeof(bad_version));
    std::stringstream version_corrupted_blob(version_corrupted);
    EXPECT_THROW((void)core.import_model(version_corrupted_blob, ov::test::utils::DEVICE_CPU), ov::Exception);
}

TEST_F(CompatibilityStringCPU, ImportRejectsMismatchedRuntimeRequirements) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_CPU));
    const auto requirements = compiled_model.get_property(ov::runtime_requirements);
    ASSERT_FALSE(requirements.empty());

    std::stringstream good_blob;
    OV_ASSERT_NO_THROW(compiled_model.export_model(good_blob));
    std::string data = good_blob.str();

    const size_t descriptor_offset =
        sizeof(ov::CacheMode) + sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t);
    ASSERT_GT(data.size(), descriptor_offset);

    std::stringstream blob(data);
    ov::CompiledModel imported;
    OV_ASSERT_NO_THROW(imported = core.import_model(blob, ov::test::utils::DEVICE_CPU));

    data[descriptor_offset] ^= 0x01;
    std::stringstream corrupted_blob(data);
    EXPECT_THROW((void)core.import_model(corrupted_blob, ov::test::utils::DEVICE_CPU), ov::Exception);
}

enum class ONNXRuntimeCompatibility {
    EP_NOT_APPLICABLE = 0,
    EP_SUPPORTED_OPTIMAL,
    EP_UNSUPPORTED,
};

ONNXRuntimeCompatibility to_onnx_runtime_compatibility(ov::CompatibilityCheck check) {
    switch (check) {
        case ov::CompatibilityCheck::SUPPORTED:
            return ONNXRuntimeCompatibility::EP_SUPPORTED_OPTIMAL;
        case ov::CompatibilityCheck::UNSUPPORTED:
            return ONNXRuntimeCompatibility::EP_UNSUPPORTED;
        case ov::CompatibilityCheck::NOT_APPLICABLE:
        default:
            return ONNXRuntimeCompatibility::EP_NOT_APPLICABLE;
    }
}

TEST_F(CompatibilityStringCPU, ONNXRuntimeFactoryValidationFlow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::string descriptor;

    ov::Core producer_core;
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = producer_core.compile_model(model, ov::test::utils::DEVICE_CPU));
    OV_ASSERT_NO_THROW(descriptor = compiled_model.get_property(ov::runtime_requirements));
    ASSERT_FALSE(descriptor.empty());

    std::stringstream channel;
    channel << descriptor;
    const std::string received = channel.str();
    ASSERT_EQ(descriptor, received) << "descriptor must round-trip byte-for-byte";

    ov::Core validator_core;
    ov::CompatibilityCheck matched = ov::CompatibilityCheck::NOT_APPLICABLE;
    OV_ASSERT_NO_THROW(matched = validator_core.get_property(ov::test::utils::DEVICE_CPU,
                                                             ov::compatibility_check,
                                                             {{ov::runtime_requirements.name(), received}}));
    EXPECT_EQ(to_onnx_runtime_compatibility(matched), ONNXRuntimeCompatibility::EP_SUPPORTED_OPTIMAL);

    ov::CompatibilityCheck tampered = ov::CompatibilityCheck::NOT_APPLICABLE;
    OV_ASSERT_NO_THROW(tampered = validator_core.get_property(ov::test::utils::DEVICE_CPU,
                                                              ov::compatibility_check,
                                                              {{ov::runtime_requirements.name(), received + ";tampered=1"}}));
    EXPECT_EQ(to_onnx_runtime_compatibility(tampered), ONNXRuntimeCompatibility::EP_UNSUPPORTED);

    ov::CompatibilityCheck empty = ov::CompatibilityCheck::SUPPORTED;
    OV_ASSERT_NO_THROW(empty = validator_core.get_property(ov::test::utils::DEVICE_CPU,
                                                           ov::compatibility_check,
                                                           {{ov::runtime_requirements.name(), std::string{}}}));
    EXPECT_EQ(to_onnx_runtime_compatibility(empty), ONNXRuntimeCompatibility::EP_NOT_APPLICABLE);
}

}  // namespace
