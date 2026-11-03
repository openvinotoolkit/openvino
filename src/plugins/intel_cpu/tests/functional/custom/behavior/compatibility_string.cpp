// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "openvino/core/version.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
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

    const size_t descriptor_offset = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t);
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

std::string get_field(const std::string& requirements, const std::string& key) {
    const auto begin = requirements.find(key + "=");
    if (begin == std::string::npos) {
        return {};
    }
    const auto value_begin = begin + key.size() + 1;
    return requirements.substr(value_begin, requirements.find(';', value_begin) - value_begin);
}

std::string replace_field(const std::string& requirements, const std::string& key, const std::string& value) {
    const auto begin = requirements.find(key + "=") + key.size() + 1;
    const auto end = requirements.find(';', begin);
    return requirements.substr(0, begin) + value + (end == std::string::npos ? "" : requirements.substr(end));
}

std::string replace_blob_requirements(const std::string& blob, const std::string& requirements) {
    constexpr size_t size_offset = sizeof(uint64_t) + sizeof(uint32_t);
    uint64_t old_size = 0;
    std::memcpy(&old_size, blob.data() + size_offset, sizeof(old_size));
    const uint64_t new_size = requirements.size();
    std::string result = blob.substr(0, size_offset);
    result.append(reinterpret_cast<const char*>(&new_size), sizeof(new_size));
    result.append(requirements);
    result.append(blob.substr(size_offset + sizeof(uint64_t) + old_size));
    return result;
}

std::shared_ptr<ov::Model> make_rms_model() {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, 128});
    auto power = std::make_shared<ov::op::v1::Power>(data, ov::op::v0::Constant::create(ov::element::f32, {}, {2.f}));
    auto mean = std::make_shared<ov::op::v1::ReduceMean>(power,
                                                         ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
                                                         true);
    auto eps = std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, {}, {1e-5f}));
    auto norm = std::make_shared<ov::op::v1::Divide>(data, std::make_shared<ov::op::v0::Sqrt>(eps));
    auto gamma = ov::op::v0::Constant::create(ov::element::f32, {128}, std::vector<float>(128, 0.5f));
    auto result = std::make_shared<ov::op::v1::Multiply>(norm, gamma);
    return std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{data});
}

ov::CompatibilityCheck check_requirements(ov::Core& core, const std::string& requirements) {
    return core.get_property(ov::test::utils::DEVICE_CPU,
                             ov::compatibility_check,
                             {{ov::runtime_requirements.name(), requirements}});
}

class RuntimeRequirementsCPU : public CompatibilityStringCPU {
public:
    void SetUp() override {
        CompatibilityStringCPU::SetUp();
        compiled_model = core.compile_model(model, ov::test::utils::DEVICE_CPU);
        requirements = compiled_model.get_property(ov::runtime_requirements);
    }

    ov::Core core;
    ov::CompiledModel compiled_model;
    std::string requirements;
};

TEST_F(RuntimeRequirementsCPU, DescriptorFormat) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    EXPECT_EQ(requirements.rfind("meta=2.0;", 0), 0u) << requirements;
    const std::string version = std::to_string(OPENVINO_VERSION_MAJOR) + "." + std::to_string(OPENVINO_VERSION_MINOR) +
                                "." + std::to_string(OPENVINO_VERSION_PATCH);
    EXPECT_EQ(get_field(requirements, "ov"), version) << requirements;
    EXPECT_FALSE(get_field(requirements, "arch").empty()) << requirements;
    for (const auto* key : {"ops=", "caps="}) {
        EXPECT_NE(requirements.find(key), std::string::npos) << requirements;
    }
    EXPECT_LE(requirements.size(), 4096u);
}

TEST_F(RuntimeRequirementsCPU, PlainModelHasNoCapabilityRequirements) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    EXPECT_EQ(get_field(requirements, "caps"), "") << requirements;
}

TEST_F(RuntimeRequirementsCPU, OlderProducerVersionSupported) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    EXPECT_EQ(check_requirements(core, replace_field(requirements, "ov", "2026.0.0")),
              ov::CompatibilityCheck::SUPPORTED);
}

TEST_F(RuntimeRequirementsCPU, NewerProducerVersionUnsupported) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const std::string newer = std::to_string(OPENVINO_VERSION_MAJOR + 1) + ".0.0";
    EXPECT_EQ(check_requirements(core, replace_field(requirements, "ov", newer)), ov::CompatibilityCheck::UNSUPPORTED);
}

TEST_F(RuntimeRequirementsCPU, MismatchesUnsupported) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const std::vector<std::string> incompatible = {
        replace_field(requirements, "arch", "unknown_arch"),
        replace_field(requirements, "ops", "cpu_plugin_opset::UnknownOp"),
        replace_field(requirements, "ops", "ie_internal_opset::FullyConnected@2"),
        replace_field(requirements, "ops", "malformed"),
        replace_field(requirements, "caps", "unknown_capability"),
        replace_field(requirements, "ov", "2026.x.0"),
        replace_field(requirements, "meta", "1.0"),
        requirements.substr(0, requirements.find(";caps=")),
        "meta=1.0;ov=2026.4.0;isa=Intel AVX-512",
    };
    for (const auto& req : incompatible) {
        EXPECT_EQ(check_requirements(core, req), ov::CompatibilityCheck::UNSUPPORTED) << req;
    }
}

TEST_F(RuntimeRequirementsCPU, KnownInternalOpSupported) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    EXPECT_EQ(check_requirements(core, replace_field(requirements, "ops", "ie_internal_opset::FullyConnected+RMS")),
              ov::CompatibilityCheck::SUPPORTED);
}

TEST_F(RuntimeRequirementsCPU, ImportFollowsRequirements) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::stringstream blob_stream;
    compiled_model.export_model(blob_stream);
    const auto blob = blob_stream.str();

    const auto older = replace_field(requirements, "ov", "2026.0.0");
    std::stringstream older_blob(replace_blob_requirements(blob, older));
    ov::CompiledModel imported;
    OV_ASSERT_NO_THROW(imported = core.import_model(older_blob, ov::test::utils::DEVICE_CPU));
    EXPECT_EQ(imported.get_property(ov::runtime_requirements), older);

    const auto newer = replace_field(requirements, "ov", std::to_string(OPENVINO_VERSION_MAJOR + 1) + ".0.0");
    std::stringstream newer_blob(replace_blob_requirements(blob, newer));
    OV_EXPECT_THROW((void)core.import_model(newer_blob, ov::test::utils::DEVICE_CPU),
                    ov::Exception,
                    testing::HasSubstr("produced by a newer OpenVINO version"));

    const auto blob_with_tensor = replace_blob_requirements(blob, older);
    ov::Tensor tensor(ov::element::u8, ov::Shape{blob_with_tensor.size()});
    std::memcpy(tensor.data(), blob_with_tensor.data(), blob_with_tensor.size());
    OV_ASSERT_NO_THROW((void)core.import_model(tensor, ov::test::utils::DEVICE_CPU));
}

TEST_F(CompatibilityStringCPU, RMSRequiresAvx2) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
#if defined(OPENVINO_ARCH_X86_64)
    if (!ov::with_cpu_x86_avx2()) {
        GTEST_SKIP() << "RMS is decomposed without avx2";
    }
    ov::Core core;
    auto compiled_model = core.compile_model(make_rms_model(), ov::test::utils::DEVICE_CPU);
    const auto requirements = compiled_model.get_property(ov::runtime_requirements);
    EXPECT_NE(get_field(requirements, "ops").find("RMS"), std::string::npos) << requirements;
    EXPECT_EQ(get_field(requirements, "caps"), "avx2") << requirements;
    EXPECT_EQ(check_requirements(core, requirements), ov::CompatibilityCheck::SUPPORTED);
#else
    GTEST_SKIP() << "x86_64 only";
#endif
}

TEST_F(CompatibilityStringCPU, CacheKeyIsSharedByCpusOfSameArchitecture) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    const auto caching_properties = core.get_property(ov::test::utils::DEVICE_CPU, ov::internal::caching_properties);
    EXPECT_NE(std::find(caching_properties.begin(), caching_properties.end(), ov::device::architecture.name()),
              caching_properties.end());
    EXPECT_EQ(std::find(caching_properties.begin(), caching_properties.end(), ov::device::full_name.name()),
              caching_properties.end());
}

}  // namespace
