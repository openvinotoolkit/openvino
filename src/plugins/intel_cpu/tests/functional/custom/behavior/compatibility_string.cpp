// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "openvino/core/version.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
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

[[maybe_unused]] std::shared_ptr<ov::Model> make_rms_model() {
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
    EXPECT_EQ(get_field(requirements, "ov"), std::string(ov::get_openvino_version().buildNumber)) << requirements;
    EXPECT_FALSE(get_field(requirements, "arch").empty()) << requirements;
    EXPECT_NE(requirements.find(";caps="), std::string::npos) << requirements;
    EXPECT_EQ(requirements.find("ops="), std::string::npos) << requirements;
    EXPECT_LE(requirements.size(), 4096u);
}

TEST_F(RuntimeRequirementsCPU, PlainModelHasNoCapabilityRequirements) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto fp32_model =
        core.compile_model(model, ov::test::utils::DEVICE_CPU, ov::hint::inference_precision(ov::element::f32));
    const auto fp32_requirements = fp32_model.get_property(ov::runtime_requirements);
    EXPECT_EQ(get_field(fp32_requirements, "caps"), "") << fp32_requirements;
}

std::vector<std::string> other_versions(const std::string& version) {
    return {"2026.0.0", std::to_string(OPENVINO_VERSION_MAJOR + 1) + ".0.0", version + "-other_build"};
}

TEST_F(RuntimeRequirementsCPU, OtherProducerVersionUnsupported) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    for (const auto& version : other_versions(get_field(requirements, "ov"))) {
        EXPECT_EQ(check_requirements(core, replace_field(requirements, "ov", version)),
                  ov::CompatibilityCheck::UNSUPPORTED)
            << version;
    }
}

TEST_F(RuntimeRequirementsCPU, MismatchesUnsupported) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const std::vector<std::string> incompatible = {
        replace_field(requirements, "arch", "unknown_arch"),
        replace_field(requirements, "caps", "unknown_capability"),
        replace_field(requirements, "meta", "1.0"),
        requirements + ";ops=ie_internal_opset::RMS",
        requirements.substr(0, requirements.find(";caps=")),
        "meta=1.0;ov=2026.4.0;isa=Intel AVX-512",
    };
    for (const auto& req : incompatible) {
        EXPECT_EQ(check_requirements(core, req), ov::CompatibilityCheck::UNSUPPORTED) << req;
    }
}

TEST_F(RuntimeRequirementsCPU, ImportFollowsRequirements) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::stringstream blob_stream;
    compiled_model.export_model(blob_stream);
    const auto blob = blob_stream.str();

    std::stringstream same_blob(blob);
    ov::CompiledModel imported;
    OV_ASSERT_NO_THROW(imported = core.import_model(same_blob, ov::test::utils::DEVICE_CPU));
    EXPECT_EQ(imported.get_property(ov::runtime_requirements), requirements);

    for (const auto& version : other_versions(get_field(requirements, "ov"))) {
        const auto other_blob = replace_blob_requirements(blob, replace_field(requirements, "ov", version));
        std::stringstream stream(other_blob);
        OV_EXPECT_THROW((void)core.import_model(stream, ov::test::utils::DEVICE_CPU),
                        ov::Exception,
                        testing::HasSubstr("produced by OpenVINO"));

        ov::Tensor tensor(ov::element::u8, ov::Shape{other_blob.size()});
        std::memcpy(tensor.data(), other_blob.data(), other_blob.size());
        OV_EXPECT_THROW((void)core.import_model(tensor, ov::test::utils::DEVICE_CPU),
                        ov::Exception,
                        testing::HasSubstr("produced by OpenVINO"));
    }
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
    EXPECT_EQ(get_field(requirements, "caps"), "ov.avx2") << requirements;
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

#if defined(OPENVINO_ARCH_X86_64)
// Golden blobs: compiled model blobs stored in the repository as data/blobs/*.blob, produced once by an earlier build.
// They check that the current runtime still reads such blobs: blob layout, requirements descriptor and the serialized
// graph with its internal operations. The producer version in the descriptor never equals the version of the build
// under test, so the test replaces it with the current one, everything else is used as it was stored.
//
// To regenerate the blobs only for an intentional change of the blob format or of the internal operations, on a host
// with f16 support: OV_CPU_GOLDEN_BLOBS_GENERATE=<dir> ov_cpu_func_tests --gtest_filter=GoldenBlobsCPU.*, then copy
// <dir>/*.blob to data/blobs/.
std::vector<float> make_values(size_t size, float scale) {
    std::vector<float> values(size);
    for (size_t i = 0; i < size; ++i) {
        values[i] = scale * (static_cast<float>((i * 7) % 23) - 11.f) / 11.f;
    }
    return values;
}

std::shared_ptr<ov::Model> make_conv_relu_model() {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 16, 16});
    auto weights = ov::op::v0::Constant::create(ov::element::f32, {4, 3, 3, 3}, make_values(4 * 3 * 3 * 3, 0.5f));
    auto conv = std::make_shared<ov::op::v1::Convolution>(data,
                                                          weights,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::CoordinateDiff{1, 1},
                                                          ov::Strides{1, 1});
    return std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Relu>(conv)},
                                       ov::ParameterVector{data});
}

// becomes a snippets Subgraph
std::shared_ptr<ov::Model> make_eltwise_chain_model() {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 16, 16});
    auto scale = ov::op::v0::Constant::create(ov::element::f32, {1, 8, 1, 1}, make_values(8, 2.f));
    auto shift = ov::op::v0::Constant::create(ov::element::f32, {1, 8, 1, 1}, make_values(8, 0.5f));
    auto mul = std::make_shared<ov::op::v1::Multiply>(data, scale);
    auto add = std::make_shared<ov::op::v1::Add>(mul, shift);
    auto relu = std::make_shared<ov::op::v0::Relu>(add);
    auto sqrt = std::make_shared<ov::op::v0::Sqrt>(relu);
    auto result = std::make_shared<ov::op::v1::Multiply>(sqrt, data);
    return std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{data});
}

// u8 weights with f16 decompression, becomes FullyConnectedCompressed
std::shared_ptr<ov::Model> make_compressed_fc_model() {
    constexpr size_t size = 128;
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, size});
    std::vector<uint8_t> raw(size * size);
    for (size_t i = 0; i < raw.size(); ++i) {
        raw[i] = static_cast<uint8_t>((i * 13) % 251);
    }
    auto weights =
        std::make_shared<ov::op::v0::Convert>(ov::op::v0::Constant::create(ov::element::u8, {size, size}, raw),
                                              ov::element::f16);
    auto zero_point = std::make_shared<ov::op::v0::Convert>(
        ov::op::v0::Constant::create(ov::element::u8, {size, 1}, std::vector<uint8_t>(size, 128)),
        ov::element::f16);
    auto scale = ov::op::v0::Constant::create(ov::element::f16, {size, 1}, std::vector<float>(size, 0.01f));
    auto decompressed =
        std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v1::Subtract>(weights, zero_point), scale);
    auto converted = std::make_shared<ov::op::v0::Convert>(decompressed, ov::element::f32);
    return std::make_shared<ov::Model>(
        ov::OutputVector{std::make_shared<ov::op::v0::MatMul>(data, converted, false, true)},
        ov::ParameterVector{data});
}

struct GoldenBlob {
    std::string name;
    std::function<std::shared_ptr<ov::Model>()> make_model;
    ov::AnyMap config;
    std::string caps;  // capabilities the stored descriptor must list
    float tolerance;
};

std::vector<GoldenBlob> golden_blobs() {
    const ov::AnyMap f32 = {ov::hint::inference_precision(ov::element::f32)};
    return {
        {"conv_relu_f32", make_conv_relu_model, f32, "", 1e-4f},
        {"conv_relu_f16",
         make_conv_relu_model,
         {ov::hint::inference_precision(ov::element::f16)},
         "dnnl.avx2,f16",
         1e-2f},
        {"eltwise_chain_f32", make_eltwise_chain_model, f32, "dnnl.avx2", 1e-4f},
        {"rms_f32", make_rms_model, f32, "ov.avx2", 1e-4f},
        {"compressed_fc_u8_f32", make_compressed_fc_model, f32, "dnnl.avx2,ov.avx2", 1e-3f},
    };
}

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::string blob_requirements(const std::string& blob) {
    constexpr size_t size_offset = sizeof(uint64_t) + sizeof(uint32_t);
    uint64_t size = 0;
    std::memcpy(&size, blob.data() + size_offset, sizeof(size));
    return blob.substr(size_offset + sizeof(uint64_t), size);
}

ov::Tensor make_input(const ov::Output<const ov::Node>& input) {
    ov::Tensor tensor(input.get_element_type(), input.get_shape());
    const auto values = make_values(tensor.get_size(), 1.f);
    std::copy(values.begin(), values.end(), tensor.data<float>());
    return tensor;
}

ov::Tensor infer(ov::CompiledModel& compiled_model) {
    auto request = compiled_model.create_infer_request();
    request.set_input_tensor(make_input(compiled_model.input()));
    request.infer();
    return request.get_output_tensor();
}

void expect_close(const ov::Tensor& actual, const ov::Tensor& expected, float tolerance) {
    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    for (size_t i = 0; i < actual.get_size(); ++i) {
        const float e = expected.data<float>()[i];
        ASSERT_NEAR(actual.data<float>()[i], e, tolerance * std::max(1.f, std::abs(e))) << "element " << i;
    }
}

TEST(GoldenBlobsCPU, StoredBlobsAreImportedOrRejectedByCapabilities) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const char* generate_dir = std::getenv("OV_CPU_GOLDEN_BLOBS_GENERATE");
    ov::Core core;
    for (const auto& golden : golden_blobs()) {
        SCOPED_TRACE(golden.name);
        if (generate_dir != nullptr) {
            std::ofstream out(std::string(generate_dir) + "/" + golden.name + ".blob", std::ios::binary);
            core.compile_model(golden.make_model(), ov::test::utils::DEVICE_CPU, golden.config).export_model(out);
            continue;
        }

        const auto path = ov::test::utils::getExecutableDirectory() + "/cpu_golden_blobs/" + golden.name + ".blob";
        const auto blob = read_file(path);
        ASSERT_FALSE(blob.empty()) << "cannot read " << path;
        const auto stored = blob_requirements(blob);
        EXPECT_EQ(stored.rfind("meta=2.0;", 0), 0u) << stored;
        EXPECT_EQ(get_field(stored, "arch"), "x86_64") << stored;
        EXPECT_EQ(get_field(stored, "caps"), golden.caps) << stored;

        const auto current = replace_field(stored, "ov", ov::get_openvino_version().buildNumber);
        std::stringstream stream(replace_blob_requirements(blob, current));
        if (check_requirements(core, current) == ov::CompatibilityCheck::UNSUPPORTED) {
            // the host lacks a capability of the blob, it is rejected before the graph is read
            OV_EXPECT_THROW((void)core.import_model(stream, ov::test::utils::DEVICE_CPU, golden.config),
                            ov::Exception,
                            testing::HasSubstr("host lacks required capability"));
            continue;
        }
        ov::CompiledModel imported;
        OV_ASSERT_NO_THROW(imported = core.import_model(stream, ov::test::utils::DEVICE_CPU, golden.config));
        auto fresh = core.compile_model(golden.make_model(), ov::test::utils::DEVICE_CPU, golden.config);
        expect_close(infer(imported), infer(fresh), golden.tolerance);
    }
}
#endif  // OPENVINO_ARCH_X86_64

}  // namespace
