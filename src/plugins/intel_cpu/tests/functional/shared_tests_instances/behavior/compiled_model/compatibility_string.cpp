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

}  // namespace
