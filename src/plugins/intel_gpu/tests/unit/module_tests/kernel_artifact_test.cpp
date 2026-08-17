// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/kernel_builder.hpp"

#include <gtest/gtest.h>

#include <array>

using namespace cldnn;

namespace {

class recording_kernel_builder final : public kernel_builder {
public:
    using kernel_builder::build_kernels;

    void build_kernels(const void* source,
                       size_t source_size,
                       KernelFormat source_format,
                       const std::string& options,
                       std::vector<kernel::ptr>&) const override {
        payload = source;
        payload_size = source_size;
        format = source_format;
        build_options = options;
    }

    mutable const void* payload = nullptr;
    mutable size_t payload_size = 0;
    mutable KernelFormat format = KernelFormat::SOURCE;
    mutable std::string build_options;
};

}  // namespace

TEST(kernel_artifact, formats_have_distinct_semantics) {
    constexpr std::array formats{KernelFormat::SOURCE,
                                 KernelFormat::NATIVE_BIN,
                                 KernelFormat::SPIRV,
                                 KernelFormat::DRIVER_PIPELINE_CACHE};

    for (size_t lhs = 0; lhs < formats.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < formats.size(); ++rhs) {
            EXPECT_NE(formats[lhs], formats[rhs]);
        }
    }
}

TEST(kernel_artifact, legacy_builder_adapter_preserves_build_inputs) {
    const std::array<uint32_t, 2> payload{0x07230203, 0};
    kernel_artifact artifact;
    artifact.payload = payload.data();
    artifact.payload_size = sizeof(payload);
    artifact.format = KernelFormat::SPIRV;
    artifact.entry_point = "main";
    artifact.build_options = "-DTEST=1";
    artifact.stable_id = 42;
    artifact.backend_environment = "test-device:test-driver";
    artifact.specialization_constants.push_back({7, 16});
    artifact.serialization.portable = true;

    recording_kernel_builder builder;
    std::vector<kernel::ptr> kernels;
    builder.build_kernels(artifact, kernels);

    EXPECT_EQ(builder.payload, artifact.payload);
    EXPECT_EQ(builder.payload_size, artifact.payload_size);
    EXPECT_EQ(builder.format, KernelFormat::SPIRV);
    EXPECT_EQ(builder.build_options, artifact.build_options);
    EXPECT_EQ(artifact.entry_point, "main");
    EXPECT_EQ(artifact.stable_id, 42u);
    ASSERT_EQ(artifact.specialization_constants.size(), 1u);
    EXPECT_EQ(artifact.specialization_constants.front().id, 7u);
    EXPECT_EQ(artifact.serialization.version, kernel_artifact_serialization_info::current_version);
    EXPECT_TRUE(artifact.serialization.portable);
}
