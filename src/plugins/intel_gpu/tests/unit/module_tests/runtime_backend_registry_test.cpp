// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/runtime_backend_registry.hpp"

#include <gtest/gtest.h>

using namespace cldnn;

TEST(runtime_backend_registry, default_backend_is_compiled_and_first) {
    const auto& backends = runtime_backend_registry::compiled_backends();

    ASSERT_FALSE(backends.empty());
    EXPECT_EQ(backends.front().runtime_type, get_default_runtime_type());
    EXPECT_EQ(runtime_backend_registry::default_backend().runtime_type, get_default_runtime_type());
}

TEST(runtime_backend_registry, tagged_device_identity_round_trips_for_every_backend) {
    for (const auto& backend : runtime_backend_registry::compiled_backends()) {
        const auto tagged_id = runtime_backend_registry::make_device_id(backend.runtime_type, "0.1");
        runtime_types parsed_runtime = get_default_runtime_type();
        std::string backend_device_id;

        ASSERT_TRUE(runtime_backend_registry::parse_device_id(tagged_id, parsed_runtime, backend_device_id));
        EXPECT_EQ(parsed_runtime, backend.runtime_type);
        EXPECT_EQ(backend_device_id, "0.1");
        EXPECT_EQ(tagged_id, std::string(backend.name) + "_0.1");
    }
}

TEST(runtime_backend_registry, public_identity_preserves_established_default_runtime_ids) {
    const auto default_runtime = runtime_backend_registry::default_backend().runtime_type;
    for (const auto& backend : runtime_backend_registry::compiled_backends()) {
        const auto public_id = runtime_backend_registry::make_public_device_id(backend.runtime_type, "0.1");
        if (backend.runtime_type == default_runtime) {
            EXPECT_EQ(public_id, "0.1");
        } else {
            EXPECT_EQ(public_id, std::string(backend.name) + "_0.1");
        }
    }
}

TEST(runtime_backend_registry, rejects_older_or_incomplete_tagged_identity) {
    runtime_types runtime_type = get_default_runtime_type();
    std::string backend_device_id;

    EXPECT_FALSE(runtime_backend_registry::parse_device_id("0", runtime_type, backend_device_id));
    const auto incomplete_tag = std::string(runtime_backend_registry::default_backend().name) + "_";
    EXPECT_FALSE(runtime_backend_registry::parse_device_id(incomplete_tag, runtime_type, backend_device_id));
    EXPECT_FALSE(runtime_backend_registry::parse_device_id("_0", runtime_type, backend_device_id));
    EXPECT_FALSE(runtime_backend_registry::parse_device_id("unknown_0", runtime_type, backend_device_id));
}

TEST(runtime_backend_no_fallback, missing_default_runtime_device_does_not_select_another_runtime) {
    const auto default_runtime = runtime_backend_registry::default_backend().runtime_type;
    const auto alternate_runtime = default_runtime == runtime_types::vulkan ? runtime_types::ocl : runtime_types::vulkan;
    const std::vector<std::pair<runtime_types, std::string>> available_devices{{alternate_runtime, "0"}};

    EXPECT_TRUE(runtime_backend_registry::select_default_device_id(available_devices).empty());
}

TEST(runtime_backend_no_fallback, selects_first_device_from_configured_default_runtime) {
    const auto default_runtime = runtime_backend_registry::default_backend().runtime_type;
    const auto alternate_runtime = default_runtime == runtime_types::vulkan ? runtime_types::ocl : runtime_types::vulkan;
    const std::vector<std::pair<runtime_types, std::string>> available_devices{
        {alternate_runtime, "0"},
        {default_runtime, "1"},
        {default_runtime, "2"},
    };

    EXPECT_EQ(runtime_backend_registry::select_default_device_id(available_devices), "1");
}
