// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <level_zero/ze_api.h>
#include <ze_command_queue_npu_ext.h>
#include <ze_driver_npu_ext.h>

#include <cstring>
#include <string>
#include <vector>

#include "intel_npu/utils/zero/zero_utils.hpp"

using intel_npu::zeroUtils::isExtensionNameMatching;
using intel_npu::zeroUtils::queryDriverExtensionVersion;

namespace {

ze_driver_extension_properties_t makeExtension(const std::string& name, uint32_t version) {
    ze_driver_extension_properties_t property = {};
    std::strncpy(property.name, name.c_str(), sizeof(property.name) - 1);
    property.version = version;
    return property;
}

std::tuple<uint32_t, std::string> query(const char* extName,
                                        uint32_t targetVersion,
                                        const std::vector<ze_driver_extension_properties_t>& extProps) {
    return queryDriverExtensionVersion(extName, targetVersion, extProps, static_cast<uint32_t>(extProps.size()));
}

}  // namespace

TEST(ZeroExtensionNameMatching, ExactNameMatches) {
    EXPECT_TRUE(isExtensionNameMatching(ZE_GRAPH_EXT_NAME, ZE_GRAPH_EXT_NAME));
    EXPECT_TRUE(isExtensionNameMatching(ZE_DRIVER_NPU_EXT_NAME, ZE_DRIVER_NPU_EXT_NAME));
    EXPECT_TRUE(isExtensionNameMatching(ZE_COMMAND_QUEUE_NPU_EXT_NAME, ZE_COMMAND_QUEUE_NPU_EXT_NAME));
}

TEST(ZeroExtensionNameMatching, GraphExtAcceptsVersionedNames) {
    EXPECT_TRUE(isExtensionNameMatching("ZE_extension_graph_1_1", ZE_GRAPH_EXT_NAME));
    EXPECT_TRUE(isExtensionNameMatching("ZE_extension_graph_1_2", ZE_GRAPH_EXT_NAME));
    EXPECT_TRUE(isExtensionNameMatching("ZE_extension_graph_1_19", ZE_GRAPH_EXT_NAME));
}

TEST(ZeroExtensionNameMatching, GraphExtRejectsNonVersionedSuffixes) {
    EXPECT_FALSE(isExtensionNameMatching("ZE_extension_graph_profiling", ZE_GRAPH_EXT_NAME));
    EXPECT_FALSE(isExtensionNameMatching("ZE_extension_graph_", ZE_GRAPH_EXT_NAME));
    EXPECT_FALSE(isExtensionNameMatching("ZE_extension_graphs", ZE_GRAPH_EXT_NAME));
    EXPECT_FALSE(isExtensionNameMatching("ZE_extension_graph1_1", ZE_GRAPH_EXT_NAME));
    EXPECT_FALSE(isExtensionNameMatching("ZE_extension_gra", ZE_GRAPH_EXT_NAME));
}

TEST(ZeroExtensionNameMatching, NonGraphExtRequiresExactName) {
    EXPECT_FALSE(isExtensionNameMatching("ZE_extension_driver_npu_1_1", ZE_DRIVER_NPU_EXT_NAME));
    EXPECT_FALSE(isExtensionNameMatching("ZE_extension_command_queue_npu_ext", ZE_COMMAND_QUEUE_NPU_EXT_NAME));
    EXPECT_FALSE(isExtensionNameMatching("ZE_extension_command_queue", ZE_COMMAND_QUEUE_NPU_EXT_NAME));
}

TEST(ZeroExtensionQuery, ReturnsTargetVersionWhenDriverIsNewer) {
    const std::vector<ze_driver_extension_properties_t> extProps = {
        makeExtension(ZE_DRIVER_NPU_EXT_NAME, ZE_DRIVER_NPU_EXT_VERSION_1_0),
        makeExtension(ZE_GRAPH_EXT_NAME, ZE_MAKE_VERSION(1, 25)),
    };

    const auto [version, name] = query(ZE_GRAPH_EXT_NAME, ZE_GRAPH_EXT_VERSION_1_19, extProps);
    EXPECT_EQ(version, static_cast<uint32_t>(ZE_GRAPH_EXT_VERSION_1_19));
    EXPECT_EQ(name, ZE_GRAPH_EXT_NAME);
}

TEST(ZeroExtensionQuery, PicksLatestVersionedGraphExtForOlderDriver) {
    const std::vector<ze_driver_extension_properties_t> extProps = {
        makeExtension("ZE_extension_graph_1_1", ZE_MAKE_VERSION(1, 1)),
        makeExtension("ZE_extension_graph_1_3", ZE_MAKE_VERSION(1, 3)),
        makeExtension("ZE_extension_graph_1_2", ZE_MAKE_VERSION(1, 2)),
    };

    const auto [version, name] = query(ZE_GRAPH_EXT_NAME, ZE_GRAPH_EXT_VERSION_1_19, extProps);
    EXPECT_EQ(version, static_cast<uint32_t>(ZE_MAKE_VERSION(1, 3)));
    EXPECT_EQ(name, "ZE_extension_graph_1_3");
}

TEST(ZeroExtensionQuery, IgnoresOtherExtensionsSharingGraphPrefix) {
    const std::vector<ze_driver_extension_properties_t> extProps = {
        makeExtension("ZE_extension_graph_profiling", ZE_MAKE_VERSION(1, 30)),
        makeExtension("ZE_extension_graph_1_2", ZE_MAKE_VERSION(1, 2)),
    };

    const auto [version, name] = query(ZE_GRAPH_EXT_NAME, ZE_GRAPH_EXT_VERSION_1_19, extProps);
    EXPECT_EQ(version, static_cast<uint32_t>(ZE_MAKE_VERSION(1, 2)));
    EXPECT_EQ(name, "ZE_extension_graph_1_2");
}

TEST(ZeroExtensionQuery, NonGraphExtIgnoresNamesWithSuffix) {
    const std::vector<ze_driver_extension_properties_t> extProps = {
        makeExtension(std::string(ZE_COMMAND_QUEUE_NPU_EXT_NAME) + "_1_1", ZE_COMMAND_QUEUE_NPU_EXT_VERSION_1_1),
    };

    const auto [version, name] = query(ZE_COMMAND_QUEUE_NPU_EXT_NAME, ZE_COMMAND_QUEUE_NPU_EXT_VERSION_1_1, extProps);
    EXPECT_EQ(version, 0u);
    EXPECT_TRUE(name.empty());
}

TEST(ZeroExtensionQuery, NonGraphExtFindsExactName) {
    const std::vector<ze_driver_extension_properties_t> extProps = {
        makeExtension(ZE_GRAPH_EXT_NAME, ZE_GRAPH_EXT_VERSION_1_19),
        makeExtension(ZE_COMMAND_QUEUE_NPU_EXT_NAME, ZE_COMMAND_QUEUE_NPU_EXT_VERSION_1_0),
    };

    const auto [version, name] = query(ZE_COMMAND_QUEUE_NPU_EXT_NAME, ZE_COMMAND_QUEUE_NPU_EXT_VERSION_1_1, extProps);
    EXPECT_EQ(version, static_cast<uint32_t>(ZE_COMMAND_QUEUE_NPU_EXT_VERSION_1_0));
    EXPECT_EQ(name, ZE_COMMAND_QUEUE_NPU_EXT_NAME);
}

TEST(ZeroExtensionQuery, ReturnsEmptyWhenExtensionMissing) {
    const std::vector<ze_driver_extension_properties_t> extProps = {
        makeExtension(ZE_DRIVER_NPU_EXT_NAME, ZE_DRIVER_NPU_EXT_VERSION_1_0),
    };

    const auto [version, name] = query(ZE_GRAPH_EXT_NAME, ZE_GRAPH_EXT_VERSION_1_19, extProps);
    EXPECT_EQ(version, 0u);
    EXPECT_TRUE(name.empty());
}
