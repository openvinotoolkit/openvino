// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "openvino/runtime/core.hpp"

namespace ov {
namespace test {
namespace utils {

// global plugin config is set up for conformance
extern ov::AnyMap global_plugin_config;
extern std::string target_device;
extern std::string target_plugin_name;
extern std::unordered_set<std::string> available_devices;

void register_plugin(ov::Core& ov_core) noexcept;
void register_template_plugin(ov::Core& ov_core) noexcept;
ov::Core create_core(const std::string& in_target_device = std::string());

class PluginCache {
public:
    std::shared_ptr<ov::Core> core(const std::string& in_target_device = std::string());

    static PluginCache& get();
    void reset();

    PluginCache(const PluginCache&) = delete;
    PluginCache& operator=(const PluginCache&) = delete;

private:
    PluginCache();
    ~PluginCache() = default;

    std::mutex g_mtx;
    bool disable_plugin_cache;
    std::shared_ptr<ov::Core> ov_core;
};

}  // namespace utils
}  // namespace test
}  // namespace ov
