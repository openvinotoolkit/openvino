// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "openvino/runtime/core.hpp"

namespace ov {
namespace test {
namespace utils {

class PluginCache {
public:
    std::shared_ptr<ov::Core> core(const std::string& deviceToCheck = std::string());

    static PluginCache& get();

    void reset();

    PluginCache(const PluginCache&) = delete;

    PluginCache& operator=(const PluginCache&) = delete;

private:
    PluginCache();

    ~PluginCache() = default;

    std::mutex g_mtx;
    std::shared_ptr<ov::Core> ov_core;
};
}  // namespace utils
}  // namespace test
}  // namespace ov
