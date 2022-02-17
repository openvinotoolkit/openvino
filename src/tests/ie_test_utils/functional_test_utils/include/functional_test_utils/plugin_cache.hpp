// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <ie_core.hpp>

class PluginCache {
public:
    std::shared_ptr<InferenceEngine::Core> ie(const std::string &deviceToCheck = std::string());

    static PluginCache& get();

    void reset();

    PluginCache(const PluginCache&) = delete;
    PluginCache& operator=(const PluginCache&) = delete;

private:
    PluginCache();
    ~PluginCache() = default;

    std::mutex g_mtx;
    std::shared_ptr<InferenceEngine::Core> ie_core;
};
