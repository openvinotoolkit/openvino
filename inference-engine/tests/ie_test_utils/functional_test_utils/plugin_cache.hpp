// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <ie_core.hpp>

class PluginCache {
public:
    static PluginCache& get();

    std::shared_ptr<InferenceEngine::Core> ie(const std::string &deviceToCheck = std::string()) const;

    void reset();

public:
    PluginCache(const PluginCache&) = delete;
    PluginCache& operator=(const PluginCache&) = delete;

private:
    PluginCache();
    ~PluginCache() = default;
};
