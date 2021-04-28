// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/ifrontend_manager.hpp>

namespace ngraph {
namespace frontend {

    class PluginHandle {
    public:
        PluginHandle(std::function<void()> callOnDestruct): m_callOnDestruct(callOnDestruct) {}
        PluginHandle(const PluginHandle&) = delete;
        PluginHandle& operator=(const PluginHandle&) = delete;
        PluginHandle(PluginHandle&&) = default;
        PluginHandle& operator=(PluginHandle&&) = default;
        ~PluginHandle() { if (m_callOnDestruct) m_callOnDestruct(); }

    private:
        std::function<void()> m_callOnDestruct;
    };

    struct PluginData {
        PluginData(const PluginInfo& b, FrontEndFactory&& c, PluginHandle&& h): baseInfo(b), creator(std::move(c)), libHandle(std::move(h)) {}
        PluginInfo baseInfo;
        FrontEndFactory creator;
        PluginHandle libHandle;
    };

    std::vector<PluginData> loadPlugins(const std::string& dirName);

}  // namespace frontend
}  // namespace ngraph
