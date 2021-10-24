// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "frontend_manager/frontend_manager.hpp"

#ifdef _WIN32
static const char FileSeparator[] = "\\";
static const char PathSeparator[] = ";";
#else
static const char FileSeparator[] = "/";
static const char PathSeparator[] = ":";
#endif  // _WIN32

namespace ov {
namespace frontend {

struct PluginData {
    PluginData(const std::shared_ptr<void>& h, FrontEndPluginInfo&& info) : m_lib_handle(h), m_plugin_info(info) {}

    std::shared_ptr<void> m_lib_handle;  // Shall be destroyed when plugin is not needed anymore to free memory
    FrontEndPluginInfo m_plugin_info;
};

// Searches for available plugins in a specified directory
std::vector<PluginData> load_plugins(const std::string& dir_name);

}  // namespace frontend
}  // namespace ov
