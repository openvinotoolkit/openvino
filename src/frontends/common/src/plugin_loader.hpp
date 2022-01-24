// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/frontend/manager.hpp>

#ifdef _WIN32
static const char PathSeparator[] = ";";
#else
static const char PathSeparator[] = ":";
#endif  // _WIN32

namespace ov {
namespace frontend {

/// \brief Internal data structure holding plugin information including library handle, file names and paths, etc.
class PluginInfo {
    std::shared_ptr<void> m_so;  // Library shared object, must be first data member to be destroyed last
    bool m_loaded = false;
    FrontEndPluginInfo m_fe_info;  // Loaded Frontend Plugin Info obtained from exported API
    bool m_load_failed = false;    // Remember if loading of plugin is already failed
    bool load_internal();

public:
    std::string m_file_name;  // Plugin file name, e.g. "libov_ir_frontend.so"
    std::string m_file_path;  // Plugin file full path

    PluginInfo() = default;

    PluginInfo(std::string name, FrontEndFactory creator) {
        m_fe_info.m_name = std::move(name);
        m_fe_info.m_creator = std::move(creator);
        m_loaded = true;
    }

    const FrontEndPluginInfo& get_creator() const {
        return m_fe_info;
    }

    // Use in future to pass library handle pointer to frontend/input_model/function/executable_network
    std::shared_ptr<void> get_so_pointer() const {
        return m_so;
    }

    bool is_loaded() const {
        return m_loaded;
    }

    std::string get_name_from_file() const;

    bool is_file_name_match(const std::string& name) const;

    bool load();
};

// Searches for available plugins in a specified directory
// Appends found plugins to existing list
void find_plugins(const std::string& dir_name, std::vector<PluginInfo>& res);

}  // namespace frontend
}  // namespace ov
