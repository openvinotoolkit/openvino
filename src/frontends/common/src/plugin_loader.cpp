// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <Windows.h>
#    include <direct.h>
#else  // _WIN32
#    include <dirent.h>
#    include <dlfcn.h>
#    include <unistd.h>
#endif  // _WIN32

#include <sys/stat.h>

#include <openvino/util/log.hpp>
#include <string>
#include <vector>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "plugin_loader.hpp"

using namespace ov;
using namespace ov::frontend;

#ifdef OPENVINO_STATIC_LIBRARY

#    include "ov_frontends.hpp"

namespace {

void load_static_plugins(std::vector<PluginInfo>& res) {
    for (const auto& frontend : getStaticFrontendsRegistry()) {
        FrontEndPluginInfo factory;
        {
            std::unique_ptr<FrontEndPluginInfo> fact{reinterpret_cast<FrontEndPluginInfo*>(frontend.m_dataFunc())};
            factory = std::move(*fact);
        }
        PluginInfo plugin_info(factory.m_name, factory.m_creator);
        static const std::map<std::string, std::string> predefined_frontends = {
            {"ir", "ir"},
            {"onnx", "onnx"},
            {"tf", "tensorflow"},
            {"paddle", "paddle"},
        };
        auto it = predefined_frontends.find(factory.m_name);
        if (it != predefined_frontends.end()) {
            plugin_info.m_file_name = it->second;
        }
        res.emplace_back(std::move(plugin_info));
    }
}

}  // namespace

#endif  // OPENVINO_STATIC_LIBRARY

// TODO: change to std::filesystem for C++17
static std::vector<std::string> list_files(const std::string& path) {
    std::vector<std::string> res;
    try {
        const auto prefix = std::string(FRONTEND_LIB_PREFIX);
        const auto suffix = std::string(FRONTEND_LIB_SUFFIX);
        ov::util::iterate_files(
            path,
            [&res, &prefix, &suffix](const std::string& file_path, bool is_dir) {
                auto file = ov::util::get_file_name(file_path);
                if (!is_dir && (prefix.empty() || file.compare(0, prefix.length(), prefix) == 0) &&
                    file.length() > suffix.length() &&
                    file.rfind(suffix) == (file.length() - std::string(suffix).length())) {
                    res.push_back(file_path);
                }
            },
            false,
            true);
    } catch (...) {
        // Ignore exceptions
    }
    return res;
}

void ov::frontend::find_plugins(const std::string& dir_name, std::vector<PluginInfo>& res) {
#ifdef OPENVINO_STATIC_LIBRARY
    load_static_plugins(res);
#endif  // OPENVINO_STATIC_LIBRARY
    for (const auto& file_path : list_files(dir_name)) {
        PluginInfo plugin_info;
        plugin_info.m_file_path = file_path;
        plugin_info.m_file_name = ov::util::get_file_name(file_path);
        // if frontend is registered already (e.g. as static), skip found version
        if (std::find_if(res.begin(), res.end(), [&plugin_info](const PluginInfo& pd) {
                return plugin_info.get_name_from_file() == pd.get_name_from_file();
            }) == res.end()) {
            res.emplace_back(std::move(plugin_info));
        } else {
            OPENVINO_DEBUG << "Static frontend for '" << plugin_info.m_file_name << "' is already loaded\n";
        }
    }
}

/////////////////////////
std::string PluginInfo::get_name_from_file() const {
    const auto prefix = std::string(FRONTEND_LIB_PREFIX);
    const auto suffix = std::string(FRONTEND_LIB_SUFFIX);
    auto prefix_pos = m_file_name.find(prefix);
    auto suffix_pos = m_file_name.rfind(suffix);
    if (prefix_pos == 0 && suffix_pos + suffix.length() == m_file_name.length()) {
        return m_file_name.substr(prefix_pos + prefix.length(), suffix_pos - prefix_pos - prefix.length());
    }
    return m_file_name;
}

bool PluginInfo::is_file_name_match(const std::string& name) const {
    std::string file_name = std::string(FRONTEND_LIB_PREFIX) + name + std::string(FRONTEND_LIB_SUFFIX);
    return file_name == m_file_name;
}

bool PluginInfo::load() {
    if (m_loaded) {
        return true;
    } else if (m_load_failed) {
        return false;
    }
    if (!load_internal()) {
        m_load_failed = true;
        return false;
    }
    return true;
}

bool PluginInfo::load_internal() {
    std::shared_ptr<void> so;
    try {
        so = ov::util::load_shared_object(m_file_path.c_str(), ov::util::SharedObjectDeferCloser());
    } catch (const std::exception& ex) {
        OPENVINO_DEBUG << "Error loading FrontEnd '" << m_file_path << "': " << ex.what() << std::endl;
        return false;
    }

    auto info_addr = reinterpret_cast<void* (*)()>(ov::util::get_symbol(so, "GetAPIVersion"));
    if (!info_addr) {
        OPENVINO_DEBUG << "Loaded FrontEnd [" << m_file_path << "] doesn't have API version" << std::endl;
        return false;
    }
    FrontEndVersion plug_info{reinterpret_cast<FrontEndVersion>(info_addr())};

    if (plug_info != OV_FRONTEND_API_VERSION) {
        // Plugin has incompatible API version, do not load it
        OPENVINO_DEBUG << "Loaded FrontEnd [" << m_file_path << "] has incompatible API version" << plug_info
                       << std::endl;
        return false;
    }

    auto creator_addr = reinterpret_cast<void* (*)()>(ov::util::get_symbol(so, "GetFrontEndData"));
    if (!creator_addr) {
        OPENVINO_DEBUG << "Loaded FrontEnd [" << m_file_path << "] doesn't have Frontend Data" << std::endl;
        return false;
    }

    FrontEndPluginInfo factory;
    {
        std::unique_ptr<FrontEndPluginInfo> fact{reinterpret_cast<FrontEndPluginInfo*>(creator_addr())};
        factory = std::move(*fact);
    }
    // Fill class members (noexcept)
    m_fe_info = std::move(factory);
    m_so = std::move(so);
    m_loaded = true;
    return true;
}
