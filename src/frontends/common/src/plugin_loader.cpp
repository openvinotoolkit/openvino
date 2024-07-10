// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <direct.h>
#    include <windows.h>
#else  // _WIN32
#    include <dirent.h>
#    include <dlfcn.h>
#    include <unistd.h>
#endif  // _WIN32

#include <sys/stat.h>

#include <string>
#include <vector>

#include "openvino/util/file_util.hpp"
#include "openvino/util/log.hpp"
#include "openvino/util/shared_object.hpp"
#include "plugin_loader.hpp"

using namespace ov;
using namespace ov::frontend;

// Note, static methods below are required to create an order of initialization of static variables
// e.g. if users (not encouraged) created ov::Model globally, we need to ensure proper order of initialization

/// \return map of shared object per frontend <frontend_name, frontend_so_ptr>
std::unordered_map<std::string, std::shared_ptr<void>>& ov::frontend::get_shared_objects_map() {
    static std::unordered_map<std::string, std::shared_ptr<void>> shared_objects_map;
    return shared_objects_map;
}

/// \return Mutex to guard access the shared object map
std::mutex& ov::frontend::get_shared_objects_mutex() {
    static std::mutex shared_objects_map_mutex;
    return shared_objects_map_mutex;
}

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
            {"tflite", "tensorflow_lite"},
            {"paddle", "paddle"},
            {"pytorch", "pytorch"},
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
            OPENVINO_DEBUG("Static frontend for '", plugin_info.m_file_name, "' is already loaded\n");
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

    std::lock_guard<std::mutex> guard(get_shared_objects_mutex());
    get_shared_objects_map().emplace(get_creator().m_name, get_so_pointer());

    return true;
}

bool PluginInfo::load_internal() {
    std::shared_ptr<void> so;
    try {
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
        so = ov::util::load_shared_object(ov::util::string_to_wstring(m_file_path).c_str());
#else
        so = ov::util::load_shared_object(m_file_path.c_str());
#endif
    }
#ifdef ENABLE_OPENVINO_DEBUG
    catch (const std::exception& ex) {
        OPENVINO_DEBUG("Error loading FrontEnd '",
                       m_file_path,
                       "': ",
                       ex.what(),
                       " Please check that frontend library doesn't have unresolved dependencies.\n");
        return false;
    }
#else
    catch (const std::exception&) {
        return false;
    }
#endif

    auto info_addr = reinterpret_cast<void* (*)()>(ov::util::get_symbol(so, "get_api_version"));
    if (!info_addr) {
        OPENVINO_DEBUG("Loaded FrontEnd [", m_file_path, "] doesn't have API version");
        return false;
    }
    FrontEndVersion plug_info{reinterpret_cast<FrontEndVersion>(info_addr())};

    if (plug_info != OV_FRONTEND_API_VERSION) {
        // Plugin has incompatible API version, do not load it
        OPENVINO_DEBUG("Loaded FrontEnd [", m_file_path, "] has incompatible API version", plug_info, "\n");
        return false;
    }

    auto creator_addr = reinterpret_cast<void* (*)()>(ov::util::get_symbol(so, "get_front_end_data"));
    if (!creator_addr) {
        OPENVINO_DEBUG("Loaded FrontEnd [", m_file_path, "] doesn't have Frontend Data", "\n");
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
