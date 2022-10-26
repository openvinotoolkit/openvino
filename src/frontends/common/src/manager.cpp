// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/manager.hpp"

#include <openvino/util/env_util.hpp>
#include <openvino/util/file_util.hpp>

#include "openvino/frontend/exception.hpp"
#include "openvino/util/env_util.hpp"
#include "plugin_loader.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::frontend;

class FrontEndManager::Impl {
    std::mutex m_loading_mutex;
    std::vector<PluginInfo> m_plugins;

    /// \brief map of shared object per frontend <frontend_name, frontend_so_ptr>
    static std::unordered_map<std::string, std::shared_ptr<void>> m_shared_objects_map;
    /// \brief Mutex to guard access the shared object map
    static std::mutex m_shared_objects_map_mutex;

public:
    Impl() {
        search_all_plugins();
    }

    ~Impl() = default;

    FrontEnd::Ptr make_frontend(const ov::frontend::PluginInfo& plugin) {
        auto fe_obj = std::make_shared<FrontEnd>();
        fe_obj->m_shared_object = std::make_shared<FrontEndSharedData>(plugin.get_so_pointer());
        fe_obj->m_actual = plugin.get_creator().m_creator();

        std::lock_guard<std::mutex> guard(m_shared_objects_map_mutex);
        m_shared_objects_map.emplace(plugin.get_creator().m_name, fe_obj->m_shared_object);

        return fe_obj;
    }

    FrontEnd::Ptr load_by_framework(const std::string& framework) {
        // Mapping of default FE name to file name (without prefix and suffix)
        static const std::map<std::string, std::string> predefined_frontends = {
            {"ir", "ir"},
            {"onnx", "onnx"},
            {"tf", "tensorflow"},
            {"paddle", "paddle"},
        };
        auto it = predefined_frontends.find(framework);
        std::lock_guard<std::mutex> guard(m_loading_mutex);
        if (it != predefined_frontends.end()) {
            auto file_name = it->second;
            auto plugin_it = std::find_if(m_plugins.begin(), m_plugins.end(), [&file_name](const PluginInfo& item) {
                return item.is_file_name_match(file_name);
            });
            if (plugin_it != m_plugins.end()) {
                if (plugin_it->load()) {
                    return make_frontend(*plugin_it);
                }
            }
        }
        // Load plugins until we found the right one
        for (auto& plugin : m_plugins) {
            OPENVINO_ASSERT(plugin.load(), "Cannot load frontend ", plugin.get_name_from_file());
            if (plugin.get_creator().m_name == framework) {
                return make_frontend(plugin);
            }
        }
        FRONT_END_INITIALIZATION_CHECK(false, "FrontEnd for Framework ", framework, " is not found");
    }

    std::vector<std::string> available_front_ends() {
        std::vector<std::string> names;
        // Load all not loaded plugins/frontends
        std::lock_guard<std::mutex> guard(m_loading_mutex);
        for (auto& plugin_info : m_plugins) {
            if (!plugin_info.load()) {
                continue;
            }
            names.push_back(plugin_info.get_creator().m_name);
        }
        return names;
    }

    FrontEnd::Ptr load_by_model(const std::vector<ov::Any>& variants) {
        std::lock_guard<std::mutex> guard(m_loading_mutex);
        // Step 1: Search from hard-coded prioritized frontends first
        auto ptr = search_priority(variants);
        if (ptr) {
            return ptr;
        }
        // Step 2: Load and search from all available frontends
        for (auto& plugin : m_plugins) {
            if (!plugin.load()) {
                continue;
            }
            auto fe = plugin.get_creator().m_creator();
            OPENVINO_ASSERT(fe, "Frontend error: frontend '", plugin.get_creator().m_name, "' created null FrontEnd");
            if (fe->supported(variants)) {
                return make_frontend(plugin);
            }
        }
        return nullptr;
    }

    void register_front_end(const std::string& name, FrontEndFactory creator) {
        PluginInfo plugin_info(name, std::move(creator));
        std::lock_guard<std::mutex> guard(m_loading_mutex);
        m_plugins.push_back(std::move(plugin_info));
    }

    static void shutdown() {
        std::lock_guard<std::mutex> guard(m_shared_objects_map_mutex);
        m_shared_objects_map.clear();
    }

private:
    // Helper structure for searching plugin either by name or by file name
    // File name here doesn't contain prefix/suffix (like "openvino_*_frontend.so")
    struct FrontEndNames {
        FrontEndNames(std::string n, std::string f) : name(std::move(n)), file_name(std::move(f)) {}
        bool operator==(const FrontEndNames& other) const {
            return name == other.name && file_name == other.file_name;
        }
        std::string name;
        std::string file_name;
    };

    static bool name_match(const PluginInfo& info, const FrontEndNames& names) {
        return info.is_file_name_match(names.file_name) || info.get_creator().m_name == names.name;
    }

    FrontEnd::Ptr search_priority(const std::vector<ov::Any>& variants) {
        // Map between file extension and suitable frontend
        static const std::map<std::string, FrontEndNames> priority_fe_extensions = {
            {".xml", {"ir", "ir"}},
            {".onnx", {"onnx", "onnx"}},
            {".pb", {"tf", "tensorflow"}},
            {".pdmodel", {"paddle", "paddle"}},
        };

        // List of prioritized frontends.
        std::list<FrontEndNames> priority_list = {
            {"ir", "ir"},
            {"onnx", "onnx"},
            {"tf", "tensorflow"},
            {"paddle", "paddle"},
        };
        if (variants.empty()) {
            return nullptr;
        }
        std::string model_path;

        const auto& model_variant = variants.at(0);
        if (model_variant.is<std::string>()) {
            const auto& tmp_path = model_variant.as<std::string>();
            model_path = tmp_path;
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        } else if (model_variant.is<std::wstring>()) {
            auto wpath = model_variant.as<std::wstring>();
            model_path = ov::util::wstring_to_string(wpath);
#endif
        }
        if (!model_path.empty()) {
            auto ext = ov::util::get_file_ext(model_path);
            auto it = priority_fe_extensions.find(ext);
            if (it != priority_fe_extensions.end()) {
                // Priority FE is found by file extension, try this first
                auto list_it = std::find(priority_list.begin(), priority_list.end(), it->second);
                OPENVINO_ASSERT(list_it != priority_list.end(),
                                "Internal error. Incorrect priority frontends configuration");
                // Move frontend matched by extension (e.g. ".onnx") to the top of priority list
                priority_list.splice(priority_list.begin(), priority_list, list_it);
            }
        }
        for (const auto& priority_info : priority_list) {
            auto plugin_it = std::find_if(m_plugins.begin(), m_plugins.end(), [&priority_info](const PluginInfo& info) {
                return name_match(info, priority_info);
            });
            if (plugin_it == m_plugins.end()) {
                continue;  // One of standard plugins is missing (correct case)
            }
            auto& plugin_info = *plugin_it;
            if (!plugin_info.is_loaded()) {
                if (!plugin_info.load()) {
                    // If standard plugin can't be loaded, it can also be ok (incompatible version, etc)
                    continue;
                }
            }
            // Plugin from priority list is loaded, create FrontEnd and check if it supports model loading
            auto fe = plugin_info.get_creator().m_creator();
            if (fe && fe->supported(variants)) {
                // Priority FE (e.g. IR) is found and is suitable
                return make_frontend(*plugin_it);
            }
        }
        return {};
    }

    void search_all_plugins() {
        auto search_from_dir = [&](const std::string& dir) {
            if (!dir.empty()) {
                find_plugins(dir, m_plugins);
            }
        };
        std::string env_path = ov::util::getenv_string("OV_FRONTEND_PATH");
        if (!env_path.empty()) {
            size_t start = 0;
            auto sep_pos = env_path.find(PathSeparator, start);
            while (sep_pos != std::string::npos) {
                search_from_dir(env_path.substr(start, sep_pos - start));
                start = sep_pos + 1;
                sep_pos = env_path.find(PathSeparator, start);
            }
            search_from_dir(env_path.substr(start, sep_pos));
        } else {
            search_from_dir(get_frontend_library_path());
        }
    }
};

std::unordered_map<std::string, std::shared_ptr<void>> FrontEndManager::Impl::m_shared_objects_map{};
std::mutex FrontEndManager::Impl::m_shared_objects_map_mutex{};

FrontEndManager::FrontEndManager() : m_impl(new Impl()) {}

FrontEndManager::FrontEndManager(FrontEndManager&&) noexcept = default;
FrontEndManager& FrontEndManager::operator=(FrontEndManager&&) noexcept = default;

FrontEndManager::~FrontEndManager() = default;

FrontEnd::Ptr FrontEndManager::load_by_framework(const std::string& framework) {
    return m_impl->load_by_framework(framework);
}

FrontEnd::Ptr FrontEndManager::load_by_model_impl(const std::vector<ov::Any>& variants) {
    return m_impl->load_by_model(variants);
}

std::vector<std::string> FrontEndManager::get_available_front_ends() {
    return m_impl->available_front_ends();
}

void FrontEndManager::register_front_end(const std::string& name, FrontEndFactory creator) {
    m_impl->register_front_end(name, std::move(creator));
}

template <>
FrontEnd::Ptr FrontEndManager::load_by_model(const std::vector<ov::Any>& variants) {
    return load_by_model_impl(variants);
}

void FrontEndManager::shutdown() {
    Impl::shutdown();
}
