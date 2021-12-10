// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/util/env_util.hpp>
#include <openvino/util/file_util.hpp>

#include "common/frontend_exceptions.hpp"
#include "common/place.hpp"
#include "manager.hpp"
#include "ngraph/except.hpp"
#include "openvino/util/env_util.hpp"
#include "plugin_loader.hpp"
#include "so_extension.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::frontend;

//----------- FrontEndManager ---------------------------
class FrontEndManager::Impl {
    std::mutex m_loading_mutex;
    std::vector<PluginInfo> m_plugins;

public:
    Impl() {
        search_all_plugins();
    }

    ~Impl() = default;

    FrontEnd::Ptr load_by_framework(const std::string& framework) {
        // Mapping of default FE name to file name (without prefix and suffix)
        std::map<std::string, std::string> predefined_frontends = {
            {"ir", "ir"},
            {"onnx", "onnx"},
            {"tf", "tensorflow"},
            {"paddle", "paddlepaddle"},
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
                    return plugin_it->get_creator().m_creator();
                }
            }
        }
        // Load plugins until we found the right one
        for (auto& plugin : m_plugins) {
            plugin.load();
            if (plugin.get_creator().m_name == framework) {
                return plugin.get_creator().m_creator();
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
                return fe;
            }
        }
        return {};
    }

    void register_front_end(const std::string& name, FrontEndFactory creator) {
        PluginInfo plugin_info(name, std::move(creator));
        std::lock_guard<std::mutex> guard(m_loading_mutex);
        m_plugins.push_back(std::move(plugin_info));
    }

private:
    // Helper structure for searching plugin either by name or by file name
    // File name here doesn't contain prefix/suffix (like "_ov_frontend.so")
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
            {".pdmodel", {"paddle", "paddlepaddle"}},
        };

        // List of prioritized frontends.
        std::list<FrontEndNames> priority_list = {
            {"ir", "ir"},
            {"onnx", "onnx"},
            {"tf", "tensorflow"},
            {"paddle", "paddlepaddle"},
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
                return fe;
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
            auto start = 0u;
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

//----------- FrontEnd ---------------------------

FrontEnd::FrontEnd() = default;

FrontEnd::~FrontEnd() = default;

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    return false;
}

InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& params) const {
    FRONT_END_NOT_IMPLEMENTED(load_impl);
}
std::shared_ptr<Model> FrontEnd::convert(InputModel::Ptr model) const {
    FRONT_END_NOT_IMPLEMENTED(convert);
}

void FrontEnd::convert(std::shared_ptr<Model>) const {
    FRONT_END_NOT_IMPLEMENTED(convert);
}

std::shared_ptr<Model> FrontEnd::convert_partially(InputModel::Ptr model) const {
    FRONT_END_NOT_IMPLEMENTED(convert_partially);
}

std::shared_ptr<Model> FrontEnd::decode(InputModel::Ptr model) const {
    FRONT_END_NOT_IMPLEMENTED(decode);
}

void FrontEnd::normalize(std::shared_ptr<Model> model) const {
    FRONT_END_NOT_IMPLEMENTED(normalize);
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    // Left unimplemented intentionally.
    // Each frontend can support own set of extensions, so this method should be implemented on the frontend side
}

void FrontEnd::add_extension(const std::vector<std::shared_ptr<ov::Extension>>& extensions) {
    for (const auto& ext : extensions)
        add_extension(ext);
}

void FrontEnd::add_extension(const std::string& library_path) {
    add_extension(ov::detail::load_extensions(library_path));
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
void FrontEnd::add_extension(const std::wstring& library_path) {
    add_extension(ov::detail::load_extensions(library_path));
}
#endif

std::string FrontEnd::get_name() const {
    return std::string();
}

//----------- InputModel ---------------------------
std::vector<Place::Ptr> InputModel::get_inputs() const {
    return {};
}

std::vector<Place::Ptr> InputModel::get_outputs() const {
    return {};
}

Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensor_name) const {
    return nullptr;
}

Place::Ptr InputModel::get_place_by_operation_name(const std::string& operation_name) const {
    return nullptr;
}

Place::Ptr InputModel::get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                                  int input_port_index) {
    return nullptr;
}

Place::Ptr InputModel::get_place_by_operation_name_and_output_port(const std::string& operation_name,
                                                                   int output_port_index) {
    return nullptr;
}

void InputModel::set_name_for_tensor(Place::Ptr tensor, const std::string& new_name) {
    FRONT_END_NOT_IMPLEMENTED(set_name_for_tensor);
}

void InputModel::add_name_for_tensor(Place::Ptr tensor, const std::string& new_name) {
    FRONT_END_NOT_IMPLEMENTED(add_name_for_tensor);
}

void InputModel::set_name_for_operation(Place::Ptr operation, const std::string& new_name) {
    FRONT_END_NOT_IMPLEMENTED(set_name_for_operation);
}

void InputModel::free_name_for_tensor(const std::string& name) {
    FRONT_END_NOT_IMPLEMENTED(free_name_for_tensor);
}

void InputModel::free_name_for_operation(const std::string& name) {
    FRONT_END_NOT_IMPLEMENTED(free_name_for_operation);
}

void InputModel::set_name_for_dimension(Place::Ptr place, size_t shape_dim_index, const std::string& dim_name) {
    FRONT_END_NOT_IMPLEMENTED(set_name_for_dimension);
}

void InputModel::cut_and_add_new_input(Place::Ptr place, const std::string& new_name_optional) {
    FRONT_END_NOT_IMPLEMENTED(cut_and_add_new_input);
}

void InputModel::cut_and_add_new_output(Place::Ptr place, const std::string& new_name_optional) {
    FRONT_END_NOT_IMPLEMENTED(cut_and_add_new_output);
}

Place::Ptr InputModel::add_output(Place::Ptr place) {
    FRONT_END_NOT_IMPLEMENTED(add_output);
}

void InputModel::remove_output(Place::Ptr place) {
    FRONT_END_NOT_IMPLEMENTED(remove_output);
}

void InputModel::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    FRONT_END_NOT_IMPLEMENTED(override_all_outputs);
}

void InputModel::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    FRONT_END_NOT_IMPLEMENTED(override_all_inputs);
}

void InputModel::extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    FRONT_END_NOT_IMPLEMENTED(extract_subgraph);
}

// Setting tensor properties
void InputModel::set_partial_shape(Place::Ptr place, const PartialShape&) {
    FRONT_END_NOT_IMPLEMENTED(set_partial_shape);
}

PartialShape InputModel::get_partial_shape(Place::Ptr place) const {
    FRONT_END_NOT_IMPLEMENTED(get_partial_shape);
}

void InputModel::set_element_type(Place::Ptr place, const element::Type&) {
    FRONT_END_NOT_IMPLEMENTED(set_element_type);
}

void InputModel::set_tensor_value(Place::Ptr place, const void* value) {
    FRONT_END_NOT_IMPLEMENTED(set_tensor_value);
}

void InputModel::set_tensor_partial_value(Place::Ptr place, const void* min_value, const void* max_value) {
    FRONT_END_NOT_IMPLEMENTED(set_tensor_partial_value);
}

//----------- Place ---------------------------
std::vector<std::string> Place::get_names() const {
    FRONT_END_NOT_IMPLEMENTED(get_names);
}

std::vector<Place::Ptr> Place::get_consuming_operations() const {
    return {};
}

std::vector<Place::Ptr> Place::get_consuming_operations(int output_port_index) const {
    return {};
}

std::vector<Place::Ptr> Place::get_consuming_operations(const std::string& outputPortName, int outputPortIndex) const {
    return {};
}

Place::Ptr Place::get_target_tensor() const {
    return nullptr;
}

Place::Ptr Place::get_target_tensor(int output_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_producing_operation() const {
    return nullptr;
}

Place::Ptr Place::get_producing_operation(int input_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_producing_port() const {
    return nullptr;
}

Place::Ptr Place::get_input_port() const {
    return nullptr;
}

Place::Ptr Place::get_input_port(int input_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_input_port(const std::string& input_name) const {
    return nullptr;
}

Place::Ptr Place::get_input_port(const std::string& input_name, int input_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_output_port() const {
    return nullptr;
}

Place::Ptr Place::get_output_port(int output_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_output_port(const std::string& output_name) const {
    return nullptr;
}

Place::Ptr Place::get_output_port(const std::string& output_name, int output_port_index) const {
    return nullptr;
}

std::vector<Place::Ptr> Place::get_consuming_ports() const {
    return {};
}

bool Place::is_input() const {
    FRONT_END_NOT_IMPLEMENTED(is_input);
}

bool Place::is_output() const {
    FRONT_END_NOT_IMPLEMENTED(is_output);
}

bool Place::is_equal(Ptr another) const {
    FRONT_END_NOT_IMPLEMENTED(is_equal);
}

bool Place::is_equal_data(Ptr another) const {
    FRONT_END_NOT_IMPLEMENTED(is_equal_data);
}

Place::Ptr Place::get_source_tensor() const {
    return nullptr;
}

Place::Ptr Place::get_source_tensor(int input_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_source_tensor(const std::string& inputName, int inputPortIndex) const {
    return nullptr;
}

Place::Ptr Place::get_source_tensor(const std::string& inputName) const {
    return nullptr;
}

Place::Ptr Place::get_target_tensor(const std::string& outputPortName) const {
    return nullptr;
}

Place::Ptr Place::get_target_tensor(const std::string& outputPortName, int outputPortIndex) const {
    return nullptr;
}

Place::Ptr Place::get_producing_operation(const std::string& inputName) const {
    return nullptr;
}

Place::Ptr Place::get_producing_operation(const std::string& inputName, int inputPortIndex) const {
    return nullptr;
}

std::vector<Place::Ptr> Place::get_consuming_operations(const std::string& outputPortName) const {
    return {};
}
