// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "samples/affinity_utils.hpp"

#include <fstream>
#include <sstream>
#include <unordered_set>

#include "samples/slog.hpp"

#ifdef JSON_HEADER
#    include <json.hpp>
#else
#    include <nlohmann/json.hpp>
#endif

namespace {

void apply_affinities_from_file(const std::shared_ptr<ov::Model>& model,
                                const std::string& file_path,
                                const std::vector<std::string>& hardware_devices = {},
                                bool fallback_unmapped_ops = true) {
    std::ifstream input(file_path);
    if (!input.is_open()) {
        OPENVINO_THROW("Failed to open affinity file: ", file_path);
    }

    nlohmann::json affinity_json;
    try {
        input >> affinity_json;
    } catch (const nlohmann::json::parse_error& ex) {
        OPENVINO_THROW("Failed to parse affinity file ", file_path, ": ", ex.what());
    }
    if (!affinity_json.is_object()) {
        OPENVINO_THROW("Affinity file must contain a JSON object with {node_name: device_name} mappings: ", file_path);
    }

    const std::unordered_set<std::string> allowed_devices(hardware_devices.begin(), hardware_devices.end());

    std::unordered_set<std::string> mapped_devices;
    for (const auto& item : affinity_json.items()) {
        if (!item.value().is_string()) {
            OPENVINO_THROW("Affinity file ",
                           file_path,
                           " contains non-string mapping value for node '",
                           item.key(),
                           "'. Expected device name as string.");
        }

        const auto device = item.value().get<std::string>();
        if (!allowed_devices.empty() && allowed_devices.find(device) == allowed_devices.end()) {
            std::ostringstream devices_oss;
            for (size_t i = 0; i < hardware_devices.size(); ++i) {
                if (i != 0) {
                    devices_oss << ", ";
                }
                devices_oss << hardware_devices[i];
            }
            OPENVINO_THROW("Affinity file ",
                           file_path,
                           " references device '",
                           device,
                           "' for node '",
                           item.key(),
                           "', but it is not present in hardware devices list: ",
                           devices_oss.str());
        }

        mapped_devices.insert(device);
    }

    std::vector<std::string> unmapped_hardware_devices;
    for (const auto& hardware_device : hardware_devices) {
        if (mapped_devices.find(hardware_device) == mapped_devices.end()) {
            unmapped_hardware_devices.push_back(hardware_device);
        }
    }

    const std::string fallback_device =
        fallback_unmapped_ops && unmapped_hardware_devices.size() == 1 ? unmapped_hardware_devices.front()
                                                                       : std::string{};

    size_t applied_count = 0;
    size_t fallback_count = 0;
    bool has_unmapped_ops = false;
    for (auto&& node : model->get_ops()) {
        const auto& friendly_name = node->get_friendly_name();
        auto it = affinity_json.find(friendly_name);
        if (it == affinity_json.end()) {
            it = affinity_json.find(node->get_name());
        }
        if (it != affinity_json.end() && it->is_string()) {
            node->get_rt_info()["affinity"] = it->get<std::string>();
            applied_count++;
        } else if (!fallback_device.empty()) {
            node->get_rt_info()["affinity"] = fallback_device;
            fallback_count++;
        } else {
            has_unmapped_ops = true;
        }
    }

    if (has_unmapped_ops && fallback_unmapped_ops && fallback_device.empty()) {
        if (hardware_devices.empty()) {
            OPENVINO_THROW("Affinity file ",
                           file_path,
                           " does not cover all ops, and no hardware devices were provided to infer a fallback device. "
                           "Please map the remaining ops explicitly.");
        }

        std::ostringstream oss;
        for (size_t i = 0; i < unmapped_hardware_devices.size(); ++i) {
            if (i != 0) {
                oss << ", ";
            }
            oss << unmapped_hardware_devices[i];
        }

        if (unmapped_hardware_devices.empty()) {
            OPENVINO_THROW("Affinity file ",
                           file_path,
                           " does not cover all ops, and every hardware device is already used in the JSON mappings. "
                           "Please map the remaining ops explicitly.");
        }

        OPENVINO_THROW("Affinity file ",
                       file_path,
                       " does not cover all ops, and the sample cannot infer a unique fallback device from the "
                       "remaining hardware devices: ",
                       oss.str(),
                       ". Please map the remaining ops explicitly.");
    }

    slog::info << "Applied manual affinity mappings to " << applied_count << " ops from " << file_path << slog::endl;
    if (fallback_count != 0) {
        slog::info << "Assigned fallback affinity \"" << fallback_device << "\" to " << fallback_count
                   << " ops not listed in " << file_path << slog::endl;
    } else if (!fallback_unmapped_ops) {
        slog::info << "Automatic affinity fallback is disabled; ops not listed in " << file_path
                   << " can be assigned by an explicit fallback device" << slog::endl;
    }
}

}  // namespace

void apply_manual_affinities(const std::shared_ptr<ov::Model>& model,
                             const std::string& affinity_spec,
                             const std::vector<std::string>& hardware_devices,
                             bool fallback_unmapped_ops) {
    if (affinity_spec.empty()) {
        return;
    }

    const bool has_json_ext =
        affinity_spec.size() >= 5 && (affinity_spec.compare(affinity_spec.size() - 5, 5, ".json") == 0 ||
                                      affinity_spec.compare(affinity_spec.size() - 5, 5, ".JSON") == 0);

    if (has_json_ext) {
        apply_affinities_from_file(model, affinity_spec, hardware_devices, fallback_unmapped_ops);
        return;
    }

    if (!hardware_devices.empty()) {
        const std::unordered_set<std::string> allowed_devices(hardware_devices.begin(), hardware_devices.end());
        if (allowed_devices.find(affinity_spec) == allowed_devices.end()) {
            std::ostringstream devices_oss;
            for (size_t i = 0; i < hardware_devices.size(); ++i) {
                if (i != 0) {
                    devices_oss << ", ";
                }
                devices_oss << hardware_devices[i];
            }
            OPENVINO_THROW("Affinity value '",
                           affinity_spec,
                           "' is not present in hardware devices list: ",
                           devices_oss.str());
        }
    }

    for (auto&& node : model->get_ops()) {
        node->get_rt_info()["affinity"] = affinity_spec;
    }

    slog::info << "Applied manual affinity \"" << affinity_spec << "\" to all ops" << slog::endl;
}
