// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "affinity_utils.hpp"

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
                                const std::vector<std::string>& hardware_devices = {}) {
    std::ifstream input(file_path);
    if (!input.is_open()) {
        OPENVINO_THROW("Failed to open affinity file: ", file_path);
    }

    nlohmann::json affinity_json;
    input >> affinity_json;
    if (!affinity_json.is_object()) {
        OPENVINO_THROW("Affinity file must contain a JSON object with {node_name: device_name} mappings: ",
                       file_path);
    }

    std::unordered_set<std::string> mapped_devices;
    for (const auto& item : affinity_json.items()) {
        if (item.value().is_string()) {
            mapped_devices.insert(item.value().get<std::string>());
        }
    }

    std::vector<std::string> unmapped_hardware_devices;
    for (const auto& hardware_device : hardware_devices) {
        if (mapped_devices.find(hardware_device) == mapped_devices.end()) {
            unmapped_hardware_devices.push_back(hardware_device);
        }
    }

    const std::string fallback_device = unmapped_hardware_devices.size() == 1 ? unmapped_hardware_devices.front()
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

    if (has_unmapped_ops && fallback_device.empty()) {
        if (hardware_devices.empty()) {
            OPENVINO_THROW("Affinity file ",
                           file_path,
                           " does not cover all ops, and benchmark_app cannot infer a fallback device because -d "
                           "does not provide hardware devices for affinity fallback. Please map the remaining ops "
                           "explicitly.");
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
                           " does not cover all ops, and every hardware device from -d is already used in the JSON "
                           "mappings. Please map the remaining ops explicitly.");
        }

        OPENVINO_THROW("Affinity file ",
                       file_path,
                       " does not cover all ops, and benchmark_app cannot infer a unique fallback device from -d. "
                       "Devices not referenced in the JSON mappings: ",
                       oss.str(),
                       ". Please map the remaining ops explicitly.");
    }

    slog::info << "Applied manual affinity mappings to " << applied_count << " ops from " << file_path
               << slog::endl;
    if (fallback_count != 0) {
        slog::info << "Assigned fallback affinity \"" << fallback_device << "\" to " << fallback_count
                   << " ops not listed in " << file_path << slog::endl;
    }
}

}  // namespace

void apply_manual_affinities(const std::shared_ptr<ov::Model>& model,
                             const std::string& affinity_spec,
                             const std::vector<std::string>& hardware_devices) {
    if (affinity_spec.empty()) {
        return;
    }

    const bool has_json_ext =
        affinity_spec.size() >= 5 &&
        (affinity_spec.compare(affinity_spec.size() - 5, 5, ".json") == 0 ||
         affinity_spec.compare(affinity_spec.size() - 5, 5, ".JSON") == 0);

    if (has_json_ext) {
        std::ifstream file_check(affinity_spec);
        if (!file_check.good()) {
            OPENVINO_THROW("Failed to open affinity file: ", affinity_spec);
        }
        apply_affinities_from_file(model, affinity_spec, hardware_devices);
        return;
    }

    for (auto&& node : model->get_ops()) {
        node->get_rt_info()["affinity"] = affinity_spec;
    }

    slog::info << "Applied manual affinity \"" << affinity_spec << "\" to all ops" << slog::endl;
}