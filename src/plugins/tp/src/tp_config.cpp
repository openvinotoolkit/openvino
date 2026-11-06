// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tp/tp_config.hpp"

#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace tp {

void write_log(const std::string& message) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    const auto path = TP_DEBUG_GLOBAL_OPT(log_to_file, std::string{});
    if (!path.empty()) {
        // Opened at the first message; a later change of the option is not followed.
        static std::ofstream file(path);
        if (file) {
            file << message << std::flush;
            return;
        }
        static bool reported = false;
        if (!std::exchange(reported, true))
            std::cout << "[TP] Cannot write TP_LOG_TO_FILE '" << path << "', logging to the standard output\n";
    }
    std::cout << message << std::flush;
}

std::vector<std::string> TPConfig::requested_device_names() const {
    const auto& list = get_priorities();
    std::vector<std::string> names;
    std::stringstream entries(list);
    std::string entry;
    while (std::getline(entries, entry, ',')) {
        const auto name = ov::util::trim(entry);
        OPENVINO_ASSERT(!name.empty(), "TP: the device list \"", list, "\" has an empty entry");
        names.emplace_back(name);
    }
    return names;
}

std::vector<std::string> TPConfig::resolve_device_names() const {
    auto names = requested_device_names();
    OPENVINO_ASSERT(names.size() >= 2,
                    "TP requires at least 2 devices, got ",
                    names.size(),
                    ". List them in the device name, e.g. \"TP:GPU.0,GPU.1\".");
    device_type(names);
    return names;
}

std::string TPConfig::device_type(const std::vector<std::string>& names) {
    OPENVINO_ASSERT(!names.empty(), "TP: no rank devices");
    const auto type = ov::DeviceIDParser(names.front()).get_device_name();
    for (const auto& name : names) {
        OPENVINO_ASSERT(ov::DeviceIDParser(name).get_device_name() == type,
                        "TP: all rank devices must have one device type, got ",
                        ov::util::join(names, ","));
    }
    return type;
}

bool TPConfig::owns(const std::string& name) const {
    return m_options_map.find(name) != m_options_map.end();
}

void TPConfig::erase_own_properties(ov::AnyMap& config) const {
    for (const auto& entry : m_options_map) {
        config.erase(entry.first);
    }
}

std::vector<ov::PropertyName> TPConfig::supported_properties() const {
    std::vector<ov::PropertyName> properties;
    for (const auto& entry : m_options_map) {
        if (entry.second->get_visibility() == ov::OptionVisibility::RELEASE) {
            properties.emplace_back(entry.first, ov::PropertyMutability::RW);
        }
    }
    return properties;
}

}  // namespace tp
}  // namespace ov
