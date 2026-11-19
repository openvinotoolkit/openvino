// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The devices the TP functional tests put their ranks on. The tests check the TP
// plugin; the rank devices only run the rank models, so their type is a parameter:
// OV_TP_TEST_DEVICE, GPU by default.

#pragma once

#include <cstdlib>
#include <string>
#include <vector>

#include "openvino/runtime/core.hpp"

namespace ov::tp::tests {

/// Type of the rank devices, e.g. "GPU".
inline const std::string& rank_device_type() {
    static const std::string type = [] {
        const char* value = std::getenv("OV_TP_TEST_DEVICE");
        return std::string(value != nullptr && *value != '\0' ? value : "GPU");
    }();
    return type;
}

/// "<type>.<id>" of every available device of the type, in the order the device plugin lists them.
inline const std::vector<std::string>& rank_devices() {
    static const std::vector<std::string> devices = [] {
        std::vector<std::string> names;
        try {
            for (const auto& id : ov::Core{}.get_property(rank_device_type(), ov::available_devices))
                names.push_back(rank_device_type() + "." + id);
        } catch (const std::exception&) {
        }
        return names;
    }();
    return devices;
}

inline const std::string& rank_device(size_t rank) {
    return rank_devices().at(rank);
}

/// Why `count` ranks cannot run here; empty if they can.
inline std::string missing_rank_devices(size_t count) {
    if (rank_devices().size() >= count)
        return {};
    return "need " + std::to_string(count) + " " + rank_device_type() + " devices, found " +
           std::to_string(rank_devices().size());
}

/// The TP device name for `world_size` ranks on the first rank devices.
inline std::string tp_device(size_t world_size) {
    std::string name = "TP:";
    for (size_t rank = 0; rank < world_size; ++rank)
        name += (rank == 0 ? "" : ",") + rank_device(rank);
    return name;
}

}  // namespace ov::tp::tests
