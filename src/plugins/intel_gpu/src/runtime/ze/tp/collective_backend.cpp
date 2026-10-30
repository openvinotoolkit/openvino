// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "collective_backend.hpp"

#include <algorithm>
#include <sstream>

#include "openvino/core/except.hpp"
#include "openvino/runtime/tp/collective_contract.hpp"

namespace cldnn {
namespace ze {
namespace tp {

namespace {

// Every backend the plugin has, in the order "AUTO" tries them.
const std::vector<std::string>& known_backends() {
    static const std::vector<std::string> names{l0_p2p_backend};
    return names;
}

std::string join(const std::vector<std::string>& names) {
    std::string out;
    for (const auto& name : names) {
        out += (out.empty() ? "" : ",") + name;
    }
    return out;
}

// Every device reaches every other device's memory, in both directions.
std::string l0_p2p_support(const DeviceGroup& group) {
    if (!group.problem.empty()) {
        return group.problem;
    }
    for (std::size_t i = 0; i < group.devices.size(); ++i) {
        for (std::size_t j = 0; j < group.devices.size(); ++j) {
            if (i == j) {
                continue;
            }
            ze_bool_t can_access = 0;
            const auto result = ov::zeDeviceCanAccessPeer(group.devices[i], group.devices[j], &can_access);
            if (result != ZE_RESULT_SUCCESS) {
                std::ostringstream reason;
                reason << "cannot query peer access from " << group.names[i] << " to " << group.names[j]
                       << " (Level Zero error 0x" << std::hex << static_cast<uint32_t>(result) << ")";
                return reason.str();
            }
            if (!can_access) {
                return "P2P access from " + group.names[i] + " to " + group.names[j] +
                       " is not available; collective backend " + l0_p2p_backend +
                       " needs it between every pair of devices of the group. Check the BIOS settings and the "
                       "GPU driver version";
            }
        }
    }
    return {};
}

std::string support_of(const DeviceGroup& group, const std::string& name) {
    if (name == l0_p2p_backend) {
        return l0_p2p_support(group);
    }
    OPENVINO_THROW("[GPU][TP] no support check for collective backend ", name);
}

}  // namespace

std::vector<std::string> supported_backends(const DeviceGroup& group) {
    std::vector<std::string> names;
    for (const auto& name : known_backends()) {
        if (support_of(group, name).empty()) {
            names.push_back(name);
        }
    }
    return names;
}

std::string backend_support(const DeviceGroup& group, const std::string& backend) {
    const auto& known = known_backends();
    if (backend == ov::tp::auto_backend) {
        // Nothing serves the group: report why the preferred backend does not.
        return supported_backends(group).empty() ? support_of(group, known.front()) : std::string{};
    }
    if (std::find(known.begin(), known.end(), backend) == known.end()) {
        const auto supported = supported_backends(group);
        return "Unknown collective backend \"" + backend + "\"; supported for " + join(group.names) + ": " +
               (supported.empty() ? std::string("none") : join(supported));
    }
    return support_of(group, backend);
}

std::string resolve_backend(const DeviceGroup& group, const std::string& requested) {
    const auto reason = backend_support(group, requested);
    OPENVINO_ASSERT(reason.empty(), "[GPU][TP] ", reason);
    return requested == ov::tp::auto_backend ? supported_backends(group).front() : requested;
}

}  // namespace tp
}  // namespace ze
}  // namespace cldnn
