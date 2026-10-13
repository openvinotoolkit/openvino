// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device_descriptor.hpp"
#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "openvino/core/except.hpp"

#include <sstream>
#include <string>
#include <vector>

namespace cldnn {
namespace ocl {

// HW-free device carrying a device_info restored from an enumeration-time descriptor. No real
// cl::Device: never enumerated, never initialized against hardware.
class offline_device : public device {
public:
    explicit offline_device(const device_info& info)
        : _info(info)
        , _mem_caps(info.supports_usm
                        ? std::vector<allocation_type>{allocation_type::cl_mem, allocation_type::usm_host,
                                                       allocation_type::usm_shared, allocation_type::usm_device}
                        : std::vector<allocation_type>{allocation_type::cl_mem}) {}

    const device_info& get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }
    void initialize() override {}
    bool is_initialized() const override { return true; }
    bool is_same(const device::ptr other) override { return this == other.get(); }
    void set_mem_caps(const memory_capabilities& memory_capabilities) override { _mem_caps = memory_capabilities; }

private:
    device_info _info;
    memory_capabilities _mem_caps;
};

// Restores the target's device_info from an enumeration-time descriptor. No fallback profile on
// purpose: guessing the target silently changes kernel and layout selection.
inline device_info make_offline_device_info(const std::string& descriptor) {
    OPENVINO_ASSERT(!descriptor.empty(),
                    "[GPU] Offline compilation requires a device descriptor. Query it on a machine with "
                    "the target GPU via core.get_property(\"GPU.x\", \"GPU_DEVICE_DESCRIPTOR\") and pass "
                    "it back through GPU_OFFLINE_COMPILE_DEVICE.");
    return deserialize_device_info(descriptor);
}

// ocloc's `-device` target derived from the descriptor, so the ISA target and the compile-time
// decisions always come from the same source.
inline std::string offline_ocloc_target(const device_info& info) {
    std::ostringstream os;
    os << info.gfx_ver.major << "." << static_cast<unsigned>(info.gfx_ver.minor) << "."
       << static_cast<unsigned>(info.gfx_ver.revision);
    return os.str();
}

}  // namespace ocl
}  // namespace cldnn
