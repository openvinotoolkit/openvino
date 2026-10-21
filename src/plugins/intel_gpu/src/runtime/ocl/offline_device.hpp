// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device_descriptor.hpp"
#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "openvino/core/except.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl.hpp>
#endif

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cldnn {
namespace ocl {

// HW-free device carrying a device_info restored from an enumeration-time descriptor. No real
// cl::Device: never enumerated, never initialized against hardware.
class offline_device : public device {
public:
    explicit offline_device(const device_info& info, std::vector<uint8_t> onednn_device_info = {})
        : _info(info)
        , _onednn_device_info(std::move(onednn_device_info))
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

    // oneDNN's own serialized device_info, see device_descriptor.hpp.
    const std::vector<uint8_t>& onednn_device_info() const { return _onednn_device_info; }

private:
    device_info _info;
    std::vector<uint8_t> _onednn_device_info;
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

#ifdef ENABLE_ONEDNN_FOR_GPU
// Fingerprint of the oneDNN that produced (or will consume) the opaque device_info blob. The blob
// itself carries no version tag, so a mismatched oneDNN would misread its fields silently.
inline std::string onednn_fingerprint() {
    const auto* v = dnnl::version();
    std::ostringstream os;
    os << v->major << "." << v->minor << "." << v->patch << "+" << (v->hash ? v->hash : "");
    return os.str();
}
#endif

// Restores oneDNN's own device_info blob from the descriptor. Empty when the enumerating build had
// no oneDNN; a blob from a different oneDNN is rejected rather than misread.
inline std::vector<uint8_t> make_offline_onednn_device_info(const std::string& descriptor) {
    auto blob = deserialize_onednn_device_info(descriptor);
#ifdef ENABLE_ONEDNN_FOR_GPU
    if (!blob.empty()) {
        auto produced_by = deserialize_onednn_version(descriptor);
        OPENVINO_ASSERT(produced_by == onednn_fingerprint(),
                        "[GPU] Device descriptor carries a oneDNN device_info from '", produced_by,
                        "' but this build links oneDNN '", onednn_fingerprint(), "'");
    }
#endif
    return blob;
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
