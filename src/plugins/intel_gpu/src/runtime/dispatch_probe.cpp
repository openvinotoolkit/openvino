// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/dispatch_probe.hpp"

#include "intel_gpu/runtime/device.hpp"

namespace cldnn {
namespace {

// Append the raw bytes of a trivially-copyable value to a byte buffer.
template <typename T>
void append_bytes(std::vector<uint8_t>& out, const T& value) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

}  // namespace

std::vector<uint8_t> make_fingerprint(const device_info& info) {
    std::vector<uint8_t> fp;
    append_bytes(fp, info.pci_info.pci_domain);
    append_bytes(fp, info.pci_info.pci_bus);
    append_bytes(fp, info.pci_info.pci_device);
    append_bytes(fp, info.pci_info.pci_function);
    append_bytes(fp, info.vendor_id);
    append_bytes(fp, info.sub_device_idx);
    return fp;
}

ov::DeviceCompatibilityScore probe_score(runtime_types rt, const device_info& info, std::string_view forced_rt) noexcept {
    const bool intel = (info.vendor_id == INTEL_VENDOR_ID);
    // Perf tier from gfx_ver.major (Xe2+ == 20: BMG/LNL 20.x, PTL 30.x), NOT from gpu_arch,
    // which is unknown in this no-oneDNN probe build (device_ops_table).
    const bool xe2_plus = intel && info.gfx_ver.major >= 20;

    ov::DeviceCompatibilityScore score = ov::PROBE_SCORE_SERVABLE;
    if (rt == runtime_types::ze) {
        // Level Zero is Intel-only. ZE is PREFERRED only when perf-ideal (Xe2+) AND the real
        // ZE<->OCL interop capability is present (supports_leo, read without engine init).
        if (!intel)
            score = ov::PROBE_SCORE_INCOMPATIBLE;
        else if (xe2_plus && info.supports_leo)
            score = ov::PROBE_SCORE_PREFERRED;
        else
            score = ov::PROBE_SCORE_SERVABLE;
    } else if (rt == runtime_types::ocl) {
        // OCL serves interop natively (no LEO concept). CAPABLE on Xe2+ so it wins when ZE
        // lacks LEO (ZE=SERVABLE) but loses when ZE has it (ZE=PREFERRED).
        if (!intel)
            score = ov::PROBE_SCORE_SERVABLE;
        else if (!xe2_plus)
            score = ov::PROBE_SCORE_PREFERRED;
        else
            score = ov::PROBE_SCORE_CAPABLE;
    }

    // The override names what a group ships, so anything else (e.g. SYCL) is ignored: honouring
    // it would drop every candidate's Intel devices and hide the device entirely.
    if (forced_rt != "OCL" && forced_rt != "ZE")
        return score;
    // It applies only to Intel devices (the runtimes' shared, contended set); a non-Intel GPU
    // keeps its OCL score so it stays served regardless of the override.
    if (!intel || score == ov::PROBE_SCORE_INCOMPATIBLE)
        return score;
    return forced_rt == to_cache_tag(rt) ? ov::PROBE_SCORE_PREFERRED : ov::PROBE_SCORE_INCOMPATIBLE;
}

}  // namespace cldnn
