// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "device_info.hpp"
#include "engine_configuration.hpp"
#include "openvino/runtime/iplugin.hpp"

namespace cldnn {

// Opaque cross-runtime device identity: PCI-BDF + vendor id + sub-device index, the fields both
// stacks populate identically. UUID is excluded - ZE sets it while legacy OCL zero-fills it.
std::vector<uint8_t> make_fingerprint(const device_info& info);

// The dispatch-group score runtime 'rt' reports for a device. 'forced_rt' is the raw
// OV_GPU_RUNTIME value; only "OCL"/"ZE" override, and never for a non-Intel device.
ov::DeviceCompatibilityScore probe_score(runtime_types rt, const device_info& info, std::string_view forced_rt) noexcept;

}  // namespace cldnn
