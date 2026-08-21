// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the GPU dispatch-probe policy: the score each runtime reports for a device,
// and the cross-runtime device identity Core merges candidates by. Both are pure functions of
// device_info, so a single binary covers the OCL and ZE rows regardless of the runtime it links.

#include "intel_gpu/runtime/dispatch_probe.hpp"

#include <gtest/gtest.h>

#include "intel_gpu/runtime/device.hpp"  // INTEL_VENDOR_ID

using namespace cldnn;

namespace {

constexpr uint32_t OTHER_VENDOR_ID = 0x10de;

device_info make_info(uint32_t vendor_id, uint16_t gfx_major, bool supports_leo) {
    device_info info{};
    info.vendor_id = vendor_id;
    info.gfx_ver.major = gfx_major;
    info.supports_leo = supports_leo;
    return info;
}

device_info intel_pre_xe2() {
    return make_info(INTEL_VENDOR_ID, 12, false);
}
device_info intel_xe2_with_leo() {
    return make_info(INTEL_VENDOR_ID, 20, true);
}
device_info intel_xe2_no_leo() {
    return make_info(INTEL_VENDOR_ID, 20, false);
}
device_info non_intel() {
    return make_info(OTHER_VENDOR_ID, 20, true);
}

// The winner Core would pick for a device: highest score, ties to the first candidate.
runtime_types winner(const device_info& info, std::string_view forced = "") {
    const auto ze = probe_score(runtime_types::ze, info, forced);
    const auto ocl = probe_score(runtime_types::ocl, info, forced);
    return ze > ocl ? runtime_types::ze : runtime_types::ocl;
}

}  // namespace

TEST(dispatch_probe, ze_serves_only_intel_devices) {
    EXPECT_EQ(probe_score(runtime_types::ze, non_intel(), ""), ov::PROBE_SCORE_INCOMPATIBLE);
    EXPECT_NE(probe_score(runtime_types::ze, intel_pre_xe2(), ""), ov::PROBE_SCORE_INCOMPATIBLE);
}

TEST(dispatch_probe, ze_prefers_xe2_only_with_leo) {
    EXPECT_EQ(probe_score(runtime_types::ze, intel_xe2_with_leo(), ""), ov::PROBE_SCORE_PREFERRED);
    // Without the interop capability ZE yields, even on a perf-ideal device.
    EXPECT_EQ(probe_score(runtime_types::ze, intel_xe2_no_leo(), ""), ov::PROBE_SCORE_SERVABLE);
    // LEO alone does not lift a pre-Xe2 device.
    EXPECT_EQ(probe_score(runtime_types::ze, make_info(INTEL_VENDOR_ID, 12, true), ""), ov::PROBE_SCORE_SERVABLE);
}

TEST(dispatch_probe, ocl_prefers_pre_xe2_and_stays_capable_on_xe2) {
    EXPECT_EQ(probe_score(runtime_types::ocl, intel_pre_xe2(), ""), ov::PROBE_SCORE_PREFERRED);
    EXPECT_EQ(probe_score(runtime_types::ocl, intel_xe2_with_leo(), ""), ov::PROBE_SCORE_CAPABLE);
    EXPECT_EQ(probe_score(runtime_types::ocl, intel_xe2_no_leo(), ""), ov::PROBE_SCORE_CAPABLE);
}

TEST(dispatch_probe, ocl_serves_non_intel_devices) {
    EXPECT_EQ(probe_score(runtime_types::ocl, non_intel(), ""), ov::PROBE_SCORE_SERVABLE);
}

// The product behaviour: which backend actually gets the device.
TEST(dispatch_probe, winner_per_device_class) {
    EXPECT_EQ(winner(intel_xe2_with_leo()), runtime_types::ze);
    EXPECT_EQ(winner(intel_xe2_no_leo()), runtime_types::ocl);
    EXPECT_EQ(winner(intel_pre_xe2()), runtime_types::ocl);
    EXPECT_EQ(winner(non_intel()), runtime_types::ocl);
}

TEST(dispatch_probe, override_forces_the_named_runtime_for_intel_devices) {
    // ZE would lose this device on score, but the override hands it over.
    EXPECT_EQ(probe_score(runtime_types::ze, intel_pre_xe2(), "ZE"), ov::PROBE_SCORE_PREFERRED);
    EXPECT_EQ(probe_score(runtime_types::ocl, intel_pre_xe2(), "ZE"), ov::PROBE_SCORE_INCOMPATIBLE);
    EXPECT_EQ(winner(intel_pre_xe2(), "ZE"), runtime_types::ze);

    // And the other way round for a device ZE would otherwise win.
    EXPECT_EQ(probe_score(runtime_types::ocl, intel_xe2_with_leo(), "OCL"), ov::PROBE_SCORE_PREFERRED);
    EXPECT_EQ(probe_score(runtime_types::ze, intel_xe2_with_leo(), "OCL"), ov::PROBE_SCORE_INCOMPATIBLE);
    EXPECT_EQ(winner(intel_xe2_with_leo(), "OCL"), runtime_types::ocl);
}

TEST(dispatch_probe, override_never_promotes_a_device_ze_cannot_serve) {
    // ZE is INCOMPATIBLE with non-Intel devices; the override must not override that.
    EXPECT_EQ(probe_score(runtime_types::ze, non_intel(), "ZE"), ov::PROBE_SCORE_INCOMPATIBLE);
    // And OCL keeps serving it, so the device stays visible under any override.
    EXPECT_EQ(probe_score(runtime_types::ocl, non_intel(), "ZE"), ov::PROBE_SCORE_SERVABLE);
    EXPECT_EQ(winner(non_intel(), "ZE"), runtime_types::ocl);
}

TEST(dispatch_probe, unshipped_or_malformed_override_is_ignored) {
    // A runtime no group ships (or garbage) must not drop every candidate's Intel devices.
    for (const auto& forced : {"SYCL", "ocl", "", "GARBAGE"}) {
        EXPECT_EQ(probe_score(runtime_types::ze, intel_xe2_with_leo(), forced), ov::PROBE_SCORE_PREFERRED) << forced;
        EXPECT_EQ(probe_score(runtime_types::ocl, intel_xe2_with_leo(), forced), ov::PROBE_SCORE_CAPABLE) << forced;
    }
}

TEST(dispatch_probe, fingerprint_ignores_uuid_but_separates_pci_addresses) {
    auto a = intel_xe2_with_leo();
    auto b = a;
    // ZE populates the UUID and legacy OCL zero-fills it, so the same device must still match.
    b.uuid.uuid[0] = 0x42;
    EXPECT_EQ(make_fingerprint(a), make_fingerprint(b));

    b = a;
    b.pci_info.pci_device += 1;
    EXPECT_NE(make_fingerprint(a), make_fingerprint(b));

    b = a;
    b.sub_device_idx = 3;
    EXPECT_NE(make_fingerprint(a), make_fingerprint(b));

    b = a;
    b.vendor_id = OTHER_VENDOR_ID;
    EXPECT_NE(make_fingerprint(a), make_fingerprint(b));
}

TEST(dispatch_probe, fingerprint_is_fixed_width_and_never_empty) {
    // Core rejects an empty fingerprint, and compares by == only, so the layout must be stable.
    const auto fp = make_fingerprint(device_info{});
    EXPECT_FALSE(fp.empty());
    EXPECT_EQ(fp.size(), make_fingerprint(intel_xe2_with_leo()).size());
}
