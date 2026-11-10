// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Host-only truth tables of the sdpa_ocl lane predicates (paged_attention.hpp) and of the test mirror built on them.
// Every device is a fake device_info, so xe_hpg rows run on any machine. The oracles spell the architecture sets out
// instead of using '>=' (the implementation's comparison), so a wrong comparison cannot be shared with its oracle.

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "dpas_backend_test_helper.h"

namespace {
using cldnn::gpu_arch;
using pa = cldnn::paged_attention;

const std::vector<gpu_arch> kArchs = {gpu_arch::unknown,
                                      gpu_arch::gen9,
                                      gpu_arch::gen11,
                                      gpu_arch::xe_lp,
                                      gpu_arch::xe_hp,
                                      gpu_arch::xe_hpg,
                                      gpu_arch::xe_hpc,
                                      gpu_arch::xe2,
                                      gpu_arch::xe3,
                                      gpu_arch::xe3p};
const std::vector<bool> kBools = {false, true};

bool xe2_plus(gpu_arch a) {
    return a == gpu_arch::xe2 || a == gpu_arch::xe3 || a == gpu_arch::xe3p;  // spelled out, NOT '>='
}
cldnn::device_info make_info(gpu_arch a, bool immad) {
    cldnn::device_info i{};
    i.arch = a;
    i.supports_immad = immad;
    return i;
}
constexpr bool kOneDnn =
#ifdef ENABLE_ONEDNN_FOR_GPU
    true;
#else
    false;
#endif

std::string env_trace() {
    const auto get = [](const char* n) {
        const char* v = std::getenv(n);
        return std::string(n) + "=" + (v ? v : "<unset>");
    };
    return get("TEST_USE_SDPA_OCL_DECODE");
}
}  // namespace

TEST(SdpaOclPredicates, ArchOkTruthTable) {
    for (auto a : kArchs) {
        SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a));
        const bool want = xe2_plus(a) || a == gpu_arch::xe_hpg;
        EXPECT_EQ(pa::sdpa_ocl_arch_ok(make_info(a, true)), want);
    }
}

TEST(SdpaOclPredicates, SelectedTruthTable) {
    for (auto a : kArchs) {
        for (bool immad : kBools) {
            SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a) << " immad=" << immad);
            const bool want = kOneDnn && immad && (xe2_plus(a) || a == gpu_arch::xe_hpg);
            EXPECT_EQ(pa::sdpa_ocl_selected(make_info(a, immad)), want);
        }
    }
}

TEST(SdpaOclPredicates, DecodeReaderAvailableTruthTable) {
    using sw = pa::decode_switch;
    for (auto a : kArchs) {
        for (bool immad : kBools) {
            for (sw dec : {sw::unset, sw::on, sw::off}) {
                SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a) << " immad=" << immad << " decode=" << static_cast<int>(dec));
                // Xe2+ (SG16) unless switched off, and xe_hpg (SG8) only when asked for; no oneDNN term.
                const bool want = dec != sw::off && immad && (xe2_plus(a) || (a == gpu_arch::xe_hpg && dec == sw::on));
                EXPECT_EQ(pa::sdpa_ocl_decode_reader_available(make_info(a, immad), dec), want);
            }
        }
    }
}

TEST(SdpaOclPredicates, ProcessEnvOverloadsUseProcessEnv) {
    SCOPED_TRACE(env_trace());
    for (auto a : kArchs) {
        for (bool immad : kBools) {
            SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a) << " immad=" << immad);
            const auto info = make_info(a, immad);
            EXPECT_EQ(pa::sdpa_ocl_decode_reader_available(info), pa::sdpa_ocl_decode_reader_available(info, pa::sdpa_ocl_decode_switch()));
        }
    }
}

TEST(SdpaOclPredicates, ExpectedDpasBackendTruthTable) {
    using tests::dpas_backend;
    // sdpa_ocl on its lane, the opt kernels everywhere else (no other DPAS kernel since sdpa_micro was removed).
    const auto oracle = [](gpu_arch a, bool immad) {
        if (!kOneDnn || !immad)
            return dpas_backend::none;
        if (xe2_plus(a) || a == gpu_arch::xe_hpg)
            return dpas_backend::ocl;
        return dpas_backend::none;
    };
    for (auto a : kArchs) {
        for (bool immad : kBools) {
            SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a) << " immad=" << immad);
            EXPECT_EQ(tests::expected_dpas_backend_for(make_info(a, immad)), oracle(a, immad));
        }
    }
    if (!kOneDnn)
        return;
    // Named rows, for readers.
    const auto fn = [](gpu_arch a, bool immad = true) {
        return tests::expected_dpas_backend_for(make_info(a, immad));
    };
    EXPECT_EQ(fn(gpu_arch::xe_hpg), dpas_backend::ocl);          // DG2 (A770)
    EXPECT_EQ(fn(gpu_arch::xe_hpg, false), dpas_backend::none);  // xe_hpg without XMX
    EXPECT_EQ(fn(gpu_arch::xe_hpc), dpas_backend::none);         // PVC
    EXPECT_EQ(fn(gpu_arch::xe2), dpas_backend::ocl);             // B70
    EXPECT_EQ(fn(gpu_arch::xe3p), dpas_backend::ocl);            // xe3p
}
