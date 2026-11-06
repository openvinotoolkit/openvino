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
    return get("TEST_USE_SDPA_OCL_DECODE") + " " + get("TEST_USE_SDPA_OCL_HPG");
}
}  // namespace

TEST(SdpaOclPredicates, ArchOkTruthTable) {
    for (auto a : kArchs) {
        for (bool hpg : kBools) {
            SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a) << " hpg=" << hpg);
            const bool want = xe2_plus(a) || (a == gpu_arch::xe_hpg && hpg);
            EXPECT_EQ(pa::sdpa_ocl_arch_ok(make_info(a, true), hpg), want);
        }
    }
}

TEST(SdpaOclPredicates, SelectedTruthTable) {
    for (auto a : kArchs) {
        for (bool immad : kBools) {
            for (bool hpg : kBools) {
                SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a) << " immad=" << immad << " hpg=" << hpg);
                const bool want = kOneDnn && immad && (xe2_plus(a) || (a == gpu_arch::xe_hpg && hpg));
                EXPECT_EQ(pa::sdpa_ocl_selected(make_info(a, immad), hpg), want);
            }
        }
    }
}

TEST(SdpaOclPredicates, DecodeReaderAvailableTruthTable) {
    for (auto a : kArchs) {
        for (bool immad : kBools) {
            for (bool dec : kBools) {
                for (bool hpg : kBools) {
                    SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a) << " immad=" << immad << " decode=" << dec
                                                    << " hpg=" << hpg);
                    // Xe2+ (SG16), and xe_hpg on its opt-in lane (SG8); no oneDNN term.
                    const bool want = dec && immad && (xe2_plus(a) || (a == gpu_arch::xe_hpg && hpg));
                    EXPECT_EQ(pa::sdpa_ocl_decode_reader_available(make_info(a, immad), dec, hpg), want);
                }
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
            EXPECT_EQ(pa::sdpa_ocl_arch_ok(info), pa::sdpa_ocl_arch_ok(info, pa::sdpa_ocl_hpg_enabled()));
            EXPECT_EQ(pa::sdpa_ocl_selected(info), pa::sdpa_ocl_selected(info, pa::sdpa_ocl_hpg_enabled()));
            EXPECT_EQ(pa::sdpa_ocl_decode_reader_available(info),
                      pa::sdpa_ocl_decode_reader_available(info, pa::sdpa_ocl_decode_enabled(), pa::sdpa_ocl_hpg_enabled()));
        }
    }
}

// The landmine row: with TEST_USE_SDPA_OCL_HPG=1 both sdpa_ocl_selected(xe_hpg) and its SG8 GENERATE reader are there,
// yet sdpa_ocl's MIXED does not read every kind of token-major page on SG8, so the page must stay d-major. The
// expectation looks at no kernel switch: the layout is static per device and config (TEST_USE_SDPA_OCL_DECODE=0 or
// OV_GPU_ENABLE_SDPA_MICRO=0 only move the reads to paged_attention_opt.cl, which reads the token-major page too).
TEST(SdpaOclPredicates, TokenMajorRuleTruthTable) {
    SCOPED_TRACE(env_trace());
    if (kOneDnn && pa::sdpa_ocl_hpg_enabled()) {
        // Positive control: the landmine is armed (a check of the two readers alone would say yes here).
        ASSERT_TRUE(pa::sdpa_ocl_selected(make_info(gpu_arch::xe_hpg, true)));
        if (pa::sdpa_ocl_decode_enabled()) {
            ASSERT_TRUE(pa::sdpa_ocl_decode_reader_available(make_info(gpu_arch::xe_hpg, true)));
        }
    }
    for (auto a : kArchs) {
        for (bool immad : kBools) {
            SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a) << " immad=" << immad);
            const auto info = make_info(a, immad);
            const bool device = kOneDnn && immad && xe2_plus(a);
            EXPECT_EQ(pa::k_token_major_device(info), device);
            const bool want = device && pa::k_token_major_switch() != pa::k_layout_switch::d_major;
            // The same rule for every page kind (k_token_major_kind()).
            EXPECT_EQ(pa::k_token_major_rule(info, ov::element::f16, ov::element::i8, true), want);
            EXPECT_EQ(pa::k_token_major_rule(info, ov::element::f16, ov::element::u4, true), want);
            EXPECT_EQ(pa::k_token_major_rule(info, ov::element::f16, ov::element::f16, false), want);
            EXPECT_EQ(pa::k_token_major_rule(info, ov::element::f16, ov::element::i8, false), want);
            // Not a kind, or not an f16 inference precision: d-major everywhere.
            EXPECT_FALSE(pa::k_token_major_rule(info, ov::element::f16, ov::element::u8, false));
            EXPECT_FALSE(pa::k_token_major_rule(info, ov::element::f16, ov::element::u4, false));
            EXPECT_FALSE(pa::k_token_major_rule(info, ov::element::f32, ov::element::i8, true));
            EXPECT_FALSE(pa::k_token_major_rule(info, ov::element::bf16, ov::element::f16, false));
        }
    }
}

TEST(SdpaOclPredicates, ExpectedDpasBackendTruthTable) {
    using tests::dpas_backend;
    // sdpa_ocl on its lane, the opt kernels everywhere else (no other DPAS kernel since sdpa_micro was removed).
    const auto oracle = [](gpu_arch a, bool immad, bool hpg, bool tiers_ready) {
        if (!kOneDnn || !immad)
            return dpas_backend::none;
        if (xe2_plus(a))
            return dpas_backend::ocl;
        if (a == gpu_arch::xe_hpg && hpg && tiers_ready)
            return dpas_backend::ocl;
        return dpas_backend::none;
    };
    for (auto a : kArchs) {
        for (bool immad : kBools) {
            for (bool hpg : kBools) {
                for (bool tiers_ready : kBools) {
                    SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a) << " immad=" << immad << " hpg=" << hpg
                                                    << " tiers_ready=" << tiers_ready);
                    EXPECT_EQ(tests::expected_dpas_backend_for(make_info(a, immad), hpg, tiers_ready), oracle(a, immad, hpg, tiers_ready));
                }
            }
        }
    }
    if (!kOneDnn)
        return;
    // Named rows, for readers.
    const auto fn = [](gpu_arch a, bool hpg, bool tiers_ready = true) {
        return tests::expected_dpas_backend_for(make_info(a, true), hpg, tiers_ready);
    };
    EXPECT_EQ(fn(gpu_arch::xe_hpg, false), dpas_backend::none);        // DG2 default
    EXPECT_EQ(fn(gpu_arch::xe_hpg, true), dpas_backend::ocl);          // DG2 HPG=1, tier ready
    EXPECT_EQ(fn(gpu_arch::xe_hpg, true, false), dpas_backend::none);  // DG2 HPG=1, no tier ready: sdpa_ocl refuses the op
    EXPECT_EQ(fn(gpu_arch::xe_hpc, true), dpas_backend::none);         // PVC
    EXPECT_EQ(fn(gpu_arch::xe2, false), dpas_backend::ocl);            // B70
    EXPECT_EQ(fn(gpu_arch::xe2, true, false), dpas_backend::ocl);      // B70: no tier mask
    EXPECT_EQ(fn(gpu_arch::xe3p, false), dpas_backend::ocl);           // xe3p
}
