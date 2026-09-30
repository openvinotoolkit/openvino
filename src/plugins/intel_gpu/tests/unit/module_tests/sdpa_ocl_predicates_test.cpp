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
bool xe_hpg_plus(gpu_arch a) {
    return a == gpu_arch::xe_hpg || a == gpu_arch::xe_hpc || xe2_plus(a);
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
    return get("TEST_USE_SDPA_OCL") + " " + get("TEST_USE_SDPA_OCL_DECODE") + " " + get("TEST_USE_SDPA_OCL_HPG");
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
            for (bool ocl : kBools) {
                for (bool hpg : kBools) {
                    SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a) << " immad=" << immad << " ocl=" << ocl
                                                    << " hpg=" << hpg);
                    const bool want = kOneDnn && ocl && immad && (xe2_plus(a) || (a == gpu_arch::xe_hpg && hpg));
                    EXPECT_EQ(pa::sdpa_ocl_selected(make_info(a, immad), ocl, hpg), want);
                }
            }
        }
    }
}

TEST(SdpaOclPredicates, DecodeReaderAvailableTruthTable) {
    for (auto a : kArchs) {
        for (bool immad : kBools) {
            for (bool dec : kBools) {
                SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a) << " immad=" << immad << " decode=" << dec);
                // No HPG term: the signature has none, which pins the reader to Xe2+ whatever the lane switch says.
                EXPECT_EQ(pa::sdpa_ocl_decode_reader_available(make_info(a, immad), dec), dec && immad && xe2_plus(a));
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
            EXPECT_EQ(pa::sdpa_ocl_selected(info),
                      pa::sdpa_ocl_selected(info, pa::sdpa_ocl_enabled(), pa::sdpa_ocl_hpg_enabled()));
            EXPECT_EQ(pa::sdpa_ocl_decode_reader_available(info),
                      pa::sdpa_ocl_decode_reader_available(info, pa::sdpa_ocl_decode_enabled()));
        }
    }
}

// The landmine row: with TEST_USE_SDPA_OCL_HPG=1 sdpa_ocl_selected(xe_hpg) is true, yet no GENERATE reader reads a
// token-major page there, so the page must stay d-major. The expectation does not look at the HPG switch.
TEST(SdpaOclPredicates, ByChannelReadableTruthTable) {
    SCOPED_TRACE(env_trace());
    if (kOneDnn && pa::sdpa_ocl_enabled() && pa::sdpa_ocl_hpg_enabled()) {
        // Positive control: the landmine is armed (the old one-predicate check would have said yes here).
        ASSERT_TRUE(pa::sdpa_ocl_selected(make_info(gpu_arch::xe_hpg, true)));
    }
    pa::by_channel_tm_op_info op;
    op.k_head_size = 64;
    op.v_head_size = 64;
    op.heads_num = 8;
    op.kv_heads_num = 2;
    for (auto a : kArchs) {
        for (bool immad : kBools) {
            for (bool mk : kBools) {
                SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a) << " immad=" << immad << " mk=" << mk);
                const bool want = kOneDnn && pa::sdpa_ocl_enabled() && pa::sdpa_ocl_decode_enabled() && immad && mk && xe2_plus(a);
                EXPECT_EQ(pa::by_channel_token_major_readable(make_info(a, immad), mk, ov::element::f16, {op}), want);
            }
        }
    }
}

TEST(SdpaOclPredicates, ExpectedDpasBackendTruthTable) {
    using tests::dpas_backend;
    const auto oracle = [](gpu_arch a, bool immad, bool mk, bool ocl_enabled, bool hpg, bool paged, size_t head) {
        if (!kOneDnn || !immad || !mk)
            return dpas_backend::none;
        if (!xe_hpg_plus(a))
            return dpas_backend::none;
        if (ocl_enabled && (xe2_plus(a) || (a == gpu_arch::xe_hpg && hpg)))
            return dpas_backend::ocl;
        if (a == gpu_arch::xe3p && (paged || head <= 64))
            return dpas_backend::none;
        return dpas_backend::micro;
    };
    for (auto a : kArchs) {
        for (bool immad : kBools) {
            for (bool mk : kBools) {
                for (bool ocl : kBools) {
                    for (bool hpg : kBools) {
                        for (bool paged : kBools) {
                            for (size_t head : {size_t{64}, size_t{128}}) {
                                SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(a) << " immad=" << immad << " mk=" << mk
                                                                << " ocl=" << ocl << " hpg=" << hpg << " paged=" << paged
                                                                << " head=" << head);
                                EXPECT_EQ(tests::expected_dpas_backend_for(make_info(a, immad), mk, ocl, hpg, paged, head),
                                          oracle(a, immad, mk, ocl, hpg, paged, head));
                            }
                        }
                    }
                }
            }
        }
    }
    if (!kOneDnn)
        return;
    // Named rows, for readers.
    const auto fn = [](gpu_arch a, bool mk, bool ocl, bool hpg, bool paged, size_t head) {
        return tests::expected_dpas_backend_for(make_info(a, true), mk, ocl, hpg, paged, head);
    };
    EXPECT_EQ(fn(gpu_arch::xe_hpg, true, true, false, false, 128), dpas_backend::micro);   // DG2 default
    EXPECT_EQ(fn(gpu_arch::xe_hpg, true, false, false, false, 128), dpas_backend::micro);  // DG2 TEST_USE_SDPA_OCL=0
    EXPECT_EQ(fn(gpu_arch::xe_hpg, true, true, true, false, 128), dpas_backend::ocl);      // DG2 HPG=1
    EXPECT_EQ(fn(gpu_arch::xe2, true, true, false, false, 128), dpas_backend::ocl);        // B70
    EXPECT_EQ(fn(gpu_arch::xe2, true, false, false, false, 128), dpas_backend::micro);     // B70 TEST_USE_SDPA_OCL=0
    EXPECT_EQ(fn(gpu_arch::xe3p, true, false, false, true, 128), dpas_backend::none);      // xe3p PA, =0
    EXPECT_EQ(fn(gpu_arch::xe3p, true, false, false, false, 128), dpas_backend::micro);    // xe3p plain SDPA head 128, =0
    EXPECT_EQ(fn(gpu_arch::xe2, false, true, false, false, 128), dpas_backend::none);      // no microkernels
}
