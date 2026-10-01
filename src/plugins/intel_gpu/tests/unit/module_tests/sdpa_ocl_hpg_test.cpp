// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Host-only checks of the xe_hpg (SG8) sdpa_ocl bring-up: the tiling and its device limits (Xe2 unchanged, xe_hpg valid for every
// head size) and the tier mask. No device is needed; the tiling is a pure function of the arch.

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "test_utils.h"
#ifdef ENABLE_ONEDNN_FOR_GPU
#    include "impls/ocl_v2/sdpa/sdpa_ocl_hpg.hpp"
#endif

#ifdef ENABLE_ONEDNN_FOR_GPU
namespace {
using cldnn::gpu_arch;
using namespace ov::intel_gpu::ocl;

const std::vector<size_t> kHeads = {32, 40, 48, 64, 72, 80, 96, 112, 128, 160, 192, 256, 320, 384, 448, 512};

size_t pow2_ceil(size_t h) {
    size_t d = 32;
    while (d < h)
        d *= 2;
    return d;
}

// The kernel's local memory, spelled out again from sdpa_ocl.cl (Q_slm, S_slm, S_sum_slm, S_max_slm) so an edit of the
// host formula alone is caught.
size_t oracle_slm(const SDPAOclTilingInfo& t, size_t d_max) {
    const size_t sg = static_cast<size_t>(t.subgroup_size);
    const size_t wgq = static_cast<size_t>(t.kq_wg_tile_queries);
    const size_t q_slm_dwords = (d_max / 16) * (wgq / sg) * 8 * sg;
    const size_t s_slm_dwords = static_cast<size_t>(t.kq_wg_tile_keys) * wgq / 2;
    return 4 * (q_slm_dwords + s_slm_dwords + wgq * static_cast<size_t>(t.kq_sg_per_wg_keys) + wgq);
}

// The tiling invariants ("Tiling" in docs/sdpa_ocl.md), independent of solve_sv_split().
void expect_valid_tiling(const SDPAOclTilingInfo& t, size_t vd_max, const std::string& what) {
    SCOPED_TRACE(what);
    EXPECT_EQ(t.sg_per_wg, t.kq_sg_per_wg_keys * t.kq_sg_per_wg_queries);
    EXPECT_EQ(t.sv_sg_per_wg_values * t.sv_sg_per_wg_scores, t.sg_per_wg) << "S*V must use exactly the KQ subgroups";
    EXPECT_EQ(static_cast<size_t>(t.sv_sg_tile_values * t.sv_sg_per_wg_values), vd_max);
    EXPECT_EQ(t.sv_sg_tile_scores * t.sv_sg_per_wg_scores, t.kq_wg_tile_queries);
    EXPECT_EQ(t.sv_sg_tile_values % t.subgroup_size, 0);
    EXPECT_EQ(t.sv_sg_tile_scores % 8, 0);
    for (int sg = 0; sg < t.sg_per_wg; sg++) {  // alpha[] nesting, every subgroup
        const int j0_kq = (sg / t.kq_sg_per_wg_keys) * t.kq_sg_tile_queries;
        const int i0_sv = (sg / t.sv_sg_per_wg_values) * t.sv_sg_tile_scores;
        EXPECT_GE(i0_sv, j0_kq) << "sg " << sg;
        EXPECT_LT(i0_sv + t.sv_sg_tile_scores - 1 - j0_kq, t.kq_sg_tile_queries) << "sg " << sg;
    }
    EXPECT_EQ(t.wg_size, t.sg_per_wg * t.subgroup_size);
}

struct scoped_env {
    std::vector<std::string> names;
    void set(const char* n, const char* v) {
        names.emplace_back(n);
#    ifdef _WIN32
        _putenv_s(n, v);
#    else
        setenv(n, v, 1);
#    endif
    }
    ~scoped_env() {
        for (const auto& n : names) {
#    ifdef _WIN32
            _putenv_s(n.c_str(), "");
#    else
            unsetenv(n.c_str());
#    endif
        }
    }
};
}  // namespace

TEST(SdpaOclHpg, Limits) {
    EXPECT_EQ(sdpa_ocl_max_slm_bytes(gpu_arch::xe_hpg), 64u * 1024);
    EXPECT_EQ(sdpa_ocl_max_slm_bytes(gpu_arch::xe2), 128u * 1024);
    EXPECT_EQ(sdpa_ocl_max_slm_bytes(gpu_arch::xe3), 128u * 1024);
    EXPECT_EQ(sdpa_ocl_max_wg_size(gpu_arch::xe_hpg), 1024u);
    EXPECT_EQ(sdpa_ocl_max_wg_size(gpu_arch::xe2), 1024u);
}

// Xe2 must not move: the tuned rows, their local memory and the sixteen-subgroup workgroup.
TEST(SdpaOclHpg, Xe2TilingUnchanged) {
    for (const auto arch : {gpu_arch::xe2, gpu_arch::xe3, gpu_arch::xe3p}) {
        for (const auto h : kHeads) {
            SDPAOclTilingInfo t;
            ASSERT_TRUE(sdpa_ocl_describe_tiling(arch, h, h, t)) << "head " << h;
            SCOPED_TRACE("head " + std::to_string(h));
            EXPECT_EQ(t.subgroup_size, 16);
            EXPECT_EQ(t.sg_per_wg, 16);
            EXPECT_EQ(t.wg_size, 256);
            EXPECT_EQ(t.kq_wg_tile_keys, 128);
            EXPECT_EQ(t.slm_bytes, oracle_slm(t, pow2_ceil(h)));
            EXPECT_LE(t.slm_bytes, sdpa_ocl_max_slm_bytes(arch));
            expect_valid_tiling(t, pow2_ceil(h), "xe2 head " + std::to_string(h));
        }
    }
    SDPAOclTilingInfo t;
    ASSERT_TRUE(sdpa_ocl_describe_tiling(gpu_arch::xe2, 128, 128, t));
    EXPECT_EQ(t.slm_bytes, 17536u);  // the documented default
    ASSERT_TRUE(sdpa_ocl_describe_tiling(gpu_arch::xe2, 512, 512, t));
    EXPECT_EQ(t.slm_bytes, 42112u);
}

// A tiling exists on xe_hpg for every head size and fits DG2 (64 KiB local memory, 1024 work-items).
TEST(SdpaOclHpg, XeHpgTilingCoversEveryHead) {
    for (const auto k : kHeads) {
        for (const auto v : kHeads) {
            SDPAOclTilingInfo t;
            const bool ok = sdpa_ocl_describe_tiling(gpu_arch::xe_hpg, k, v, t);
            if (k == v) {
                ASSERT_TRUE(ok) << "head " << k;
            }
            if (!ok)
                continue;
            const std::string what = "xe_hpg k " + std::to_string(k) + " v " + std::to_string(v);
            SCOPED_TRACE(what);
            EXPECT_EQ(t.subgroup_size, 8);
            EXPECT_LE(t.wg_size, 1024);
            EXPECT_LE(t.slm_bytes, 64u * 1024);
            EXPECT_EQ(t.slm_bytes, oracle_slm(t, pow2_ceil(k)));
            expect_valid_tiling(t, pow2_ceil(v), what);
            if (k == v) {  // the seed: 16 x 16 KQ tile, 4 x 2 subgroups (measured on DG2)
                EXPECT_EQ(t.kq_sg_tile_keys, 16);
                EXPECT_EQ(t.kq_sg_tile_queries, 16);
                EXPECT_EQ(t.kq_sg_per_wg_keys, 4);
                EXPECT_EQ(t.kq_sg_per_wg_queries, 2);
            }
        }
    }
}

TEST(SdpaOclHpg, HeadSizeBounds) {
    SDPAOclTilingInfo t;
    for (const auto arch : {gpu_arch::xe_hpg, gpu_arch::xe2}) {
        EXPECT_FALSE(sdpa_ocl_describe_tiling(arch, 0, 64, t));
        EXPECT_FALSE(sdpa_ocl_describe_tiling(arch, 64, 0, t));
        EXPECT_FALSE(sdpa_ocl_describe_tiling(arch, 576, 576, t));  // must return false, not assert
        EXPECT_FALSE(sdpa_ocl_describe_tiling(arch, 64, 1024, t));
    }
}

// The SDPA_OCL_KQ_* overrides go through the same fit check as the defaults: what choose_config() would assert on,
// supported() (via describe_tiling) refuses.
TEST(SdpaOclHpg, OverrideIsCheckedAgainstTheDevice) {
    SDPAOclTilingInfo t;
    ASSERT_TRUE(sdpa_ocl_describe_tiling(gpu_arch::xe_hpg, 128, 128, t));
    {
        scoped_env env;  // a milder override still fits: the control that makes the next one mean something
        env.set("SDPA_OCL_KQ_TILE_QUERIES", "32");
        ASSERT_TRUE(sdpa_ocl_describe_tiling(gpu_arch::xe_hpg, 128, 128, t));
        EXPECT_EQ(t.kq_sg_tile_queries, 32);
        EXPECT_LE(t.slm_bytes, 64u * 1024);
    }
    for (const auto arch : {gpu_arch::xe_hpg, gpu_arch::xe2}) {
        scoped_env env;  // 128 queries x 512 depth: the Q staging alone is 128 KiB
        env.set("SDPA_OCL_KQ_TILE_QUERIES", "64");
        EXPECT_FALSE(sdpa_ocl_describe_tiling(arch, 512, 512, t)) << static_cast<int>(arch);
    }
    {
        scoped_env env;  // 2048 work-items
        env.set("SDPA_OCL_KQ_PER_WG_KEYS", "128");
        EXPECT_FALSE(sdpa_ocl_describe_tiling(gpu_arch::xe_hpg, 128, 128, t));
    }
    ASSERT_TRUE(sdpa_ocl_describe_tiling(gpu_arch::xe_hpg, 128, 128, t));  // and nothing leaked
}

TEST(SdpaOclHpg, TierMaskCover) {
    EXPECT_FALSE(hpg_tiers_cover(0, ~0u)) << "an op that needs no bit must not pass by accident";
    EXPECT_FALSE(hpg_tiers_cover(PLAIN_F16_STATIC, 0));
    EXPECT_TRUE(hpg_tiers_cover(PLAIN_F16_STATIC, PLAIN_F16_STATIC));
    EXPECT_FALSE(hpg_tiers_cover(PLAIN_F16_STATIC | PLAIN_EXT, PLAIN_F16_STATIC)) << "every bit the op touches is needed";
    EXPECT_TRUE(hpg_tiers_cover(PLAIN_F16_STATIC | PLAIN_EXT, PLAIN_F16_STATIC | PLAIN_EXT | PA_U4));
    EXPECT_FALSE(hpg_tiers_cover(PA_PREFILL | PA_MIXED_F16, PA_PREFILL));
    // The step that ports a kernel family turns its bit on (and edits this line): S6a = the plain f16 static prefill,
    // S6b = the rest of plain SDPA, S6c = plain SDPA on an i8 KV cache.
    EXPECT_EQ(kHpgTiersReady, static_cast<uint32_t>(PLAIN_F16_STATIC | PLAIN_EXT | PLAIN_I8));
    if (std::getenv("SDPA_OCL_HPG_TIERS") == nullptr) {
        EXPECT_EQ(hpg_tiers_ready(), static_cast<uint32_t>(PLAIN_F16_STATIC | PLAIN_EXT | PLAIN_I8));
        EXPECT_TRUE(hpg_tier_ready(PLAIN_F16_STATIC));
        EXPECT_TRUE(hpg_tier_ready(PLAIN_EXT));
        EXPECT_TRUE(hpg_tier_ready(PLAIN_I8));
        EXPECT_FALSE(hpg_tier_ready(PA_PREFILL));
        EXPECT_FALSE(hpg_tiers_cover(PA_PREFILL | PA_MIXED_F16, hpg_tiers_ready()));
    }
}
#endif  // ENABLE_ONEDNN_FOR_GPU
