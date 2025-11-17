// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// AMX INT8 GEMV (u8 X, s8 W) -> fp32 Y microkernel (N=1)
#include "amx_gemmv_intrin.hpp"
#include "jit_prebuilt_pool.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/amx_tile_configure.hpp"
#include "jit_gemmv_amx_bf16.hpp"
#include "jit_gemmv_amx_int8.hpp"
#include "openvino/core/except.hpp"
#include "gemmv_force_isa.hpp"
#include "xbyak/xbyak_util.h"
#include <mutex>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cmath>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <immintrin.h>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <setjmp.h>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sys/syscall.h>
#include <cerrno>
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef ARCH_GET_XCOMP_PERM
#define ARCH_GET_XCOMP_PERM 0x1022
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
#ifndef XFEATURE_XTILECFG
#define XFEATURE_XTILECFG 17
#endif
#endif


#if defined(_WIN32)
#include <malloc.h>
#endif

namespace ov::intel_cpu::x64::gemmv_jit {

namespace {

template <typename Fn>
static bool amx_safe_invoke(Fn&& fn) {
    struct sigaction old_segv{}, old_ill{};
    static thread_local sigjmp_buf buf;
    static thread_local int last_signal = 0;
    static thread_local void* last_addr = nullptr;
    auto handler = +[](int sig, siginfo_t* info, void*) {
        last_signal = sig;
        last_addr = info ? info->si_addr : nullptr;
        siglongjmp(buf, 1);
    };
    struct sigaction sa{};
    sa.sa_sigaction = handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGILL,  &sa, &old_ill);
    bool ok = true;
    if (sigsetjmp(buf, 1) == 0) {
        fn();
    } else {
        ok = false;
    }
    sigaction(SIGSEGV, &old_segv, nullptr);
    sigaction(SIGILL,  &old_ill,  nullptr);
    if (!ok) {
        if (const char* env = std::getenv("GEMMV_AMX_DIAG"); env && env[0] != '\0') {
            std::fprintf(stderr, "[GEMMV][AMX] trapped signal %d inside AMX kernel (addr=%p)\n",
                         last_signal, last_addr);
        }
    }
    return ok;
}

using dnnl::impl::status::success;
using dnnl::impl::status_t;
using dnnl::impl::cpu::x64::cpu_isa_t;
using dnnl::impl::cpu::x64::mayiuse;
using dnnl::impl::cpu::x64::amx_tile_configure;
using dnnl::impl::cpu::x64::amx_tile_release;

static bool amx_diag_enabled() {
    const char* env = std::getenv("GEMMV_AMX_DIAG");
    return env && env[0] != '\0';
}

static bool amx_ref_enabled() {
    const char* env = std::getenv("GEMMV_AMX_REF");
    return env && env[0] != '\0';
}

static bool amx_dump_tiles_enabled() {
    const char* env = std::getenv("GEMMV_AMX_DUMP_TILES");
    return env && env[0] != '\0';
}

constexpr int k_amx_i8_m_blk = 16;
constexpr int k_amx_i8_k_blk = 64;
constexpr int k_amx_i8_vnni_step = 4;
constexpr int k_amx_i8_b_rows = k_amx_i8_k_blk / k_amx_i8_vnni_step; // 16 rows (rd_block/rd_step)
// B tile (packed X, VNNI step=4): colsb must match brgemm_init_tiles for N=1
constexpr int k_amx_i8_b_colsb = 4; // single VNNI group per row (W)
constexpr int k_amx_i8_b_tile_bytes = k_amx_i8_b_rows * k_amx_i8_b_colsb;
constexpr int k_amx_i8_a_tile_bytes = k_amx_i8_k_blk * k_amx_i8_m_blk; // X packed as A (cols=64)

constexpr int k_amx_bf16_m_blk = 16;
constexpr int k_amx_bf16_k_blk = 64;
constexpr size_t k_amx_bf16_a_tile_elems
        = static_cast<size_t>(k_amx_bf16_m_blk) * k_amx_bf16_k_blk;
constexpr size_t k_amx_bf16_b_tile_elems
        = static_cast<size_t>(k_amx_bf16_k_blk) * k_amx_bf16_m_blk;

constexpr int k_tilecfg_bytes_per_row_c = 4;
constexpr int k_tilecfg_bytes_per_row_a = k_amx_i8_k_blk; // X colsb=64
constexpr int k_tilecfg_bytes_per_row_b = k_amx_i8_b_colsb; // W colsb=4
constexpr int k_tilecfg_rows_c = k_amx_i8_m_blk;
constexpr int k_tilecfg_rows_a = k_amx_i8_m_blk;
constexpr int k_tilecfg_rows_b = k_amx_i8_b_rows;

template <typename T, size_t Alignment>
struct aligned_allocator {
    using value_type = T;

    aligned_allocator() noexcept = default;

    template <typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        void* ptr = nullptr;
#if defined(_WIN32)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
#else
        const std::size_t bytes = n * sizeof(T);
        if (posix_memalign(&ptr, Alignment, bytes) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
#if defined(_WIN32)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    template <class U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };
};

template <typename T>
using aligned_vector64 = std::vector<T, aligned_allocator<T, 64>>;

struct QuantCache {
    aligned_vector64<uint8_t> xq;
    aligned_vector64<uint8_t> xq_padded;
};
thread_local QuantCache g_quant_cache;

struct alignas(64) amx_tilecfg_t {
    uint8_t palette_id = 0;
    uint8_t start_row = 0;
    uint8_t reserved[14]{};
    uint16_t colsb[16]{};
    uint8_t rows[16]{};
};

static amx_tilecfg_t make_bf16_palette() {
    amx_tilecfg_t cfg{};
    cfg.palette_id = 1;
    cfg.colsb[0] = 64;  cfg.rows[0] = 16; // C (fp32)
    cfg.colsb[1] = 128; cfg.rows[1] = 16; // A (bf16)
    cfg.colsb[2] = 32;  cfg.rows[2] = 64; // B (bf16)
    return cfg;
}

static amx_tilecfg_t make_int8_gemv_palette() {
    amx_tilecfg_t cfg{};
    cfg.palette_id = 1;
    cfg.colsb[0] = k_tilecfg_bytes_per_row_c;
    cfg.rows[0] = k_tilecfg_rows_c;
    cfg.colsb[1] = k_tilecfg_bytes_per_row_a;
    cfg.rows[1] = k_tilecfg_rows_a;
    cfg.colsb[2] = k_tilecfg_bytes_per_row_b;
    cfg.rows[2] = k_tilecfg_rows_b;
    return cfg;
}

static bool ensure_amx_thread_state() {
    const bool force_amx = (get_gemmv_force_isa() == gemmv_force_isa_t::amx_int8) ||
                           (get_gemmv_force_isa() == gemmv_force_isa_t::amx_bf16);
    auto request_amx_permissions = []() -> bool {
#if defined(__linux__)
        constexpr unsigned long k_mask_cfg = 1ul << XFEATURE_XTILECFG;
        constexpr unsigned long k_mask_data = 1ul << XFEATURE_XTILEDATA;
        auto get_mask = []() -> unsigned long {
            unsigned long bitmask = 0;
            long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
            return status == 0 ? bitmask : 0ul;
        };
        unsigned long have = get_mask();
        auto request_mask = [&](int feature_bit) -> bool {
            errno = 0;
            long status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, feature_bit);
            if (status != 0 && errno != EBUSY && errno != EEXIST) {
                return false;
            }
            have = get_mask();
            return (have & (1ul << feature_bit)) != 0;
        };
        const bool have_cfg = (have & k_mask_cfg) == k_mask_cfg ? true : request_mask(XFEATURE_XTILECFG);
        const bool have_data = (have & k_mask_data) == k_mask_data ? true : request_mask(XFEATURE_XTILEDATA);
        return have_cfg && have_data;
#else
        return true;
#endif
    };
    const bool have_perm = request_amx_permissions();
    if (!have_perm) {
        if (amx_diag_enabled()) {
            std::fprintf(stderr, "[GEMMV][AMX] ARCH_REQ_XCOMP_PERM failed\n");
        }
        return false;
    }
#if defined(__x86_64__) || defined(_M_X64)
    const uint64_t need = (1ull << 17) | (1ull << 18);
    const uint64_t cur = _xgetbv(0);
    if ((cur & need) != need) {
        _xsetbv(0, cur | need);
        if (amx_diag_enabled()) {
            std::fprintf(stderr, "[GEMMV][AMX] xsetbv set bits 17/18 (new=0x%llx)\n",
                         static_cast<unsigned long long>((cur | need)));
        }
    }
    // Clear XFD bits for AMX tilecfg/tiledata to avoid SIGILL on first AMX instruction
    const uint64_t xfd_mask = (uint64_t(1) << 17) | (uint64_t(1) << 18);
    const uint64_t xfd_cur = _xgetbv(1);
    if (xfd_cur & xfd_mask) {
        const uint64_t xfd_new = xfd_cur & ~xfd_mask;
        _xsetbv(1, xfd_new);
        if (amx_diag_enabled()) {
            std::fprintf(stderr, "[GEMMV][AMX] cleared XFD bits 17/18 (old=0x%llx new=0x%llx)\n",
                         (unsigned long long)xfd_cur,
                         (unsigned long long)xfd_new);
        }
    } else if (amx_diag_enabled()) {
        std::fprintf(stderr, "[GEMMV][AMX] XFD already clear (xfd=0x%llx)\n", (unsigned long long)xfd_cur);
    }
#endif
    // After XCR0/XFD setup do a late availability check unless forced
    if (!force_amx) {
        using dnnl::impl::cpu::x64::amx::is_available;
        const bool ok = is_available();
        if (amx_diag_enabled()) {
            std::fprintf(stderr, "[GEMMV][AMX] dnnl::amx::is_available() = %d\n", ok ? 1 : 0);
        }
        if (!ok) {
            return false;
        }
    }
    return true;
}

static inline bool configure_amx_palette(const amx_tilecfg_t& cfg) {
    if (!ensure_amx_thread_state()) {
        return false;
    }
    const auto status = dnnl::impl::cpu::x64::amx_tile_configure(
            reinterpret_cast<const char*>(&cfg));
    return status == dnnl::impl::status::success;
}

static inline bool release_amx_palette() {
    const auto status = dnnl::impl::cpu::x64::amx_tile_release();
    return status == dnnl::impl::status::success;
}

static inline uint16_t fp32_to_bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

static void pack_xbf16_tiles(const float* x_fp32, int K, int K_blk, uint16_t* dst) {
    const int K_grp = (K + K_blk - 1) / K_blk;
    alignas(64) uint16_t row_buf[k_amx_bf16_k_blk];
    for (int g = 0; g < K_grp; ++g) {
        const int k0 = g * K_blk;
        const int kk = std::min(K_blk, K - k0);
        for (int k = 0; k < K_blk; ++k) {
            row_buf[k] = (k < kk) ? fp32_to_bf16(x_fp32[k0 + k]) : uint16_t{0};
        }
        uint16_t* tile = dst + static_cast<size_t>(g) * k_amx_bf16_a_tile_elems;
        for (int r = 0; r < k_amx_bf16_m_blk; ++r) {
            std::memcpy(tile + r * K_blk, row_buf, K_blk * sizeof(uint16_t));
        }
    }
}

static void pack_xu8_tiles_a64(const uint8_t* xq, int K, int32_t zp_x,
        int K_blk, uint8_t* dst) {
    // X packed as A-tile (row-major, colsb=64, rows=16)
    const int K_grp = (K + K_blk - 1) / K_blk;
    for (int g = 0; g < K_grp; ++g) {
        const int k0 = g * K_blk;
        uint8_t* tile = dst + static_cast<size_t>(g) * static_cast<size_t>(k_amx_i8_a_tile_bytes);
        for (int r = 0; r < k_amx_i8_m_blk; ++r) {
            uint8_t* row = tile + static_cast<size_t>(r) * k_amx_i8_k_blk;
            for (int kb = 0; kb < K_blk; ++kb) {
                const int idx = k0 + kb;
                row[kb] = (idx < K) ? xq[idx] : static_cast<uint8_t>(zp_x & 0xFF);
            }
        }
    }
}

static void dump_tilecfg(const char* tag, const amx_tilecfg_t& cfg) {
    if (!amx_diag_enabled()) {
        return;
    }
    std::fprintf(stderr, "[GEMMV][AMX] palette[%s]\n", tag);
    for (int i = 0; i < 8; ++i) {
        std::fprintf(stderr, "  T%d: colsb=%d rows=%d\n", i,
                     static_cast<int>(cfg.colsb[i]),
                     static_cast<int>(cfg.rows[i]));
    }
}

} // namespace

// Helper: per-lane metadata getters
static inline float get_scale_lane(const float* s, int idx, quant_granularity_t gran, int group_size) {
    if (!s) return 1.f;
    if (gran == quant_granularity_t::per_tensor) return s[0];
    if (gran == quant_granularity_t::per_channel) return s[idx];
    const int gs = group_size > 0 ? group_size : 16;
    return s[idx / gs];
}
static inline int32_t get_zp_lane(const int32_t* z, int idx, quant_granularity_t gran, int group_size) {
    if (!z) return 0;
    if (gran == quant_granularity_t::per_tensor) return z[0];
    if (gran == quant_granularity_t::per_channel) return z[idx];
    const int gs = group_size > 0 ? group_size : 16;
    return z[idx / gs];
}
static inline float get_bias_lane(const float* b, int idx, quant_granularity_t gran, int group_size) {
    if (!b) return 0.f;
    if (gran == quant_granularity_t::per_tensor) return b[0];
    if (gran == quant_granularity_t::per_channel) return b[idx];
    const int gs = group_size > 0 ? group_size : 16;
    return b[idx / gs];
}

bool run_amx_tile_probe() {
    amx_tilecfg_t cfg{};
    cfg.palette_id = 1;
    cfg.colsb[0] = k_tilecfg_bytes_per_row_c;
    cfg.rows[0] = k_tilecfg_rows_c;
    cfg.colsb[1] = k_tilecfg_bytes_per_row_a;
    cfg.rows[1] = k_tilecfg_rows_a;
    cfg.colsb[2] = k_tilecfg_bytes_per_row_b;
    cfg.rows[2] = k_tilecfg_rows_b;
    const bool loaded = configure_amx_palette(cfg);
    if (!loaded) {
        if (amx_diag_enabled()) {
            std::fprintf(stderr, "[GEMMV][AMX] tile probe: ldtilecfg faulted\n");
        }
        return false;
    }
    const bool released = release_amx_palette();
    if (amx_diag_enabled()) {
        std::fprintf(stderr, "[GEMMV][AMX] tile probe: release %s\n", released ? "ok" : "failed");
    }
    return released;
}

#if defined(__GNUC__)
__attribute__((target("amx-int8,amx-tile,avx512bw,avx512vnni,avx512f")))
#endif
static inline void amx_epilogue_store_fp32(int valid, int base_idx,
                                           float* yb,
                                           const int32_t* Cbuf_row0,
                                           const int32_t* Sbuf_row0,
                                           const int32_t* sumW_precomp,
                                           int K_ker, int32_t zp_x, int32_t sum_x_q_padded,
                                           const float* scales, const int32_t* zps, const float* bias,
                                           quant_granularity_t gran, int group_size,
                                           float s_x,
                                           const float* lane_scales,
                                           const int32_t* lane_zps,
                                           const float* lane_bias,
                                           int32_t pad_w) {
    __mmask16 km = valid >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << valid) - 1u);
    __m512i acc_v = _mm512_maskz_loadu_epi32(km, Cbuf_row0);
    __m512i sumw_v;
    if (sumW_precomp) {
        sumw_v = _mm512_maskz_loadu_epi32(km, sumW_precomp + base_idx);
    } else {
        sumw_v = _mm512_maskz_loadu_epi32(km, Sbuf_row0);
    }
    if (pad_w != 0) {
        sumw_v = _mm512_add_epi32(sumw_v, _mm512_set1_epi32(pad_w));
    }
    __m512i zpw_v = _mm512_setzero_si512();
    if (lane_zps) {
        zpw_v = _mm512_maskz_loadu_epi32(km, lane_zps);
    } else if (zps) {
        alignas(64) int32_t zpw_buf[16];
        if (gran == quant_granularity_t::per_tensor) {
            for (int m = 0; m < 16; ++m) zpw_buf[m] = zps[0];
        } else if (gran == quant_granularity_t::per_channel) {
            for (int m = 0; m < 16; ++m) zpw_buf[m] = zps[base_idx + m];
        } else {
            const int gs = group_size > 0 ? group_size : 16;
            for (int m = 0; m < 16; ++m) { int g = (base_idx + m) / gs; zpw_buf[m] = zps[g]; }
        }
        zpw_v = _mm512_maskz_loadu_epi32(km, zpw_buf);
    }
    __m512i neg_zpx_v = _mm512_set1_epi32(-zp_x);
    __m512i comp_v = _mm512_mullo_epi32(neg_zpx_v, sumw_v);
    __m512i c_v = _mm512_set1_epi32(K_ker * zp_x - sum_x_q_padded);
    __m512i zpw_term = _mm512_mullo_epi32(zpw_v, c_v);
    comp_v = _mm512_add_epi32(comp_v, zpw_term);
    __m512 yf = _mm512_cvtepi32_ps(_mm512_add_epi32(acc_v, comp_v));
    __m512 s_v;
    if (lane_scales) {
        s_v = _mm512_maskz_loadu_ps(km, lane_scales);
        s_v = _mm512_mul_ps(s_v, _mm512_set1_ps(s_x));
    } else if (!scales) {
        s_v = _mm512_set1_ps(1.f * s_x);
    } else if (gran == quant_granularity_t::per_tensor) {
        s_v = _mm512_set1_ps(scales[0] * s_x);
    } else {
        alignas(64) float sc_buf[16];
        if (gran == quant_granularity_t::per_channel) {
            for (int m = 0; m < 16; ++m) sc_buf[m] = scales[base_idx + m] * s_x;
        } else {
            const int gs = group_size > 0 ? group_size : 16;
            for (int m = 0; m < 16; ++m) { int g = (base_idx + m) / gs; sc_buf[m] = scales[g] * s_x; }
        }
        s_v = _mm512_maskz_loadu_ps(km, sc_buf);
    }
    yf = _mm512_mul_ps(yf, s_v);
    if (lane_bias) {
        __m512 b_v = _mm512_maskz_loadu_ps(km, lane_bias);
        yf = _mm512_add_ps(yf, b_v);
    } else if (bias) {
        __m512 b_v;
        if (gran == quant_granularity_t::per_tensor) {
            b_v = _mm512_set1_ps(bias[0]);
        } else {
            alignas(64) float b_buf[16];
            if (gran == quant_granularity_t::per_channel) {
                for (int m = 0; m < 16; ++m) b_buf[m] = bias[base_idx + m];
            } else {
                const int gs = group_size > 0 ? group_size : 16;
                for (int m = 0; m < 16; ++m) { int g = (base_idx + m) / gs; b_buf[m] = bias[g]; }
            }
            b_v = _mm512_maskz_loadu_ps(km, b_buf);
        }
        yf = _mm512_add_ps(yf, b_v);
    }
    bool do_nt = true;
    if (valid >= 16 && do_nt && (((uintptr_t)yb) & 63u) == 0u) _mm512_stream_ps(yb, yf);
    else _mm512_mask_storeu_ps(yb, km, yf);
}

#if defined(__GNUC__)
__attribute__((target("amx-int8,amx-tile,avx512bw,avx512vnni,avx512f")))
#endif
bool run_gemmv_amx_i8u8_fp32(const float* x_fp32, int K,
                             const uint8_t* wq, int M, int ld_w_bytes,
                             const float* scales, const int32_t* zps,
                             float* y, const float* bias,
                             quant_granularity_t gran, int group_size,
                             const int32_t* sumW_precomp,
                             const amx_lane_meta_t* lane_meta,
                             const uint8_t* x_q8,
                             float x_scale,
                             int32_t x_zp,
                             int32_t sum_x_q) {
    const bool force_amx = (get_gemmv_force_isa() == gemmv_force_isa_t::amx_int8);
    if (!force_amx && !mayiuse(cpu_isa_t::amx_int8)) {
        return false;
    }
    const bool ref_enabled = amx_ref_enabled();
    // To avoid platform SIGILL under diag flags, we suppress diag side-effects when ref is requested.
    const bool diag = amx_diag_enabled() && !ref_enabled;
    if (diag) {
        std::fprintf(stderr, "[GEMMV][AMX][dbg] run_gemmv_amx_i8u8_fp32 diag=1\n");
        std::fflush(stderr);
        static std::atomic<bool> palette_dumped{false};
        if (!palette_dumped.exchange(true)) {
            dump_tilecfg("gemmv", make_int8_gemv_palette());
        }
    }

    if (!ensure_amx_thread_state()) {
        if (diag) {
            std::fprintf(stderr, "[GEMMV][AMX] unable to enable AMX state for the current thread\n");
        }
        return false;
    }

    const int K_blk = k_amx_i8_k_blk;
    const int K_blocks = (K + K_blk - 1) / K_blk;
    const int K_ker = K_blocks * K_blk; // padded length actually processed by tiles
    if (K_blocks <= 0) {
        return false;
    }
    const int32_t pad_w = (zps && gran == quant_granularity_t::per_tensor)
                          ? (K_ker - K) * zps[0]
                          : 0;

    auto& qcache = g_quant_cache;
    const bool use_prequant = (x_q8 != nullptr);
    float s_x = x_scale;
    int32_t zp_x = x_zp;
    int32_t sum_x_q_eff = sum_x_q;

    if (!use_prequant) {
        if (qcache.xq.size() != static_cast<size_t>(K)) {
            qcache.xq.assign(static_cast<size_t>(K), 0);
        }
        float amax = 0.f;
        for (int k = 0; k < K; ++k) {
            amax = std::max(amax, std::fabs(x_fp32[k]));
        }
        s_x = (amax > 0.f) ? (amax / 127.f) : 1.f;
        zp_x = 0;
        sum_x_q_eff = 0;
        for (int k = 0; k < K; ++k) {
            int v = static_cast<int>(std::lrintf(x_fp32[k] / s_x));
            v = std::min(127, std::max(-128, v));
            qcache.xq[static_cast<size_t>(k)] = static_cast<uint8_t>(static_cast<int8_t>(v));
            sum_x_q_eff += v;
        }
    }

    const uint8_t* xq_src = use_prequant ? x_q8 : qcache.xq.data();

    const size_t tile_bytes = static_cast<size_t>(K_blocks) * static_cast<size_t>(k_amx_i8_a_tile_bytes);
    if (qcache.xq_padded.size() != tile_bytes) {
        qcache.xq_padded.assign(tile_bytes, 0);
    }
    pack_xu8_tiles_a64(xq_src, K, zp_x, K_blk, qcache.xq_padded.data());
    const int32_t sum_x_q_padded = sum_x_q_eff; // symmetric zp_x=0

    auto fn = jit_prebuilt_pool::get_typed<jit_amx_gemmv_int8_t::kernel_fn>(kernel_kind::amx_int8);
    if (!fn) {
        return false;
    }

    alignas(64) amx_tilecfg_t cfg = make_int8_gemv_palette();
    if (diag) {
        const uint8_t* code = reinterpret_cast<const uint8_t*>(fn);
        std::fprintf(stderr, "[GEMMV][AMX][dbg] fn=%p code[0..31]:", (const void*)fn);
        for (int i = 0; i < 32; ++i) {
            std::fprintf(stderr, " %02x", code[i]);
        }
        std::fprintf(stderr, "\n");
    }

    const int M_blk = k_amx_i8_m_blk;
    const int num_full = M / M_blk;
    const int M_tail = M % M_blk;
    auto execute_block = [&](int block_idx, int rows) -> bool {
        const size_t row_offset_bytes = static_cast<size_t>(block_idx) * static_cast<size_t>(ld_w_bytes);
        const uint8_t* wblk = wq + row_offset_bytes;

        // C tile stores 16 rows with 64-byte stride => 16 lanes * 16 dwords
        alignas(64) int32_t Cbuf[k_amx_i8_m_blk * 16];
        alignas(64) int32_t runtime_sum[k_amx_i8_m_blk];
        std::fill(std::begin(Cbuf), std::end(Cbuf), 0);

        bool ok = true;
        const size_t a_tile_bytes = static_cast<size_t>(k_amx_i8_a_tile_bytes); // X tiles
        const size_t b_tile_bytes = static_cast<size_t>(k_amx_i8_b_tile_bytes); // W tiles (colsb=4)
        jit_amx_gemmv_int8_call_args args{};
        args.a_ptr = qcache.xq_padded.data(); // X (u8)
        args.b_ptr = wblk;                    // W (s8) packed as k4
        args.a_tile_bytes = a_tile_bytes;
        args.b_group_bytes = b_tile_bytes;
        args.k_blocks = static_cast<size_t>(K_blocks);
        args.c_out = Cbuf;
        args.tilecfg = &cfg;
        ok = amx_safe_invoke([&]() { fn(&args); });
        if (!ok) {
            return false;
        }

        const int base_idx = block_idx * M_blk;
        const int32_t* sum_block = nullptr;
        if (sumW_precomp) {
            sum_block = sumW_precomp + base_idx;
        } else {
            std::fill(runtime_sum, runtime_sum + k_amx_i8_m_blk, 0);
            const int K_grp4 = (K + k_amx_i8_vnni_step - 1) / k_amx_i8_vnni_step;
            for (int im = 0; im < rows; ++im) {
                int32_t acc = 0;
                for (int g = 0; g < K_grp4; ++g) {
                    const int valid = std::min(k_amx_i8_vnni_step, K - g * k_amx_i8_vnni_step);
                    const uint8_t* row_ptr = wblk
                            + static_cast<size_t>(g) * static_cast<size_t>(k_amx_i8_m_blk * k_amx_i8_vnni_step)
                            + static_cast<size_t>(im) * static_cast<size_t>(k_amx_i8_vnni_step);
                    for (int c = 0; c < valid; ++c) {
                        acc += static_cast<int32_t>(static_cast<int8_t>(row_ptr[c]));
                    }
                }
                runtime_sum[im] = acc + pad_w;
            }
            sum_block = runtime_sum;
        }
        float* y_out = y + base_idx;
        const float* block_scales = lane_meta && lane_meta->scales
                ? lane_meta->scales + base_idx
                : nullptr;
        const int32_t* block_zps = lane_meta && lane_meta->zps
                ? lane_meta->zps + base_idx
                : nullptr;
        const float* block_bias = lane_meta && lane_meta->bias
                ? lane_meta->bias + base_idx
                : nullptr;

        // Extract first dword from each 64-byte C row
        alignas(64) int32_t C_lane[k_amx_i8_m_blk];
        for (int im = 0; im < k_amx_i8_m_blk; ++im) {
            C_lane[im] = Cbuf[im * 16];
        }

        amx_epilogue_store_fp32(rows, base_idx, y_out, C_lane, /*runtime_sum*/ nullptr,
                sum_block, K_ker, zp_x, sum_x_q_padded, scales, zps, bias, gran, group_size,
                s_x, block_scales, block_zps, block_bias, pad_w);

        if (ref_enabled) {
            if (block_idx == 0) {
                float s_w_lane[16];
                int32_t zp_w_lane[16];
                float b_lane[16];
                for (int m = 0; m < M_blk; ++m) {
                    const int idx = base_idx + m;
                    s_w_lane[m] = 1.f;
                    zp_w_lane[m] = 0;
                    b_lane[m] = 0.f;
                    if (block_scales) s_w_lane[m] = block_scales[m];
                    else if (scales) {
                        if (gran == quant_granularity_t::per_tensor) s_w_lane[m] = scales[0];
                        else if (gran == quant_granularity_t::per_channel) s_w_lane[m] = scales[idx];
                        else { const int gs = group_size > 0 ? group_size : 16; s_w_lane[m] = scales[idx / gs]; }
                    }
                    if (block_zps) zp_w_lane[m] = block_zps[m];
                    else if (zps) {
                        if (gran == quant_granularity_t::per_tensor) zp_w_lane[m] = zps[0];
                        else if (gran == quant_granularity_t::per_channel) zp_w_lane[m] = zps[idx];
                        else { const int gs = group_size > 0 ? group_size : 16; zp_w_lane[m] = zps[idx / gs]; }
                    }
                    if (block_bias) b_lane[m] = block_bias[m];
                    else if (bias) {
                        if (gran == quant_granularity_t::per_tensor) b_lane[m] = bias[0];
                        else if (gran == quant_granularity_t::per_channel) b_lane[m] = bias[idx];
                        else { const int gs = group_size > 0 ? group_size : 16; b_lane[m] = bias[idx / gs]; }
                    }
                }
                float y_ref[M_blk];
                for (int im = 0; im < rows; ++im) {
                    int32_t acc = 0;
                    // Walk actual tile packing: W row-major in A tiles, X packed K-major in 64x4 B tiles
                    for (int g = 0; g < K_blocks; ++g) {
                        const uint8_t* wtile = wblk + static_cast<size_t>(g) * static_cast<size_t>(k_amx_i8_m_blk * k_amx_i8_k_blk);
                        const uint8_t* xtile = qcache.xq_padded.data() + static_cast<size_t>(g) * static_cast<size_t>(k_amx_i8_b_tile_bytes);
                        for (int kr = 0; kr < k_amx_i8_b_rows; ++kr) {
                            const int k_idx = g * k_amx_i8_k_blk + kr;
                            const int xv = (k_idx < K)
                                           ? static_cast<int>(xtile[static_cast<size_t>(kr) * k_amx_i8_b_colsb])
                                           : static_cast<int>(zp_x);
                            const int8_t wv = static_cast<int8_t>(wtile[im * k_amx_i8_k_blk + kr]);
                            acc += static_cast<int32_t>(wv) * static_cast<int32_t>(xv);
                        }
                    }
                    const int32_t sumw = sum_block ? sum_block[im] : 0;
                    const int32_t comp = (-zp_x) * sumw + zp_w_lane[im] * (K_ker * zp_x - sum_x_q_padded);
                    const float yv = (static_cast<float>(acc + comp) * (s_w_lane[im] * s_x)) + b_lane[im];
                    y_ref[im] = yv;
                }
                float max_diff = 0.f;
                float max_y = 0.f;
                for (int im = 0; im < rows; ++im) {
                    float diff = std::fabs(y_out[im] - y_ref[im]);
                    max_diff = std::max(max_diff, diff);
                    max_y = std::max(max_y, std::fabs(y_ref[im]));
                }
                std::fprintf(stderr, "[GEMMV][AMX][ref] block0 rows=%d max_diff=%g max_ref=%g\n", rows, max_diff, max_y);
                for (int im = 0; im < std::min(rows, 4); ++im) {
                    std::fprintf(stderr, "  m=%d y=%g ref=%g diff=%g\n", base_idx + im, y_out[im], y_ref[im], y_out[im]-y_ref[im]);
                }
            }
        }
        return true;
    };

    bool ok = true;
    for (int bi = 0; bi < num_full; ++bi) {
        if (!execute_block(bi, M_blk)) {
            ok = false;
            break;
        }
    }
    if (ok && M_tail) {
        if (!execute_block(num_full, M_tail)) {
            ok = false;
        }
    }
    return ok;
}

#if defined(__GNUC__)
__attribute__((target("amx-bf16,amx-tile,avx512f")))
#endif
bool run_gemmv_amx_bf16_fp32(const float* x_fp32, int K,
                             const uint16_t* w_bf16_k64, int M, int ld_w_kbytes,
                             float* y, const float* bias) {
    if (!mayiuse(cpu_isa_t::amx_bf16)) {
        return false;
    }
    const bool diag = amx_diag_enabled();

    const int K_blk = k_amx_bf16_k_blk;
    const int M_blk = k_amx_bf16_m_blk;
    const int K_grp = (K + K_blk - 1) / K_blk;
    if (K_grp <= 0) {
        return false;
    }

    aligned_vector64<uint16_t> packed_a;
    try {
        packed_a.resize(static_cast<size_t>(K_grp) * k_amx_bf16_a_tile_elems);
    } catch (const std::bad_alloc&) {
        return false;
    }
    pack_xbf16_tiles(x_fp32, K, K_blk, packed_a.data());

    const auto fn = jit_prebuilt_pool::get_typed<jit_amx_gemmv_bf16_t::kernel_fn>(kernel_kind::amx_bf16);
    OPENVINO_ASSERT(fn != nullptr, "AMX bf16 kernel pointer is null");
    const size_t a_tile_bytes = static_cast<size_t>(M_blk) * K_blk * sizeof(uint16_t);
    const size_t b_group_bytes = static_cast<size_t>(K_blk) * M_blk * sizeof(uint16_t);
    const int M_full = M / M_blk;
    const int M_tail = M % M_blk;

    auto run_block = [&](int bi, int valid) {
        const uint16_t* wblk = w_bf16_k64 + static_cast<size_t>(bi) * static_cast<size_t>(ld_w_kbytes / sizeof(uint16_t));
        alignas(64) float Cbuf[M_blk * M_blk];

        jit_amx_gemmv_bf16_call_args args{};
        args.a_ptr = packed_a.data();
        args.b_ptr = wblk;
        args.a_tile_bytes = a_tile_bytes;
        args.b_group_bytes = b_group_bytes;
        args.k_blocks = static_cast<size_t>(K_grp);
        args.c_out = Cbuf;
        fn(&args);

        __mmask16 km = valid >= M_blk ? (__mmask16)0xFFFF : (__mmask16)((1u << valid) - 1u);
        __m512 cv = _mm512_maskz_loadu_ps(km, Cbuf);
        if (bias) {
            __m512 b = _mm512_set1_ps(bias[0]);
            cv = _mm512_add_ps(cv, _mm512_maskz_mov_ps(km, b));
        }
        float* yb = y + bi * M_blk;
        if (valid >= M_blk && ((((uintptr_t)yb) & 63u) == 0u)) {
            _mm512_stream_ps(yb, cv);
        } else {
            _mm512_mask_storeu_ps(yb, km, cv);
        }
    };

    const bool disable_guard = [](){
        const char* env = std::getenv("GEMMV_AMX_NO_GUARD");
        return env && env[0] != '\0';
    }();
    bool ok = true;
    auto invoke_kernel = [&]() {
        amx_tilecfg_t cfg = make_bf16_palette();
        if (!configure_amx_palette(cfg)) {
            if (diag) {
                std::fprintf(stderr, "[GEMMV][AMX] load_palette failed (bf16 path)\n");
            }
            ok = false;
            return;
        }
        for (int bi = 0; bi < M_full; ++bi) {
            run_block(bi, M_blk);
        }
        if (M_tail) {
            run_block(M_full, M_tail);
        }
        if (!release_amx_palette() && diag) {
            std::fprintf(stderr, "[GEMMV][AMX] release failed (bf16 path)\n");
        }
    };
    if (disable_guard) {
        invoke_kernel();
    } else {
        ok = amx_safe_invoke(invoke_kernel) && ok;
    }
    if (!ok && diag) {
        std::fprintf(stderr, "[GEMMV][AMX] kernel execution failed (bf16 path)\n");
    }
    return ok;
}
} // namespace ov::intel_cpu::x64::gemmv_jit
