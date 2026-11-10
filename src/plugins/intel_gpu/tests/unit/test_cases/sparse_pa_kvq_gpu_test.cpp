// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the quantized (u8 / u4, symmetric / asymmetric, BY_TOKEN / BY_CHANNEL) main KV caches of the paged
// SparsePA primitive. A CPU model of the page format (see intel_gpu/op/sparse_kv_quant.hpp) quantizes the history,
// predicts the cache after the kernel's write stage, and dequantizes the cache the GPU actually produced:
//   1. the GPU cache must match the predicted one up to rounding-edge code differences,
//   2. the attention output must match a reference that reads the GPU cache through the same dequantization
//      (isolates the attention kernels' dequant-on-load from the write path),
//   3. the error against the unquantized f16 reference is reported (QSA_KVQ_ACCURACY).
// DISABLED_ benchmark: ov_gpu_unit_tests --gtest_also_run_disabled_tests --gtest_filter='*sparse_pa_kvq_perf*'

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <intel_gpu/op/sparse_kv_quant.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/sparse_pa.hpp>
#include <intel_gpu/primitives/sparse_sdpa.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "impls/ocl_v2/qsa_stage_events.hpp"
#include "primitive_inst.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

using ov::intel_gpu::op::kv_page_layout;
using ov::intel_gpu::op::KVQuant;
using Mode = KVQuant::Mode;

/// The QSA ops run on the OpenCL implementation, which needs the Intel subgroup extensions.
bool ocl_available() {
    const auto& info = get_test_engine().get_device_info();
    return info.supports_intel_subgroups && info.supports_intel_subgroups_short;
}

int64_t I(size_t v) {
    return static_cast<int64_t>(v);
}

float f16r(float x) {
    return static_cast<float>(ov::float16(x));
}

uint16_t f16_bits(float x) {
    return ov::float16(x).to_bits();
}

float f16_from_bits(uint16_t bits) {
    return static_cast<float>(ov::float16::from_bits(bits));
}

constexpr float kMinScale = 1.0f / 16384.0f;
constexpr float kRangeNone = 3.0e38f;  // "no range yet" bound of the BY_CHANNEL write

/// Headroom of the incremental BY_CHANNEL key write (QSA_KCH_HEADROOM, default off = -1); negative = always re-quantize.
float kch_headroom() {
    const char* env = std::getenv("QSA_KCH_HEADROOM");
    return env ? std::strtof(env, nullptr) : -1.0f;
}

/// True when the BY_CHANNEL range must include 0 (QSA_KCH_ZERO_RANGE, the previous rule).
bool kch_zero_range() {
    return std::getenv("QSA_KCH_ZERO_RANGE") != nullptr;
}

KVQuant token_quant(uint8_t bits, bool sym) {
    return {Mode::BY_TOKEN, bits, sym, 0};
}
KVQuant channel_quant(uint8_t bits, bool sym, uint16_t group) {
    return {Mode::BY_CHANNEL, bits, sym, group};
}

std::string quant_name(const KVQuant& q) {
    if (!q.enabled()) {
        return "f16";
    }
    std::string s = q.mode == Mode::BY_TOKEN ? "tok" : "ch" + std::to_string(q.group_size);
    return s + (q.bits == 8 ? "_i8" : "_u4") + (q.symmetric ? "_sym" : "_asym");
}

// ---------------------------------------------------------------------------
// CPU model of the quantized page format and of the kernels' write stage
// ---------------------------------------------------------------------------

struct quant_pages {
    KVQuant q;
    size_t heads = 0, page = 0, D = 0, pages = 0;
    ov::intel_gpu::op::KVPageLayout lay;
    std::vector<uint8_t> bytes;

    quant_pages() = default;
    quant_pages(const KVQuant& quant, size_t num_pages, size_t kv_heads, size_t page_size, size_t head_size)
        : q(quant),
          heads(kv_heads),
          page(page_size),
          D(head_size),
          pages(num_pages),
          lay(kv_page_layout(quant, page_size, head_size)),
          bytes(num_pages * kv_heads * lay.page_bytes, 0xFF) {}  // NaN-patterned: unread-parameter bugs must show

    size_t base(size_t p, size_t h) const {
        return (p * heads + h) * lay.page_bytes;
    }
    float qmax() const {
        return q.bits == 8 ? 255.0f : 15.0f;
    }
    float zero_code() const {
        return q.bits == 8 ? 128.0f : 8.0f;
    }

    void get_params_token(size_t p, size_t h, size_t tok, float& scale, float& zp) const {
        const size_t off = base(p, h) + lay.param_offset + tok * 4;
        uint16_t s = 0, z = 0;
        std::memcpy(&s, &bytes[off], 2);
        std::memcpy(&z, &bytes[off + 2], 2);
        scale = f16_from_bits(s);
        zp = q.symmetric ? zero_code() : f16_from_bits(z);
    }
    size_t record_offset(size_t p, size_t h, size_t tok) const {
        return base(p, h) + lay.param_offset + (tok / q.group_size) * D * (q.symmetric ? 2 : 4);
    }
    void get_params_channel(size_t p, size_t h, size_t tok, std::vector<float>& scale, std::vector<float>& zp) const {
        const size_t rec = record_offset(p, h, tok);
        scale.resize(D);
        zp.resize(D);
        for (size_t c = 0; c < D; c++) {
            uint16_t s = 0, z = 0;
            std::memcpy(&s, &bytes[rec + c * 2], 2);
            scale[c] = f16_from_bits(s);
            if (q.symmetric) {
                zp[c] = zero_code();
            } else {
                std::memcpy(&z, &bytes[rec + D * 2 + c * 2], 2);
                zp[c] = f16_from_bits(z);
            }
        }
    }

    uint8_t code(size_t p, size_t h, size_t tok, size_t c) const {
        const size_t row = base(p, h) + tok * lay.row_bytes;
        if (q.bits == 8) {
            return bytes[row + c];
        }
        const uint8_t b = bytes[row + c / 2];
        return (c % 2 == 0) ? (b & 0xF) : (b >> 4);
    }
    void set_code(size_t p, size_t h, size_t tok, size_t c, uint8_t v) {
        const size_t row = base(p, h) + tok * lay.row_bytes;
        if (q.bits == 8) {
            bytes[row + c] = v;
        } else {
            uint8_t& b = bytes[row + c / 2];
            b = (c % 2 == 0) ? static_cast<uint8_t>((b & 0xF0) | v) : static_cast<uint8_t>((b & 0x0F) | (v << 4));
        }
    }

    /// x = (q - zp) * scale in the kernels' f16 arithmetic.
    static float dequant_one(uint8_t code, float zp, float scale) {
        return f16r(f16r(static_cast<float>(code) - zp) * scale);
    }

    std::vector<float> read_row(size_t p, size_t h, size_t tok) const {
        std::vector<float> out(D);
        if (q.mode == Mode::BY_TOKEN) {
            float s = 0, z = 0;
            get_params_token(p, h, tok, s, z);
            for (size_t c = 0; c < D; c++) {
                out[c] = dequant_one(code(p, h, tok, c), z, s);
            }
        } else {
            std::vector<float> s, z;
            get_params_channel(p, h, tok, s, z);
            for (size_t c = 0; c < D; c++) {
                out[c] = dequant_one(code(p, h, tok, c), z[c], s[c]);
            }
        }
        return out;
    }

    uint8_t quantize(float x, float inv, float zp) const {
        float y = x * inv + zp;
        y = std::min(std::max(y, 0.0f), qmax());
        return static_cast<uint8_t>(y + 0.5f);
    }

    /// BY_TOKEN: quantizes one f16 row.
    void write_token(size_t p, size_t h, size_t tok, const float* row) {
        float mn = row[0], mx = row[0];
        for (size_t c = 0; c < D; c++) {
            mn = std::min(mn, row[c]);
            mx = std::max(mx, row[c]);
        }
        float scale = 1.0f, zp = 0.0f;
        if (q.symmetric) {
            scale = std::max(mx, -mn) / (q.bits == 8 ? 127.0f : 7.0f);
            zp = zero_code();
        } else {
            mn = std::min(mn, 0.0f);
            mx = std::max(mx, 0.0f);
            scale = (mx - mn) / qmax();
        }
        scale = std::max(scale, kMinScale);
        if (!q.symmetric) {
            zp = -mn / scale;
        }
        const uint16_t scale_bits = f16_bits(scale), zp_bits = f16_bits(zp);
        const float inv = 1.0f / f16_from_bits(scale_bits);
        const float zpf = f16_from_bits(zp_bits);
        for (size_t c = 0; c < D; c++) {
            set_code(p, h, tok, c, quantize(row[c], inv, zpf));
        }
        const size_t off = base(p, h) + lay.param_offset + tok * 4;
        std::memcpy(&bytes[off], &scale_bits, 2);
        std::memcpy(&bytes[off + 2], &zp_bits, 2);
    }

    /// Symmetric ranges (and the previous asymmetric rule, QSA_KCH_ZERO_RANGE) include 0.
    bool zero_range() const {
        return q.symmetric || kch_zero_range();
    }

    /// Smallest asymmetric BY_CHANNEL range relative to the larger magnitude (|zero point| <= 512).
    float min_width() const {
        return qmax() / 512.0f;
    }

    /// Value range a BY_CHANNEL channel with (scale, zp) can represent.
    void channel_range(float scale, float zp, float& lo, float& hi) const {
        if (q.symmetric) {
            hi = (q.bits == 8 ? 127.0f : 7.0f) * scale;
            lo = -hi;
        } else {
            lo = -zp * scale;
            hi = (qmax() - zp) * scale;
        }
    }

    /// BY_CHANNEL: re-quantizes the group starting at page-local token `tok0` from `rows` (one f16 row per
    /// valid token of the group, already dequantized where they came from the cache). `range` (per channel
    /// {min, max}, including 0) replaces the range of the rows.
    void write_group(size_t p,
                     size_t h,
                     size_t tok0,
                     const std::vector<std::vector<float>>& rows,
                     const std::vector<std::pair<float, float>>* range = nullptr) {
        std::vector<float> scale_h(D), zp_h(D), inv(D);
        std::vector<uint16_t> scale_bits(D), zp_bits(D);
        for (size_t c = 0; c < D; c++) {
            // Symmetric ranges include 0; asymmetric ones are the data range, widened to at least min_width() of the larger
            // magnitude so that the zero point stays within about +-512.
            float mn = zero_range() ? 0.0f : kRangeNone, mx = zero_range() ? 0.0f : -kRangeNone;
            if (range) {
                mn = (*range)[c].first;
                mx = (*range)[c].second;
            } else {
                for (const auto& r : rows) {
                    mn = std::min(mn, r[c]);
                    mx = std::max(mx, r[c]);
                }
            }
            float scale = 1.0f, zp = 0.0f;
            if (q.symmetric) {
                scale = std::max(mx, -mn) / (q.bits == 8 ? 127.0f : 7.0f);
                zp = zero_code();
            } else {
                if (!zero_range()) {
                    const float widen = std::max(min_width() * std::max(std::fabs(mn), std::fabs(mx)) - (mx - mn), 0.0f);
                    mx += widen;
                }
                scale = (mx - mn) / qmax();
            }
            scale = std::max(scale, kMinScale);
            if (!q.symmetric) {
                zp = -mn / scale;
            }
            scale_bits[c] = f16_bits(scale);
            zp_bits[c] = f16_bits(zp);
            scale_h[c] = f16_from_bits(scale_bits[c]);
            zp_h[c] = f16_from_bits(zp_bits[c]);
            inv[c] = 1.0f / scale_h[c];
        }
        for (size_t t = 0; t < rows.size(); t++) {
            for (size_t c = 0; c < D; c++) {
                set_code(p, h, tok0 + t, c, quantize(rows[t][c], inv[c], zp_h[c]));
            }
        }
        const size_t rec = record_offset(p, h, tok0);
        std::memcpy(&bytes[rec], scale_bits.data(), D * 2);
        if (!q.symmetric) {
            std::memcpy(&bytes[rec + D * 2], zp_bits.data(), D * 2);
        }
    }
};

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

struct dims {
    size_t Hq = 24, Hkv = 2, Dh = 256, Dv = 256, r = 4, page = 16, topk = 512;
};

dims qwen38() {
    return {};
}
dims toy() {
    dims d;
    d.Hq = 4;
    d.Hkv = 2;
    d.Dh = 64;
    d.Dv = 64;
    d.topk = 6;
    return d;
}

std::vector<float> make_values(size_t count, uint32_t seed, float scale, size_t channels) {
    std::vector<float> out(count);
    uint32_t s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < count; i++) {
        s = s * 1664525u + 1013904223u;
        float x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f);
        // Channels with different magnitudes plus a few outliers: stresses both per-token and per-channel scales.
        const size_t c = i % channels;
        x *= scale * (1.0f + static_cast<float>(c % 7) * 0.5f);
        if ((s >> 3) % 211 == 0) {
            x *= 4.0f;
        }
        out[i] = f16r(x);
    }
    return out;
}

struct seq_spec {
    size_t past, q_len;
};

struct case_data {
    dims d;
    KVQuant kq, vq;
    std::vector<seq_spec> seqs;
    size_t tokens = 0;
    std::vector<size_t> page_count;  // per sequence
    std::vector<size_t> page_first;  // first logical page of each sequence in the table
    std::vector<int32_t> block_indices, block_indices_begins, subsequence_begins, past_lens;
    std::vector<float> query;                              // [tokens, Hq*Dh]
    std::vector<std::vector<float>> key_hist, value_hist;  // per sequence [total, Hkv, D]
    std::vector<int32_t> sel, count;                       // [tokens, topk], [tokens]
    size_t total_pages = 0;
};

case_data make_case(const dims& d, const KVQuant& kq, const KVQuant& vq, const std::vector<seq_spec>& seqs, uint32_t seed) {
    case_data c;
    c.d = d;
    c.kq = kq;
    c.vq = vq;
    c.seqs = seqs;
    size_t page_total = 0;
    c.block_indices_begins = {0};
    c.subsequence_begins = {0};
    for (const auto& s : seqs) {
        const size_t total = s.past + s.q_len;
        const size_t pages = (total + d.page - 1) / d.page;
        c.page_first.push_back(page_total);
        c.page_count.push_back(pages);
        page_total += pages;
        c.block_indices_begins.push_back(static_cast<int32_t>(page_total));
        c.subsequence_begins.push_back(static_cast<int32_t>(c.tokens + s.q_len));
        c.past_lens.push_back(static_cast<int32_t>(s.past));
        c.tokens += s.q_len;
    }
    c.total_pages = page_total;
    // A deterministic page permutation: logical order != physical order.
    c.block_indices.resize(page_total);
    for (size_t i = 0; i < page_total; i++) {
        c.block_indices[i] = static_cast<int32_t>((i * 7 + 3) % page_total);
    }
    if (page_total % 7 == 0) {  // keep it a permutation
        std::iota(c.block_indices.begin(), c.block_indices.end(), 0);
    }
    c.query = make_values(c.tokens * d.Hq * d.Dh, seed + 1, 1.0f, d.Dh);
    for (size_t s = 0; s < seqs.size(); s++) {
        const size_t total = seqs[s].past + seqs[s].q_len;
        c.key_hist.push_back(make_values(total * d.Hkv * d.Dh, seed + 10 + static_cast<uint32_t>(s), 1.0f, d.Dh));
        c.value_hist.push_back(make_values(total * d.Hkv * d.Dv, seed + 20 + static_cast<uint32_t>(s), 1.0f, d.Dv));
    }
    c.sel.assign(c.tokens * d.topk, 0);
    c.count.assign(c.tokens, 0);
    size_t t = 0;
    for (const auto& s : seqs) {
        for (size_t i = 0; i < s.q_len; i++, t++) {
            const size_t n_complete = (s.past + i + 1) / d.r;
            const size_t cnt = std::min(d.topk, n_complete);
            for (size_t j = 0; j < cnt; j++) {
                c.sel[t * d.topk + j] = static_cast<int32_t>(j * n_complete / cnt);
            }
            c.count[t] = static_cast<int32_t>(cnt);
        }
    }
    return c;
}

/// Quantizes tokens [from, to) of a sequence into `qp`, in the order the GPU kernels see them (token modes row by
/// row; BY_CHANNEL group by group, re-quantizing a group that already holds cached tokens).
size_t g_kch_fit = 0, g_kch_widened = 0;  // partial-group writes that kept / widened the stored range

void cpu_write_range(quant_pages& qp, const case_data& c, size_t seq, const std::vector<float>& hist, size_t head_size, size_t from, size_t to) {
    const auto& d = c.d;
    const size_t phys_base = c.page_first[seq];
    auto phys = [&](size_t tok) {
        return static_cast<size_t>(c.block_indices[phys_base + tok / d.page]);
    };
    for (size_t h = 0; h < d.Hkv; h++) {
        if (qp.q.mode == Mode::BY_TOKEN) {
            for (size_t t = from; t < to; t++) {
                qp.write_token(phys(t), h, t % d.page, &hist[(t * d.Hkv + h) * head_size]);
            }
            continue;
        }
        const size_t G = qp.q.group_size;
        const float headroom = kch_headroom();
        for (size_t g = from / G; g * G < to; g++) {
            const size_t g0 = g * G;
            const size_t n_valid = std::min(G, to - g0);
            const size_t n_old = from > g0 ? std::min(from - g0, n_valid) : 0;
            std::vector<std::vector<float>> rows;
            for (size_t t = 0; t < n_valid; t++) {
                if (t < n_old) {
                    rows.push_back(qp.read_row(phys(g0), h, (g0 % d.page) + t));
                } else {
                    rows.emplace_back(hist.begin() + ((g0 + t) * d.Hkv + h) * head_size, hist.begin() + ((g0 + t) * d.Hkv + h + 1) * head_size);
                }
            }
            if (headroom < 0.0f || n_valid == G) {
                qp.write_group(phys(g0), h, g0 % d.page, rows);
                continue;
            }
            // A group that stays partial: keep the stored parameters when the new rows fit, else widen the range.
            std::vector<float> old_s, old_z;
            if (n_old > 0) {
                qp.get_params_channel(phys(g0), h, g0 % d.page, old_s, old_z);
            }
            bool fit = n_old > 0;
            const float none = qp.zero_range() ? 0.0f : kRangeNone;
            std::vector<std::pair<float, float>> range(qp.D, {none, -none});
            for (size_t t = n_old; t < n_valid; t++) {
                for (size_t c = 0; c < qp.D; c++) {
                    range[c].first = std::min(range[c].first, rows[t][c]);
                    range[c].second = std::max(range[c].second, rows[t][c]);
                    if (n_old > 0) {
                        const float v = rows[t][c] * (1.0f / old_s[c]) + old_z[c];
                        fit = fit && v >= -0.5f && v < qp.qmax() + 0.5f;
                    }
                }
            }
            if (n_old > 0) {
                (fit ? g_kch_fit : g_kch_widened)++;
            }
            if (fit) {
                for (size_t t = n_old; t < n_valid; t++) {
                    for (size_t c = 0; c < qp.D; c++) {
                        qp.set_code(phys(g0), h, (g0 % d.page) + t, c, qp.quantize(rows[t][c], 1.0f / old_s[c], old_z[c]));
                    }
                }
                continue;
            }
            std::vector<float> prior_s, prior_z;
            if (n_old > 0) {
                prior_s = old_s;
                prior_z = old_z;
            } else if (g0 >= G && from == g0) {
                const size_t pg0 = g0 - G;
                qp.get_params_channel(phys(pg0), h, pg0 % d.page, prior_s, prior_z);
            }
            for (size_t c = 0; c < qp.D; c++) {
                float lo = none, hi = -none;
                if (!prior_s.empty()) {
                    qp.channel_range(prior_s[c], prior_z[c], lo, hi);
                }
                if (qp.zero_range()) {
                    range[c].first = std::min(lo, range[c].first * (1.0f + headroom));
                    range[c].second = std::max(hi, range[c].second * (1.0f + headroom));
                } else {
                    const float w =
                        std::max(range[c].second - range[c].first, qp.min_width() * std::max(std::fabs(range[c].first), std::fabs(range[c].second)));
                    range[c].first = std::min(lo, range[c].first - headroom * w);
                    range[c].second = std::max(hi, range[c].second + headroom * w);
                }
            }
            qp.write_group(phys(g0), h, g0 % d.page, rows, &range);
        }
    }
}

template <typename T>
memory::ptr make_device_mem(engine& eng, const ov::PartialShape& shape, data_types dt, const std::vector<T>& data) {
    auto mem = eng.allocate_memory(layout{shape, dt, format::bfyx}, allocation_type::usm_device);
    if (!data.empty()) {
        mem->copy_from(get_test_stream(), data.data(), true);
    }
    return mem;
}

memory::ptr make_f16_mem(engine& eng, const ov::PartialShape& shape, const std::vector<float>& data) {
    std::vector<ov::float16> h(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        h[i] = ov::float16(data[i]);
    }
    return make_device_mem(eng, shape, data_types::f16, h);
}

template <typename T>
std::vector<T> read_mem(const memory::ptr& mem) {
    mem_lock<T, mem_lock_type::read> lock(mem, get_test_stream());
    return {lock.begin(), lock.end()};
}

struct gpu_run {
    std::vector<float> out;
    std::vector<uint8_t> key_bytes, value_bytes;
    std::vector<float> key_f16, value_f16;  // for f16 caches
};

struct run_options {
    bool profile = false;
    int warmup = 0;
    int iterations = 1;
    double gpu_ms_min = 0.0, gpu_ms_avg = 0.0;
    size_t kernels = 0;
    std::vector<double> stage_ms;  // per dispatched stage of the best iteration
};

ov::PartialShape cache_shape(const case_data& c, const KVQuant& q, size_t head_size) {
    const auto lay = kv_page_layout(q, c.d.page, head_size);
    return q.enabled() ? ov::PartialShape{I(c.total_pages), I(c.d.Hkv), I(lay.total_rows), I(lay.row_bytes)}
                       : ov::PartialShape{I(c.total_pages), I(c.d.Hkv), I(c.d.page), I(head_size)};
}

/// Runs the sparse_pa primitive. Caches start from `key_init` / `value_init` (raw bytes / f16 values).
gpu_run run_gpu(const case_data& c, const std::vector<uint8_t>& key_init, const std::vector<uint8_t>& value_init, run_options* opt = nullptr) {
    auto& engine = get_test_engine();
    const auto& d = c.d;
    const size_t T = c.tokens;

    // New tokens (the kernel writes them): token-major [T, Hkv*D].
    std::vector<float> key_new(T * d.Hkv * d.Dh), value_new(T * d.Hkv * d.Dv);
    size_t t0 = 0;
    for (size_t s = 0; s < c.seqs.size(); s++) {
        const auto& sp = c.seqs[s];
        std::copy(c.key_hist[s].begin() + sp.past * d.Hkv * d.Dh, c.key_hist[s].end(), key_new.begin() + t0 * d.Hkv * d.Dh);
        std::copy(c.value_hist[s].begin() + sp.past * d.Hkv * d.Dv, c.value_hist[s].end(), value_new.begin() + t0 * d.Hkv * d.Dv);
        t0 += sp.q_len;
    }

    cldnn::sparse_pa::Config cfg;
    cfg.num_heads = d.Hq;
    cfg.num_kv_heads = d.Hkv;
    cfg.num_selection_heads = 1;
    cfg.k_head_size = d.Dh;
    cfg.v_head_size = d.Dv;
    cfg.key_block_size = d.r;
    cfg.query_block_size = 1;
    cfg.causal = true;
    cfg.page_size = d.page;
    cfg.key_quant = c.kq;
    cfg.value_quant = c.vq;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d.Dh));

    const auto kshape = cache_shape(c, c.kq, d.Dh);
    const auto vshape = cache_shape(c, c.vq, d.Dv);
    const auto kdt = c.kq.enabled() ? data_types::u8 : data_types::f16;
    const auto vdt = c.vq.enabled() ? data_types::u8 : data_types::f16;

    auto key_cache = engine.allocate_memory(layout{kshape, kdt, format::bfyx}, allocation_type::usm_device);
    auto value_cache = engine.allocate_memory(layout{vshape, vdt, format::bfyx}, allocation_type::usm_device);
    key_cache->copy_from(get_test_stream(), key_init.data(), true);
    value_cache->copy_from(get_test_stream(), value_init.data(), true);

    topology topo;
    topo.add(input_layout("query", layout{{I(T), I(d.Hq * d.Dh)}, data_types::f16, format::bfyx}));
    topo.add(input_layout("key", layout{{I(T), I(d.Hkv * d.Dh)}, data_types::f16, format::bfyx}));
    topo.add(input_layout("value", layout{{I(T), I(d.Hkv * d.Dv)}, data_types::f16, format::bfyx}));
    topo.add(input_layout("key_cache", layout{kshape, kdt, format::bfyx}));
    topo.add(input_layout("value_cache", layout{vshape, vdt, format::bfyx}));
    topo.add(data("past_lens", make_device_mem(engine, {I(c.past_lens.size())}, data_types::i32, c.past_lens)));
    topo.add(data("subsequence_begins", make_device_mem(engine, {I(c.subsequence_begins.size())}, data_types::i32, c.subsequence_begins)));
    topo.add(data("block_indices", make_device_mem(engine, {I(c.block_indices.size())}, data_types::i32, c.block_indices)));
    topo.add(data("block_indices_begins", make_device_mem(engine, {I(c.block_indices_begins.size())}, data_types::i32, c.block_indices_begins)));
    topo.add(data("sel", make_device_mem(engine, {I(T), 1, I(d.topk)}, data_types::i32, c.sel)));
    topo.add(data("count", make_device_mem(engine, {I(T), 1}, data_types::i32, c.count)));
    topo.add(sparse_pa("attn",
                       {input_info("query"),
                        input_info("key"),
                        input_info("value"),
                        input_info("key_cache"),
                        input_info("value_cache"),
                        input_info("past_lens"),
                        input_info("subsequence_begins"),
                        input_info("block_indices"),
                        input_info("block_indices_begins"),
                        input_info("sel"),
                        input_info("count")},
                       cfg,
                       scale));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    if (opt && opt->profile) {
        config.set_property(ov::enable_profiling(true));
    }
    network net(engine, topo, config);
    net.set_input_data("query", make_f16_mem(engine, {I(T), I(d.Hq * d.Dh)}, c.query));
    net.set_input_data("key", make_f16_mem(engine, {I(T), I(d.Hkv * d.Dh)}, key_new));
    net.set_input_data("value", make_f16_mem(engine, {I(T), I(d.Hkv * d.Dv)}, value_new));
    net.set_input_data("key_cache", key_cache);
    net.set_input_data("value_cache", value_cache);

    gpu_run run;
    const int warmup = opt ? opt->warmup : 0;
    const int iterations = opt ? opt->iterations : 1;
    std::vector<double> gpu_times;
    for (int it = 0; it < warmup + iterations; it++) {
        const bool measure = opt != nullptr && opt->profile && it >= warmup;
        ov::intel_gpu::ocl::qsa::QsaStageEvents::events().clear();
        ov::intel_gpu::ocl::qsa::QsaStageEvents::armed() = measure;
        auto outputs = net.execute();
        outputs.at("attn").get_memory();
        net.get_stream().finish();
        ov::intel_gpu::ocl::qsa::QsaStageEvents::armed() = false;
        if (measure) {
            double total = 0.0;
            std::vector<double> stages;
            for (const auto& ev : ov::intel_gpu::ocl::qsa::QsaStageEvents::events()) {
                for (const auto& interval : ev->get_profiling_info()) {
                    if (interval.stage == instrumentation::profiling_stage::executing) {
                        const double ms = std::chrono::duration<double, std::milli>(interval.value->value()).count();
                        total += ms;
                        stages.push_back(ms);
                        break;
                    }
                }
            }
            opt->kernels = ov::intel_gpu::ocl::qsa::QsaStageEvents::events().size();
            if (gpu_times.empty() || total < *std::min_element(gpu_times.begin(), gpu_times.end())) {
                opt->stage_ms = stages;
            }
            gpu_times.push_back(total);
        }
        if (it + 1 == warmup + iterations) {
            for (auto& v : read_mem<ov::float16>(outputs.at("attn").get_memory())) {
                run.out.push_back(static_cast<float>(v));
            }
        }
    }
    if (opt && !gpu_times.empty()) {
        opt->gpu_ms_min = *std::min_element(gpu_times.begin(), gpu_times.end());
        opt->gpu_ms_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / static_cast<double>(gpu_times.size());
    }
    if (c.kq.enabled()) {
        run.key_bytes = read_mem<uint8_t>(key_cache);
    }
    if (c.vq.enabled()) {
        run.value_bytes = read_mem<uint8_t>(value_cache);
    }
    return run;
}

/// Attention reference over per-token K/V rows supplied by `k_row(seq, pos, head)` / `v_row`.
template <typename KRow, typename VRow>
std::vector<float> reference_attention(const case_data& c, KRow&& k_row, VRow&& v_row) {
    const auto& d = c.d;
    std::vector<float> out(c.tokens * d.Hq * d.Dv, 0.0f);
    const size_t group = d.Hq / d.Hkv;
    const double scale = 1.0 / std::sqrt(static_cast<double>(d.Dh));
    size_t t = 0;
    for (size_t s = 0; s < c.seqs.size(); s++) {
        for (size_t i = 0; i < c.seqs[s].q_len; i++, t++) {
            const size_t abs_pos = c.seqs[s].past + i;
            std::vector<size_t> tokens;
            for (int32_t j = 0; j < c.count[t]; j++) {
                const size_t b = static_cast<size_t>(c.sel[t * d.topk + j]);
                for (size_t k = 0; k < d.r; k++) {
                    tokens.push_back(b * d.r + k);
                }
            }
            for (size_t p = ((abs_pos + 1) / d.r) * d.r; p <= abs_pos; p++) {
                tokens.push_back(p);
            }
            for (size_t h = 0; h < d.Hq; h++) {
                const size_t kv = h / group;
                std::vector<double> logits(tokens.size());
                for (size_t n = 0; n < tokens.size(); n++) {
                    const auto krow = k_row(s, tokens[n], kv);
                    double dot = 0.0;
                    for (size_t j = 0; j < d.Dh; j++) {
                        dot += static_cast<double>(c.query[(t * d.Hq + h) * d.Dh + j]) * krow[j];
                    }
                    logits[n] = dot * scale;
                }
                const double m = *std::max_element(logits.begin(), logits.end());
                double sum = 0.0;
                for (auto& x : logits) {
                    x = std::exp(x - m);
                    sum += x;
                }
                std::vector<double> acc(d.Dv, 0.0);
                for (size_t n = 0; n < tokens.size(); n++) {
                    const auto vrow = v_row(s, tokens[n], kv);
                    for (size_t j = 0; j < d.Dv; j++) {
                        acc[j] += logits[n] * vrow[j];
                    }
                }
                for (size_t j = 0; j < d.Dv; j++) {
                    out[(t * d.Hq + h) * d.Dv + j] = static_cast<float>(acc[j] / sum);
                }
            }
        }
    }
    return out;
}

struct err_stats {
    double max_abs = 0, mean_abs = 0, max_rel = 0;
    size_t bad = 0;
};

err_stats compare(const std::vector<float>& actual, const std::vector<float>& expected, double atol, double rtol) {
    err_stats e;
    EXPECT_EQ(actual.size(), expected.size());
    double sum = 0.0;
    for (size_t i = 0; i < std::min(actual.size(), expected.size()); i++) {
        const double diff = std::fabs(static_cast<double>(actual[i]) - expected[i]);
        e.max_abs = std::max(e.max_abs, diff);
        e.max_rel = std::max(e.max_rel, diff / (std::fabs(static_cast<double>(expected[i])) + 1e-6));
        sum += diff;
        if (!std::isfinite(actual[i]) || diff > atol + rtol * std::fabs(static_cast<double>(expected[i]))) {
            e.bad++;
        }
    }
    e.mean_abs = sum / static_cast<double>(std::max<size_t>(1, actual.size()));
    return e;
}

struct kvq_case {
    std::string name;
    dims d;
    KVQuant k, v;
    std::vector<seq_spec> seqs;
};

/// Builds the initial caches (history quantized on the CPU), runs the GPU and checks the three properties.
void run_case(const kvq_case& tc) {
    const auto c = make_case(tc.d, tc.k, tc.v, tc.seqs, 1);
    const auto& d = c.d;
    const size_t D[2] = {d.Dh, d.Dv};
    std::vector<quant_pages> hist(2), expected(2);
    const KVQuant quants[2] = {c.kq, c.vq};
    for (int which = 0; which < 2; which++) {
        hist[which] = quant_pages(quants[which], c.total_pages, d.Hkv, d.page, D[which]);
        for (size_t s = 0; s < c.seqs.size(); s++) {
            cpu_write_range(hist[which], c, s, which == 0 ? c.key_hist[s] : c.value_hist[s], D[which], 0, c.seqs[s].past);
        }
        expected[which] = hist[which];
        for (size_t s = 0; s < c.seqs.size(); s++) {
            cpu_write_range(expected[which], c, s, which == 0 ? c.key_hist[s] : c.value_hist[s], D[which], c.seqs[s].past, c.seqs[s].past + c.seqs[s].q_len);
        }
    }

    const auto run = run_gpu(c, hist[0].bytes, hist[1].bytes);
    ASSERT_FALSE(run.out.empty());

    // 1. The cache the GPU wrote vs the predicted one: dequantized difference in quantization steps.
    quant_pages gpu_pages[2] = {hist[0], hist[1]};
    gpu_pages[0].bytes = run.key_bytes;
    gpu_pages[1].bytes = run.value_bytes;
    for (int which = 0; which < 2; which++) {
        double max_steps = 0.0;
        size_t differing = 0, total = 0;
        for (size_t s = 0; s < c.seqs.size(); s++) {
            const size_t tot = c.seqs[s].past + c.seqs[s].q_len;
            for (size_t t = 0; t < tot; t++) {
                const size_t page = static_cast<size_t>(c.block_indices[c.page_first[s] + t / d.page]);
                for (size_t h = 0; h < d.Hkv; h++) {
                    const auto got = gpu_pages[which].read_row(page, h, t % d.page);
                    const auto want = expected[which].read_row(page, h, t % d.page);
                    float scale_s = 0, scale_z = 0;
                    std::vector<float> ch_s, ch_z;
                    if (quants[which].mode == Mode::BY_TOKEN) {
                        expected[which].get_params_token(page, h, t % d.page, scale_s, scale_z);
                    } else {
                        expected[which].get_params_channel(page, h, t % d.page, ch_s, ch_z);
                    }
                    for (size_t j = 0; j < D[which]; j++) {
                        const double step = quants[which].mode == Mode::BY_TOKEN ? scale_s : ch_s[j];
                        const double diff = std::fabs(static_cast<double>(got[j]) - want[j]);
                        max_steps = std::max(max_steps, diff / step);
                        differing += diff > 0.5 * step ? 1 : 0;
                        total++;
                    }
                }
            }
        }
        std::cout << "QSA_KVQ_CACHE " << tc.name << (which == 0 ? " key" : " value") << " max_diff_steps=" << max_steps << " differing=" << std::setprecision(4)
                  << 100.0 * static_cast<double>(differing) / static_cast<double>(total) << "%" << std::endl;
        EXPECT_LE(max_steps, 2.5) << tc.name << (which == 0 ? " key" : " value") << " cache differs from the prediction";
        EXPECT_LE(static_cast<double>(differing) / static_cast<double>(total), 0.10) << tc.name << (which == 0 ? " key" : " value");
    }

    // 2. Attention over the GPU cache read back through the CPU dequantizer.
    auto cache_row = [&](int which, size_t s, size_t pos, size_t kv) {
        const size_t page = static_cast<size_t>(c.block_indices[c.page_first[s] + pos / d.page]);
        return gpu_pages[which].read_row(page, kv, pos % d.page);
    };
    const auto ref_cache = reference_attention(
        c,
        [&](size_t s, size_t pos, size_t kv) {
            return cache_row(0, s, pos, kv);
        },
        [&](size_t s, size_t pos, size_t kv) {
            return cache_row(1, s, pos, kv);
        });
    const auto e_attn = compare(run.out, ref_cache, 1.5e-2, 1.5e-2);
    std::cout << "QSA_KVQ_ATTN " << tc.name << " max_abs=" << e_attn.max_abs << " mean_abs=" << e_attn.mean_abs << " bad=" << e_attn.bad << std::endl;
    EXPECT_EQ(e_attn.bad, 0u) << tc.name << ": attention output differs from the dequantized-cache reference";

    // 3. Accuracy against the unquantized f16 reference.
    const auto ref_f16 = reference_attention(
        c,
        [&](size_t s, size_t pos, size_t kv) {
            return std::vector<float>(c.key_hist[s].begin() + (pos * d.Hkv + kv) * d.Dh, c.key_hist[s].begin() + (pos * d.Hkv + kv + 1) * d.Dh);
        },
        [&](size_t s, size_t pos, size_t kv) {
            return std::vector<float>(c.value_hist[s].begin() + (pos * d.Hkv + kv) * d.Dv, c.value_hist[s].begin() + (pos * d.Hkv + kv + 1) * d.Dv);
        });
    const auto e_q = compare(run.out, ref_f16, 1e9, 1e9);
    double ref_mean = 0.0;
    for (float x : ref_f16) {
        ref_mean += std::fabs(x);
    }
    ref_mean /= static_cast<double>(ref_f16.size());
    std::cout << "QSA_KVQ_ACCURACY " << tc.name << " max_abs=" << e_q.max_abs << " mean_abs=" << e_q.mean_abs << " rel_mean=" << e_q.mean_abs / ref_mean
              << std::endl;
    EXPECT_LT(e_q.mean_abs / ref_mean, tc.k.bits == 4 || tc.v.bits == 4 ? 0.5 : 0.03) << tc.name;
}

std::string case_name(const char* dim_name, const KVQuant& k, const KVQuant& v, const std::vector<seq_spec>& seqs) {
    std::ostringstream os;
    os << dim_name << "_K" << quant_name(k) << "_V" << quant_name(v);
    for (const auto& s : seqs) {
        os << "_" << s.past << "p" << s.q_len;
    }
    return os.str();
}

class sparse_pa_kvq : public ::testing::TestWithParam<kvq_case> {};

TEST_P(sparse_pa_kvq, matches_reference) {
    if (!ocl_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    run_case(GetParam());
}

std::vector<kvq_case> all_cases() {
    std::vector<kvq_case> cases;
    auto add = [&](const char* dn, const dims& d, const KVQuant& k, const KVQuant& v, std::vector<seq_spec> seqs) {
        cases.push_back({case_name(dn, k, v, seqs), d, k, v, std::move(seqs)});
    };
    const std::vector<std::vector<seq_spec>> scenarios = {{{0, 37}}, {{21, 1}}, {{123, 45}}, {{2051, 1}}, {{40, 1}, {0, 70}, {17, 3}}};
    for (const auto& seqs : scenarios) {
        // Token-mode keys, both bit widths and both symmetries.
        add("qwen", qwen38(), token_quant(8, false), token_quant(8, false), seqs);
        add("qwen", qwen38(), token_quant(4, true), token_quant(4, true), seqs);
        // BY_CHANNEL keys.
        add("qwen", qwen38(), channel_quant(8, false, 16), token_quant(8, false), seqs);
        add("qwen", qwen38(), channel_quant(4, false, 16), token_quant(4, false), seqs);
        add("qwen", qwen38(), channel_quant(8, true, 16), token_quant(8, true), seqs);
    }
    // Page 32 with 32-token groups, and the scalar kernels (head size 64 -> no GQA-DPAS for quantized caches).
    dims wide_page = qwen38();
    wide_page.page = 32;
    for (const auto& seqs : {std::vector<seq_spec>{{0, 100}}, std::vector<seq_spec>{{33, 1}}, std::vector<seq_spec>{{70, 40}}}) {
        add("qwenpage32", wide_page, channel_quant(8, false, 32), token_quant(8, false), seqs);
        add("qwenpage32", wide_page, channel_quant(4, true, 32), token_quant(4, true), seqs);
    }
    // Head size 128: GQA-DPAS workers load 32-channel slices (16-byte 4-bit rows per message).
    dims d128 = qwen38();
    d128.Dh = d128.Dv = 128;
    for (const auto& seqs :
         {std::vector<seq_spec>{{0, 37}}, std::vector<seq_spec>{{21, 1}}, std::vector<seq_spec>{{123, 45}}, std::vector<seq_spec>{{40, 1}, {0, 70}, {17, 3}}}) {
        add("d128", d128, token_quant(8, false), token_quant(8, false), seqs);
        add("d128", d128, token_quant(4, true), token_quant(4, true), seqs);
        add("d128", d128, channel_quant(8, false, 16), token_quant(8, false), seqs);
        add("d128", d128, channel_quant(4, false, 16), token_quant(4, false), seqs);
    }
    for (const auto& seqs : scenarios) {
        add("toy", toy(), token_quant(8, false), token_quant(4, false), seqs);
        add("toy", toy(), channel_quant(4, false, 16), token_quant(8, true), seqs);
        add("toy", toy(), channel_quant(8, false, 16), token_quant(8, false), seqs);
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(smoke, sparse_pa_kvq, ::testing::ValuesIn(all_cases()), [](const ::testing::TestParamInfo<kvq_case>& info) {
    return info.param.name;
});

// Attention-only: a cache that mixes an f16 key cache with a quantized value cache (and vice versa).
TEST(sparse_pa_kvq_mixed, f16_key_quantized_value) {
    if (!ocl_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    const auto c = make_case(qwen38(), KVQuant{}, token_quant(8, false), {{123, 45}}, 7);
    const auto& d = c.d;
    quant_pages vhist(c.vq, c.total_pages, d.Hkv, d.page, d.Dv);
    cpu_write_range(vhist, c, 0, c.value_hist[0], d.Dv, 0, 123);
    std::vector<ov::float16> khist(c.total_pages * d.Hkv * d.page * d.Dh, ov::float16(0.0f));
    for (size_t t = 0; t < 123; t++) {
        const size_t page = static_cast<size_t>(c.block_indices[c.page_first[0] + t / d.page]);
        for (size_t h = 0; h < d.Hkv; h++) {
            for (size_t j = 0; j < d.Dh; j++) {
                khist[((page * d.Hkv + h) * d.page + t % d.page) * d.Dh + j] = ov::float16(c.key_hist[0][(t * d.Hkv + h) * d.Dh + j]);
            }
        }
    }
    std::vector<uint8_t> kbytes(khist.size() * 2);
    std::memcpy(kbytes.data(), khist.data(), kbytes.size());
    const auto run = run_gpu(c, kbytes, vhist.bytes);
    quant_pages vpages = vhist;
    vpages.bytes = run.value_bytes;
    const auto ref = reference_attention(
        c,
        [&](size_t s, size_t pos, size_t kv) {
            return std::vector<float>(c.key_hist[s].begin() + (pos * d.Hkv + kv) * d.Dh, c.key_hist[s].begin() + (pos * d.Hkv + kv + 1) * d.Dh);
        },
        [&](size_t s, size_t pos, size_t kv) {
            const size_t page = static_cast<size_t>(c.block_indices[c.page_first[s] + pos / d.page]);
            return vpages.read_row(page, kv, pos % d.page);
        });
    const auto e = compare(run.out, ref, 1.5e-2, 1.5e-2);
    std::cout << "QSA_KVQ_ATTN mixed_f16K_i8V max_abs=" << e.max_abs << " bad=" << e.bad << std::endl;
    EXPECT_EQ(e.bad, 0u);
}

// ---------------------------------------------------------------------------
// Multi-step decode with BY_CHANNEL keys: every step appends one token per sequence to the cache the previous step
// produced. Covers the incremental key write (parameters kept / widened / group completed / first token of a group /
// page boundaries) and checks each step against the CPU model of the write.
// ---------------------------------------------------------------------------

enum class decode_pattern { iid, growing, decaying, spikes, calm, stable, offset };

const char* pattern_name(decode_pattern p) {
    switch (p) {
    case decode_pattern::iid:
        return "iid";
    case decode_pattern::growing:
        return "growing";
    case decode_pattern::decaying:
        return "decaying";
    case decode_pattern::spikes:
        return "spikes";
    case decode_pattern::calm:
        return "calm";
    case decode_pattern::stable:
        return "stable";
    case decode_pattern::offset:
        return "offset";
    }
    return "";
}

float pattern_factor(decode_pattern p, size_t step, size_t channel) {
    switch (p) {
    case decode_pattern::growing:
        return 1.0f + 0.15f * static_cast<float>(step);
    case decode_pattern::decaying:
        return std::max(0.05f, 1.0f - 0.03f * static_cast<float>(step));
    case decode_pattern::spikes:
        return (step % 5 == 4 && channel % 13 == step % 13) ? 8.0f : 1.0f;
    case decode_pattern::calm:
        return 0.5f;
    default:
        return 1.0f;
    }
}

struct decode_case {
    std::string name;
    dims d;
    KVQuant k;
    std::vector<size_t> pasts;
    size_t steps;
    decode_pattern pattern;
};

/// `base` with every sequence at `pasts[s]` cached tokens and one new token.
case_data decode_step_case(const case_data& base, const std::vector<size_t>& pasts) {
    case_data c = base;
    const auto& d = c.d;
    c.seqs.clear();
    c.past_lens.clear();
    c.subsequence_begins = {0};
    c.tokens = pasts.size();
    for (size_t s = 0; s < pasts.size(); s++) {
        c.seqs.push_back({pasts[s], 1});
        c.past_lens.push_back(static_cast<int32_t>(pasts[s]));
        c.subsequence_begins.push_back(static_cast<int32_t>(s + 1));
        c.key_hist[s].resize((pasts[s] + 1) * d.Hkv * d.Dh);
        c.value_hist[s].resize((pasts[s] + 1) * d.Hkv * d.Dv);
    }
    c.query.resize(c.tokens * d.Hq * d.Dh);
    c.sel.assign(c.tokens * d.topk, 0);
    c.count.assign(c.tokens, 0);
    for (size_t s = 0; s < pasts.size(); s++) {
        const size_t n_complete = (pasts[s] + 1) / d.r;
        const size_t cnt = std::min(d.topk, n_complete);
        for (size_t j = 0; j < cnt; j++) {
            c.sel[s * d.topk + j] = static_cast<int32_t>(j * n_complete / cnt);
        }
        c.count[s] = static_cast<int32_t>(cnt);
    }
    return c;
}

void run_decode_case(const decode_case& tc) {
    const KVQuant vq = token_quant(8, false);
    std::vector<seq_spec> final_seqs;
    for (size_t p : tc.pasts) {
        final_seqs.push_back({p, tc.steps});
    }
    auto base = make_case(tc.d, tc.k, vq, final_seqs, 5);
    const auto& d = base.d;
    if (tc.pattern == decode_pattern::offset) {
        // Every fourth channel carries a DC offset of about 10 sigma (the values of a channel are uniform in +-(1 + 0.5 * (c % 7))).
        for (size_t s = 0; s < tc.pasts.size(); s++) {
            for (size_t i = 0; i < base.key_hist[s].size(); i++) {
                const size_t c = i % d.Dh;
                if (c % 4 == 0) {
                    base.key_hist[s][i] = f16r(base.key_hist[s][i] + ((c / 4) % 2 ? -6.0f : 6.0f) * (1.0f + 0.5f * static_cast<float>(c % 7)));
                }
            }
        }
    }
    for (size_t s = 0; s < tc.pasts.size(); s++) {
        for (size_t j = 0; j < tc.steps; j++) {
            for (size_t i = 0; i < d.Hkv * d.Dh; i++) {
                float& x = base.key_hist[s][(tc.pasts[s] + j) * d.Hkv * d.Dh + i];
                x = f16r(x * pattern_factor(tc.pattern, j, i % d.Dh));
                if (tc.pattern == decode_pattern::stable && tc.pasts[s] + j >= 16) {
                    // Stationary keys: a row of the previous group, so the stored range keeps fitting.
                    x = base.key_hist[s][(tc.pasts[s] + j - 16) * d.Hkv * d.Dh + i];
                }
            }
        }
    }

    quant_pages key_model(tc.k, base.total_pages, d.Hkv, d.page, d.Dh);
    quant_pages value_model(vq, base.total_pages, d.Hkv, d.page, d.Dv);
    for (size_t s = 0; s < tc.pasts.size(); s++) {
        cpu_write_range(key_model, base, s, base.key_hist[s], d.Dh, 0, tc.pasts[s]);
        cpu_write_range(value_model, base, s, base.value_hist[s], d.Dv, 0, tc.pasts[s]);
    }
    std::vector<uint8_t> key_bytes = key_model.bytes, value_bytes = value_model.bytes;

    double worst_steps = 0.0, worst_differing = 0.0;
    g_kch_fit = g_kch_widened = 0;
    gpu_run run;
    case_data last;
    for (size_t step = 0; step < tc.steps; step++) {
        std::vector<size_t> pasts;
        for (size_t p : tc.pasts) {
            pasts.push_back(p + step);
        }
        last = decode_step_case(base, pasts);
        // Single-step check: the model starts from the cache the GPU produced, so knife-edge rounding does not accumulate.
        key_model.bytes = key_bytes;
        run = run_gpu(last, key_bytes, value_bytes);
        ASSERT_FALSE(run.out.empty());
        key_bytes = run.key_bytes;
        value_bytes = run.value_bytes;
        for (size_t s = 0; s < pasts.size(); s++) {
            cpu_write_range(key_model, base, s, base.key_hist[s], d.Dh, pasts[s], pasts[s] + 1);
        }

        // The cache the GPU produced vs the model, in quantization steps (only the sequences' valid tokens).
        quant_pages gpu_keys = key_model;
        gpu_keys.bytes = run.key_bytes;
        size_t differing = 0, total = 0;
        double step_max = 0.0;
        for (size_t s = 0; s < pasts.size(); s++) {
            for (size_t t = 0; t <= pasts[s]; t++) {
                const size_t page = static_cast<size_t>(base.block_indices[base.page_first[s] + t / d.page]);
                for (size_t h = 0; h < d.Hkv; h++) {
                    const auto got = gpu_keys.read_row(page, h, t % d.page);
                    const auto want = key_model.read_row(page, h, t % d.page);
                    std::vector<float> sc, zp;
                    key_model.get_params_channel(page, h, t % d.page, sc, zp);
                    for (size_t j = 0; j < d.Dh; j++) {
                        const double diff = std::fabs(static_cast<double>(got[j]) - want[j]);
                        step_max = std::max(step_max, diff / sc[j]);
                        differing += diff > 0.5 * sc[j] ? 1 : 0;
                        total++;
                    }
                }
            }
        }
        const double frac = static_cast<double>(differing) / static_cast<double>(total);
        worst_steps = std::max(worst_steps, step_max);
        worst_differing = std::max(worst_differing, frac);
        ASSERT_LE(step_max, 2.0) << tc.name << " step " << step << ": key cache differs from the model";
        ASSERT_LE(frac, 0.10) << tc.name << " step " << step;
    }

    // Final state: attention over the GPU cache, and the key error against the unquantized history.
    quant_pages gpu_keys = key_model, gpu_values = value_model;
    gpu_keys.bytes = key_bytes;
    gpu_values.bytes = value_bytes;
    const auto ref_cache = reference_attention(
        last,
        [&](size_t s, size_t pos, size_t kv) {
            return gpu_keys.read_row(static_cast<size_t>(last.block_indices[last.page_first[s] + pos / d.page]), kv, pos % d.page);
        },
        [&](size_t s, size_t pos, size_t kv) {
            return gpu_values.read_row(static_cast<size_t>(last.block_indices[last.page_first[s] + pos / d.page]), kv, pos % d.page);
        });
    const auto e_attn = compare(run.out, ref_cache, 1.5e-2, 1.5e-2);
    EXPECT_EQ(e_attn.bad, 0u) << tc.name << ": attention output differs from the dequantized-cache reference";

    double err = 0.0, mag = 0.0;
    size_t n = 0;
    for (size_t s = 0; s < tc.pasts.size(); s++) {
        for (size_t t = 0; t < tc.pasts[s] + tc.steps; t++) {
            const size_t page = static_cast<size_t>(base.block_indices[base.page_first[s] + t / d.page]);
            for (size_t h = 0; h < d.Hkv; h++) {
                const auto got = gpu_keys.read_row(page, h, t % d.page);
                for (size_t j = 0; j < d.Dh; j++) {
                    const float x = base.key_hist[s][(t * d.Hkv + h) * d.Dh + j];
                    err += std::fabs(got[j] - x);
                    mag += std::fabs(x);
                    n++;
                }
            }
        }
    }
    const double rel = err / mag;
    std::cout << "QSA_KVQ_DECODE " << tc.name << " headroom=" << kch_headroom() << " worst_cache_steps=" << worst_steps
              << " worst_differing=" << worst_differing * 100.0 << "% attn_max_abs=" << e_attn.max_abs << " key_rel_err=" << rel
              << " key_mae=" << err / static_cast<double>(n) << " partial_fit=" << g_kch_fit << "/" << (g_kch_fit + g_kch_widened) << std::endl;
    EXPECT_LT(rel, tc.k.bits == 4 ? 0.25 : 0.03) << tc.name;
}

std::vector<decode_case> decode_cases() {
    std::vector<decode_case> cases;
    auto add = [&](const char* dn, const dims& d, const KVQuant& k, std::vector<size_t> pasts, size_t steps, decode_pattern p) {
        std::ostringstream os;
        os << dn << "_K" << quant_name(k) << "_" << pattern_name(p);
        for (size_t past : pasts) {
            os << "_" << past;
        }
        os << "x" << steps;
        cases.push_back({os.str(), d, k, std::move(pasts), steps, p});
    };
    dims page32 = qwen38();
    page32.page = 32;
    dims d128 = qwen38();
    d128.Dh = d128.Dv = 128;
    // First token of an empty cache, a partial group, group boundaries (15 / 16 / 17), a long history.
    add("qwen", qwen38(), channel_quant(8, false, 16), {0}, 40, decode_pattern::iid);
    add("qwen", qwen38(), channel_quant(4, false, 16), {5}, 40, decode_pattern::iid);
    add("qwen", qwen38(), channel_quant(8, true, 16), {15, 16, 17, 100}, 36, decode_pattern::iid);
    add("qwen", qwen38(), channel_quant(8, false, 16), {2051}, 20, decode_pattern::iid);
    // The range moves: growing magnitudes (always widened), decaying (always fits), rare large spikes.
    add("toy", toy(), channel_quant(4, true, 16), {31, 47}, 40, decode_pattern::growing);
    add("toy", toy(), channel_quant(8, false, 16), {20, 64}, 40, decode_pattern::decaying);
    add("toy", toy(), channel_quant(8, false, 16), {3, 130}, 45, decode_pattern::spikes);
    add("toy", toy(), channel_quant(4, false, 16), {7, 33, 90}, 40, decode_pattern::spikes);
    add("toy", toy(), channel_quant(8, true, 16), {12}, 40, decode_pattern::calm);
    // 32-token groups / pages, 16-token groups in 32-token pages, head size 128.
    add("page32", page32, channel_quant(8, false, 32), {0, 33, 70}, 70, decode_pattern::iid);
    add("page32", page32, channel_quant(4, true, 32), {31, 64}, 40, decode_pattern::spikes);
    add("page32g16", page32, channel_quant(8, false, 16), {9, 30}, 40, decode_pattern::iid);
    add("d128", d128, channel_quant(4, false, 16), {10}, 40, decode_pattern::calm);
    add("d128", d128, channel_quant(8, false, 16), {0, 47}, 40, decode_pattern::growing);
    // A batch covering every fill state of a group (also the kernel benchmark: cliloader -d, QSA_KCH_HEADROOM=-1 for the old write).
    std::vector<size_t> batch;
    for (size_t i = 0; i < 16; i++) {
        batch.push_back(2048 + i);
    }
    add("batch16", qwen38(), channel_quant(8, false, 16), batch, 16, decode_pattern::iid);
    add("batch16", qwen38(), channel_quant(4, true, 16), batch, 16, decode_pattern::spikes);
    add("batch16", qwen38(), channel_quant(8, false, 16), batch, 16, decode_pattern::stable);
    add("qwen", qwen38(), channel_quant(8, false, 16), {2051}, 20, decode_pattern::stable);
    // Channels with a DC offset: the asymmetric range no longer has to include 0.
    add("qwen", qwen38(), channel_quant(4, false, 16), {5}, 40, decode_pattern::offset);
    add("qwen", qwen38(), channel_quant(8, false, 16), {0}, 40, decode_pattern::offset);
    add("page32", page32, channel_quant(4, false, 32), {31, 64}, 40, decode_pattern::offset);
    add("d128", d128, channel_quant(4, false, 16), {2051}, 20, decode_pattern::offset);
    return cases;
}

class sparse_pa_kvq_decode : public ::testing::TestWithParam<decode_case> {};

TEST_P(sparse_pa_kvq_decode, matches_model) {
    if (!ocl_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    run_decode_case(GetParam());
}

INSTANTIATE_TEST_SUITE_P(smoke, sparse_pa_kvq_decode, ::testing::ValuesIn(decode_cases()), [](const ::testing::TestParamInfo<decode_case>& info) {
    return info.param.name;
});

// ---------------------------------------------------------------------------
// Dense SparseSDPA with compressed K/V (KVCacheCompressed format, see intel_gpu/op/sparse_kv_quant.hpp): i8 data
// tensors with planar scale / zero point tensors. The history is quantized on the CPU in the format of the cache.
// ---------------------------------------------------------------------------

struct dense_case {
    dims d;
    std::vector<size_t> key_len;  // valid keys per batch row (left padded inside S = max)
    size_t q_len;
    KVQuant k = token_quant(8, false);
    KVQuant v = token_quant(8, false);
};

/// A quantized dense history [B, Hkv, S, D].
struct dense_quantized {
    std::vector<int8_t> data;      // [B, Hkv, S, row_bytes]
    std::vector<float> scale, zp;  // elements of the parameter tensors (zp unused when symmetric)
    ov::PartialShape param_shape;  // [B, Hkv, S, 1]
    std::vector<float> deq;        // dequantized [B, Hkv, S, D] in the kernels' f16 arithmetic
};

dense_quantized quantize_dense(const KVQuant& q, const std::vector<float>& x, size_t B, size_t H, size_t S, size_t D) {
    using ov::intel_gpu::op::dense_row_bytes;
    using ov::intel_gpu::op::dense_signed_codes;
    dense_quantized out;
    const size_t row_bytes = dense_row_bytes(q, D);
    const float qmax = q.bits == 8 ? 255.0f : 15.0f;
    const float zero_code = q.bits == 8 ? 128.0f : 8.0f;
    out.data.assign(B * H * S * row_bytes, 0);
    out.deq.assign(B * H * S * D, 0.0f);

    // Stores a code: signed 8-bit codes as is, unsigned ones as bytes / packed nibbles (low nibble = even channel).
    auto put = [&](size_t row, size_t c, int code) {
        if (q.bits == 8) {
            out.data[row * row_bytes + c] = static_cast<int8_t>(code);
        } else {
            auto& b = reinterpret_cast<uint8_t&>(out.data[row * row_bytes + c / 2]);
            b = (c % 2 == 0) ? static_cast<uint8_t>((b & 0xF0) | code) : static_cast<uint8_t>((b & 0x0F) | (code << 4));
        }
    };

    {
        out.param_shape = ov::PartialShape{I(B), I(H), I(S), 1};
        out.scale.assign(B * H * S, 1.0f);
        out.zp.assign(B * H * S, 0.0f);
        for (size_t row = 0; row < B * H * S; row++) {
            const float* r = &x[row * D];
            float mn = 0.0f, mx = 0.0f;
            for (size_t c = 0; c < D; c++) {
                mn = std::min(mn, r[c]);
                mx = std::max(mx, r[c]);
            }
            if (dense_signed_codes(q) && !q.symmetric) {
                // 8-bit asymmetric: signed codes with an integer zero point.
                const float scale = f16r(std::max((mx - mn) / 255.0f, kMinScale));
                const int zp = std::min(127, std::max(-128, static_cast<int>(std::lround(-128.0f - mn / scale))));
                out.scale[row] = scale;
                out.zp[row] = static_cast<float>(zp);
                for (size_t c = 0; c < D; c++) {
                    const int code = std::min(127, std::max(-128, static_cast<int>(std::lround(r[c] / scale)) + zp));
                    put(row, c, code);
                    out.deq[row * D + c] = f16r(f16r(static_cast<float>(code - zp)) * scale);
                }
            } else if (dense_signed_codes(q)) {
                // 8-bit symmetric: signed codes, no zero point.
                const float scale = f16r(std::max(std::max(mx, -mn) / 127.0f, kMinScale));
                out.scale[row] = scale;
                for (size_t c = 0; c < D; c++) {
                    const int code = std::min(127, std::max(-127, static_cast<int>(std::lround(r[c] / scale))));
                    put(row, c, code);
                    out.deq[row * D + c] = f16r(f16r(static_cast<float>(code)) * scale);
                }
            } else {
                // 4-bit: unsigned codes, f16 zero point (symmetric: biased by 8).
                float scale, zp;
                if (q.symmetric) {
                    scale = std::max(std::max(mx, -mn) / 7.0f, kMinScale);
                    zp = zero_code;
                } else {
                    scale = std::max((mx - mn) / qmax, kMinScale);
                    zp = -mn / scale;
                }
                scale = f16r(scale);
                zp = f16r(zp);
                out.scale[row] = scale;
                out.zp[row] = zp;
                for (size_t c = 0; c < D; c++) {
                    const int code = static_cast<int>(std::min(std::max(r[c] / scale + zp, 0.0f), qmax) + 0.5f);
                    put(row, c, code);
                    out.deq[row * D + c] = f16r(f16r(static_cast<float>(code) - zp) * scale);
                }
            }
        }
        return out;
    }
}

void run_dense_case(const dense_case& tc, const char* name) {
    const auto& d = tc.d;
    const size_t B = tc.key_len.size();
    const size_t L = tc.q_len;
    const size_t S = *std::max_element(tc.key_len.begin(), tc.key_len.end());

    // Logical keys/values, stored left padded: logical j of row b at index S - kl + j.
    std::vector<float> key(B * d.Hkv * S * d.Dh, 0.0f), value(B * d.Hkv * S * d.Dv, 0.0f);
    {
        const auto kv = make_values(B * d.Hkv * S * d.Dh, 5, 1.0f, d.Dh);
        const auto vv = make_values(B * d.Hkv * S * d.Dv, 6, 1.0f, d.Dv);
        for (size_t b = 0; b < B; b++) {
            for (size_t h = 0; h < d.Hkv; h++) {
                for (size_t j = 0; j < tc.key_len[b]; j++) {
                    const size_t row = (b * d.Hkv + h) * S + (S - tc.key_len[b]) + j;
                    std::copy(kv.begin() + row * d.Dh, kv.begin() + (row + 1) * d.Dh, key.begin() + row * d.Dh);
                    std::copy(vv.begin() + row * d.Dv, vv.begin() + (row + 1) * d.Dv, value.begin() + row * d.Dv);
                }
            }
        }
    }
    const auto kq = quantize_dense(tc.k, key, B, d.Hkv, S, d.Dh);
    const auto vq = quantize_dense(tc.v, value, B, d.Hkv, S, d.Dv);
    const auto query = make_values(B * d.Hq * L * d.Dh, 7, 1.0f, d.Dh);

    std::vector<int32_t> sel(B * L * d.topk, 0), count(B * L, 0), key_length;
    for (size_t b = 0; b < B; b++) {
        key_length.push_back(static_cast<int32_t>(tc.key_len[b]));
        for (size_t l = 0; l < L; l++) {
            const size_t abs_pos = tc.key_len[b] - L + l;
            const size_t n_complete = (abs_pos + 1) / d.r;
            const size_t cnt = std::min(d.topk, n_complete);
            for (size_t j = 0; j < cnt; j++) {
                sel[(b * L + l) * d.topk + j] = static_cast<int32_t>(j * n_complete / cnt);
            }
            count[b * L + l] = static_cast<int32_t>(cnt);
        }
    }

    auto& engine = get_test_engine();
    ov::intel_gpu::op::SparseSDPA::Config cfg;
    cfg.num_heads = d.Hq;
    cfg.num_kv_heads = d.Hkv;
    cfg.num_selection_heads = 1;
    cfg.k_head_size = d.Dh;
    cfg.v_head_size = d.Dv;
    cfg.key_block_size = d.r;
    cfg.query_block_size = 1;
    cfg.causal = true;
    cfg.key_quant = tc.k;
    cfg.value_quant = tc.v;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d.Dh));

    const ov::PartialShape kv_shape_k{I(B), I(d.Hkv), I(S), I(ov::intel_gpu::op::dense_row_bytes(tc.k, d.Dh))};
    const ov::PartialShape kv_shape_v{I(B), I(d.Hkv), I(S), I(ov::intel_gpu::op::dense_row_bytes(tc.v, d.Dv))};
    auto zp_mem = [&](const KVQuant& q, const dense_quantized& dq) {
        if (ov::intel_gpu::op::dense_zp_is_f16(q)) {
            return make_f16_mem(engine, dq.param_shape, dq.zp);
        }
        std::vector<int8_t> zp(dq.zp.size());
        for (size_t i = 0; i < zp.size(); i++) {
            zp[i] = static_cast<int8_t>(dq.zp[i]);
        }
        return make_device_mem(engine, dq.param_shape, data_types::i8, zp);
    };
    topology topo;
    topo.add(input_layout("query", layout{{I(B), I(d.Hq), I(L), I(d.Dh)}, data_types::f16, format::bfyx}));
    topo.add(data("key", make_device_mem(engine, kv_shape_k, data_types::i8, kq.data)));
    topo.add(data("value", make_device_mem(engine, kv_shape_v, data_types::i8, vq.data)));
    topo.add(data("sel", make_device_mem(engine, {I(B), 1, I(L), I(d.topk)}, data_types::i32, sel)));
    topo.add(data("count", make_device_mem(engine, {I(B), 1, I(L)}, data_types::i32, count)));
    topo.add(data("key_length", make_device_mem(engine, {I(B)}, data_types::i32, key_length)));
    topo.add(data("key_scale", make_f16_mem(engine, kq.param_shape, kq.scale)));
    topo.add(data("value_scale", make_f16_mem(engine, vq.param_shape, vq.scale)));
    std::vector<input_info> attn_inputs = {input_info("query"),
                                           input_info("key"),
                                           input_info("value"),
                                           input_info("sel"),
                                           input_info("count"),
                                           input_info("key_length"),
                                           input_info("key_scale"),
                                           input_info("value_scale")};
    if (!tc.k.symmetric) {
        topo.add(data("key_zp", zp_mem(tc.k, kq)));
        topo.add(data("value_zp", zp_mem(tc.v, vq)));
        attn_inputs.push_back(input_info("key_zp"));
        attn_inputs.push_back(input_info("value_zp"));
    }
    topo.add(sparse_sdpa("attn", attn_inputs, cfg, scale));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network net(engine, topo, config);
    net.set_input_data("query", make_f16_mem(engine, {I(B), I(d.Hq), I(L), I(d.Dh)}, query));
    auto outputs = net.execute();
    std::vector<float> out;
    for (auto v : read_mem<ov::float16>(outputs.at("attn").get_memory())) {
        out.push_back(static_cast<float>(v));
    }

    // Reference over the dequantized (and, for the accuracy line, the original) K/V.
    const size_t group = d.Hq / d.Hkv;
    std::vector<float> ref(B * d.Hq * L * d.Dv), ref_f16(B * d.Hq * L * d.Dv);
    for (int which = 0; which < 2; which++) {
        auto& dst = which == 0 ? ref : ref_f16;
        const auto& K = which == 0 ? kq.deq : key;
        const auto& V = which == 0 ? vq.deq : value;
        for (size_t b = 0; b < B; b++) {
            for (size_t l = 0; l < L; l++) {
                const size_t abs_pos = tc.key_len[b] - L + l;
                std::vector<size_t> tokens;
                for (int32_t j = 0; j < count[b * L + l]; j++) {
                    for (size_t k = 0; k < d.r; k++) {
                        tokens.push_back(static_cast<size_t>(sel[(b * L + l) * d.topk + j]) * d.r + k);
                    }
                }
                for (size_t p = ((abs_pos + 1) / d.r) * d.r; p <= abs_pos; p++) {
                    tokens.push_back(p);
                }
                for (size_t h = 0; h < d.Hq; h++) {
                    const size_t kv = h / group;
                    std::vector<double> logits(tokens.size());
                    for (size_t n = 0; n < tokens.size(); n++) {
                        const size_t row = (b * d.Hkv + kv) * S + (S - tc.key_len[b]) + tokens[n];
                        double dot = 0.0;
                        for (size_t j = 0; j < d.Dh; j++) {
                            dot += static_cast<double>(query[((b * d.Hq + h) * L + l) * d.Dh + j]) * K[row * d.Dh + j];
                        }
                        logits[n] = dot * scale;
                    }
                    const double m = *std::max_element(logits.begin(), logits.end());
                    double sum = 0.0;
                    for (auto& x : logits) {
                        x = std::exp(x - m);
                        sum += x;
                    }
                    for (size_t j = 0; j < d.Dv; j++) {
                        double acc = 0.0;
                        for (size_t n = 0; n < tokens.size(); n++) {
                            const size_t row = (b * d.Hkv + kv) * S + (S - tc.key_len[b]) + tokens[n];
                            acc += logits[n] * V[row * d.Dv + j];
                        }
                        dst[((b * d.Hq + h) * L + l) * d.Dv + j] = static_cast<float>(acc / sum);
                    }
                }
            }
        }
    }
    const auto e = compare(out, ref, 1.5e-2, 1.5e-2);
    const auto e_q = compare(out, ref_f16, 1e9, 1e9);
    double ref_mean = 0.0;
    for (float x : ref_f16) {
        ref_mean += std::fabs(x);
    }
    ref_mean /= static_cast<double>(ref_f16.size());
    std::cout << "QSA_KVQ_DENSE " << name << " vs_dequant max_abs=" << e.max_abs << " bad=" << e.bad << " | vs_f16 mean_abs=" << e_q.mean_abs
              << " rel_mean=" << e_q.mean_abs / ref_mean << std::endl;
    EXPECT_EQ(e.bad, 0u) << name;
    EXPECT_LT(e_q.mean_abs / ref_mean, tc.k.bits == 4 ? 0.5 : 0.03) << name;
}

TEST(sparse_sdpa_kvq, qwen_decode_left_padded) {
    if (!ocl_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    run_dense_case({qwen38(), {2051, 1500}, 1}, "qwen_decode_2x");
}

TEST(sparse_sdpa_kvq, qwen_prefill_left_padded) {
    if (!ocl_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    run_dense_case({qwen38(), {300, 211}, 150}, "qwen_prefill_2x");
}

TEST(sparse_sdpa_kvq, toy_scalar_kernel) {
    if (!ocl_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    run_dense_case({toy(), {90, 61}, 7}, "toy_prefill_2x");
    run_dense_case({toy(), {120}, 1}, "toy_decode");
}

// 4-bit and symmetric dense caches, read by the scalar and GQA-DPAS kernels.
struct dense_format_case {
    std::string name;
    dense_case tc;
};

class sparse_sdpa_kvq_formats : public ::testing::TestWithParam<dense_format_case> {};

TEST_P(sparse_sdpa_kvq_formats, matches_dequantized_reference) {
    if (!ocl_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    run_dense_case(GetParam().tc, GetParam().name.c_str());
}

std::vector<dense_format_case> dense_format_cases() {
    struct scenario {
        const char* name;
        dims d;
        std::vector<size_t> key_len;
        size_t q_len;
    };
    dims d128 = qwen38();
    d128.Dh = d128.Dv = 128;
    const std::vector<scenario> scenarios = {{"qwen_decode", qwen38(), {2051, 1500}, 1},
                                             {"qwen_prefill", qwen38(), {300, 211}, 150},
                                             {"d128_decode", d128, {500}, 1},
                                             {"toy_prefill", toy(), {90, 61}, 7},
                                             {"toy_decode", toy(), {120}, 1}};
    struct format {
        const char* name;
        KVQuant k, v;
    };
    const std::vector<format> formats = {{"Ktok_i8sym_Vtok_i8sym", token_quant(8, true), token_quant(8, true)},
                                         {"Ktok_u4_Vtok_u4", token_quant(4, false), token_quant(4, false)},
                                         {"Ktok_u4sym_Vtok_u4sym", token_quant(4, true), token_quant(4, true)}};
    std::vector<dense_format_case> cases;
    for (const auto& f : formats) {
        for (const auto& s : scenarios) {
            cases.push_back({std::string(s.name) + "_" + f.name, {s.d, s.key_len, s.q_len, f.k, f.v}});
        }
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         sparse_sdpa_kvq_formats,
                         ::testing::ValuesIn(dense_format_cases()),
                         [](const ::testing::TestParamInfo<dense_format_case>& info) {
                             return info.param.name;
                         });

// ---------------------------------------------------------------------------
// Performance (DISABLED_): GPU time of the whole SparsePA call (write + attention stages) per KV cache format.
// ---------------------------------------------------------------------------

struct perf_case {
    std::string name;
    KVQuant k, v;
    size_t past, q_len;
    size_t page = 16;
};

class sparse_pa_kvq_perf : public ::testing::TestWithParam<perf_case> {};

TEST_P(sparse_pa_kvq_perf, qwen38) {
    if (!ocl_available()) {
        GTEST_SKIP() << "Intel subgroup extensions are not supported on this device";
    }
    const auto& pc = GetParam();
    dims perf_dims = qwen38();
    perf_dims.page = pc.page;
    const auto c = make_case(perf_dims, pc.k, pc.v, {{pc.past, pc.q_len}}, 3);
    const auto& d = c.d;
    const KVQuant quants[2] = {pc.k, pc.v};
    const size_t D[2] = {d.Dh, d.Dv};
    std::vector<uint8_t> init[2];
    for (int which = 0; which < 2; which++) {
        const auto& hist = which == 0 ? c.key_hist[0] : c.value_hist[0];
        if (quants[which].enabled()) {
            quant_pages qp(quants[which], c.total_pages, d.Hkv, d.page, D[which]);
            cpu_write_range(qp, c, 0, hist, D[which], 0, pc.past);
            init[which] = qp.bytes;
        } else {
            std::vector<ov::float16> h(c.total_pages * d.Hkv * d.page * D[which], ov::float16(0.0f));
            for (size_t t = 0; t < pc.past; t++) {
                const size_t page = static_cast<size_t>(c.block_indices[c.page_first[0] + t / d.page]);
                for (size_t hd = 0; hd < d.Hkv; hd++) {
                    for (size_t j = 0; j < D[which]; j++) {
                        h[((page * d.Hkv + hd) * d.page + t % d.page) * D[which] + j] = ov::float16(hist[(t * d.Hkv + hd) * D[which] + j]);
                    }
                }
            }
            init[which].resize(h.size() * 2);
            std::memcpy(init[which].data(), h.data(), init[which].size());
        }
    }
    run_options opt;
    opt.profile = true;
    opt.warmup = 20;
    opt.iterations = 50;
    run_gpu(c, init[0], init[1], &opt);
    const auto kl = kv_page_layout(pc.k, d.page, d.Dh);
    const auto vl = kv_page_layout(pc.v, d.page, d.Dv);
    const double cache_mb = static_cast<double>(c.total_pages * d.Hkv * (kl.page_bytes + vl.page_bytes)) / 1e6;
    std::cout << "QSA_KVQ_PERF " << std::left << std::setw(26) << pc.name << " mode=" << std::setw(7) << (pc.q_len == 1 ? "decode" : "prefill")
              << " ctx=" << std::setw(7) << (pc.past + pc.q_len) << " q=" << std::setw(5) << pc.q_len << std::right << std::fixed << std::setprecision(4)
              << " gpu_min_ms=" << std::setw(9) << opt.gpu_ms_min << " gpu_avg_ms=" << std::setw(9) << opt.gpu_ms_avg << std::setprecision(1)
              << " cache_MB=" << std::setw(8) << cache_mb << " kernels=" << opt.kernels << " stages_ms=[";
    for (size_t i = 0; i < opt.stage_ms.size(); i++) {
        std::cout << (i ? "," : "") << std::setprecision(4) << opt.stage_ms[i];
    }
    std::cout << "]" << std::endl;
}

std::vector<perf_case> perf_cases() {
    struct fmt {
        const char* name;
        KVQuant k, v;
    };
    const std::vector<fmt> formats = {
        {"f16", {}, {}},
        {"Ktok_i8_Vtok_i8", token_quant(8, false), token_quant(8, false)},
        {"Ktok_u4_Vtok_u4", token_quant(4, false), token_quant(4, false)},
        {"Kch16_i8_Vtok_i8", channel_quant(8, false, 16), token_quant(8, false)},
        {"Kch16_u4_Vtok_u4", channel_quant(4, false, 16), token_quant(4, false)},
        {"Kch32_i8_Vtok_i8_page32", channel_quant(8, false, 32), token_quant(8, false)},
        {"Ktok_i8sym_Vtok_i8sym", token_quant(8, true), token_quant(8, true)},
        {"Kch16_u4sym_Vtok_u4sym", channel_quant(4, true, 16), token_quant(4, true)},
    };
    std::vector<perf_case> out;
    for (const auto& f : formats) {
        const size_t page = std::string(f.name).find("page32") != std::string::npos ? 32 : 16;
        for (size_t ctx : {size_t{4096}, size_t{16384}, size_t{32768}, size_t{65536}, size_t{131072}}) {
            out.push_back({f.name, f.k, f.v, ctx - 1, 1, page});
        }
        out.push_back({f.name, f.k, f.v, 0, 4096, page});
        out.push_back({f.name, f.k, f.v, 0, 16384, page});
    }
    return out;
}

INSTANTIATE_TEST_SUITE_P(DISABLED_sparse_pa_kvq_perf,
                         sparse_pa_kvq_perf,
                         ::testing::ValuesIn(perf_cases()),
                         [](const ::testing::TestParamInfo<perf_case>& info) {
                             return info.param.name + "_ctx" + std::to_string(info.param.past + info.param.q_len) + "_q" + std::to_string(info.param.q_len);
                         });

}  // namespace
