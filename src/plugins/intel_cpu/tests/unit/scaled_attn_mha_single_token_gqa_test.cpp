// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Checks the single-token SDPA kernel with a u8 cache against a scalar double-precision reference, for
// grouped-query attention (several query heads per key/value head). Runs the baseline build of the kernel.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/mha_single_token.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/plain_tensor.hpp"

using ov::intel_cpu::PlainTensor;

namespace {

struct Case {
    ov::element::Type q_prec;
    size_t B, H, Hk, q_len, kv_len, S;
    size_t group;  // quantization group, in channels (in tokens for by_channel keys)
    bool beams;
    bool causal;
    bool by_channel;
};

std::string case_name(const ::testing::TestParamInfo<Case>& info) {
    const auto& c = info.param;
    return c.q_prec.get_type_name() + "_B" + std::to_string(c.B) + "_H" + std::to_string(c.H) + "_Hk" +
           std::to_string(c.Hk) + "_q" + std::to_string(c.q_len) + "_kv" + std::to_string(c.kv_len) + "_S" +
           std::to_string(c.S) + "_grp" + std::to_string(c.group) + (c.beams ? "_beams" : "") +
           (c.causal ? "_causal" : "") + (c.by_channel ? "_bychannel" : "");
}

class MhaSingleTokenGqaTest : public ::testing::TestWithParam<Case> {
protected:
    void SetUp() override {
        c = GetParam();
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
        std::uniform_int_distribution<int> byte(0, 255);
        std::uniform_real_distribution<float> scale(0.004F, 0.012F);
        std::uniform_real_distribution<float> zp(96.0F, 160.0F);

        // Layouts of the node: the query is a permuted [B, L, H, S] tensor and the caches are views into
        // buffers with room for more tokens.
        q_full.resize({c.B, c.q_len, c.H, c.S}, c.q_prec.size(), c.q_prec);
        query = q_full.permute({0, 2, 1, 3});
        for (size_t i = 0; i < c.B * c.q_len * c.H * c.S; i++) {
            const float v = unit(rng);
            if (c.q_prec == ov::element::f16) {
                q_full.ptr<ov::float16>()[i] = ov::float16(v);
            } else {
                q_full.ptr<float>()[i] = v;
            }
        }
        k_full.resize<uint8_t>({c.B, c.Hk, c.kv_len + 8, c.S});
        v_full.resize<uint8_t>({c.B, c.Hk, c.kv_len + 8, c.S});
        for (size_t i = 0; i < c.B * c.Hk * (c.kv_len + 8) * c.S; i++) {
            k_full.ptr<uint8_t>()[i] = static_cast<uint8_t>(byte(rng));
            v_full.ptr<uint8_t>()[i] = static_cast<uint8_t>(byte(rng));
        }
        present_key = k_full.slice(2, 0, static_cast<int>(c.kv_len));
        present_value = v_full.slice(2, 0, static_cast<int>(c.kv_len));

        // Interleaved {scale, zero point} per token and channel group; by-channel keys instead keep a row of
        // per-channel scales and a row of zero points per token group.
        if (c.by_channel) {
            k_scale_zp.resize<float>({2 * ((c.kv_len + c.group - 1) / c.group), c.B, c.Hk, c.S});
        } else {
            k_scale_zp.resize<float>({c.kv_len, c.B, c.Hk, 2 * (c.S / c.group)});
        }
        v_scale_zp.resize<float>({c.kv_len, c.B, c.Hk, 2 * (c.S / c.group)});
        for (size_t i = 0; i < k_scale_zp.size(0) * k_scale_zp.stride(0); i++) {
            const bool is_zp = c.by_channel ? (i / k_scale_zp.stride(0)) % 2 : i % 2;
            k_scale_zp.ptr<float>()[i] = is_zp ? zp(rng) : scale(rng);
        }
        for (size_t i = 0; i < v_scale_zp.size(0) * v_scale_zp.stride(0); i++) {
            v_scale_zp.ptr<float>()[i] = i % 2 ? zp(rng) : scale(rng);
        }
        if (c.beams) {
            beams.resize<int32_t>({c.B, c.kv_len});
            std::uniform_int_distribution<int32_t> pick(0, static_cast<int32_t>(c.B) - 1);
            for (size_t i = 0; i < c.B * c.kv_len; i++) {
                beams.ptr<int32_t>()[i] = pick(rng);
            }
        }
    }

    double query_at(size_t b, size_t h, size_t l, size_t s) const {
        return c.q_prec == ov::element::f16 ? static_cast<float>(*query.ptr<ov::float16>(b, h, l, s))
                                            : *query.ptr<float>(b, h, l, s);
    }
    double key_at(size_t b, size_t hk, size_t t, size_t s) const {
        double scale = 0.0;
        double zero_point = 0.0;
        if (c.by_channel) {
            scale = *k_scale_zp.ptr<float>(t / c.group * 2, b, hk, s);
            zero_point = *k_scale_zp.ptr<float>(t / c.group * 2 + 1, b, hk, s);
        } else {
            scale = k_scale_zp.ptr<float>(t, b, hk)[s / c.group * 2];
            zero_point = k_scale_zp.ptr<float>(t, b, hk)[s / c.group * 2 + 1];
        }
        return (present_key.ptr<uint8_t>(b, hk, t)[s] - zero_point) * scale;
    }
    double value_at(size_t b, size_t hk, size_t t, size_t s) const {
        const float* p = v_scale_zp.ptr<float>(t, b, hk) + s / c.group * 2;
        return (present_value.ptr<uint8_t>(b, hk, t)[s] - static_cast<double>(p[1])) * p[0];
    }

    // Attention in double precision: query head h reads key/value head h / G, as the repeated key/value heads of a
    // GQA model do. With causal, the q_len tokens are the last ones of the cache (the node's fused past).
    std::vector<double> reference() const {
        const size_t G = c.H / c.Hk;
        std::vector<double> out(c.B * c.H * c.q_len * c.S, 0.0);
        std::vector<double> w(c.kv_len);
        for (size_t b = 0; b < c.B; b++) {
            for (size_t h = 0; h < c.H; h++) {
                for (size_t l = 0; l < c.q_len; l++) {
                    const size_t valid = c.causal ? c.kv_len - c.q_len + l + 1 : c.kv_len;
                    double max = -INFINITY;
                    double sum = 0.0;
                    for (size_t t = 0; t < valid; t++) {
                        const size_t bk = c.beams ? beams.ptr<int32_t>(b)[t] : b;
                        double dot = 0.0;
                        for (size_t s = 0; s < c.S; s++) {
                            dot += query_at(b, h, l, s) * key_at(bk, h / G, t, s);
                        }
                        w[t] = dot / std::sqrt(static_cast<double>(c.S));
                        max = std::max(max, w[t]);
                    }
                    for (size_t t = 0; t < valid; t++) {
                        w[t] = std::exp(w[t] - max);
                        sum += w[t];
                    }
                    for (size_t t = 0; t < valid; t++) {
                        const size_t bk = c.beams ? beams.ptr<int32_t>(b)[t] : b;
                        for (size_t s = 0; s < c.S; s++) {
                            out[((b * c.H + h) * c.q_len + l) * c.S + s] += w[t] / sum * value_at(bk, h / G, t, s);
                        }
                    }
                }
            }
        }
        return out;
    }

    // Runs the kernel with the given number of threads; returns out[b][h][l][s].
    std::vector<float> run(int threads) {
        PlainTensor output;
        output.resize({c.B, c.H, c.q_len, c.S}, c.q_prec.size(), c.q_prec);
        // The node allocates the score buffer as f32 rows padded to 16, whatever the kernel's score type.
        PlainTensor attn_w;
        attn_w.resize<float>({c.B, c.H, c.q_len, (c.kv_len + 15) / 16 * 16});
        PlainTensor attn_score;
        PlainTensor head_sum;
        PlainTensor none;
        auto cpu_parallel = std::make_shared<ov::intel_cpu::CpuParallel>(ov::intel_cpu::TbbPartitioner::STATIC);
        auto call = [&]() {
            ov::Extensions::Cpu::XARCH::mha_single_token(query,
                                                         present_key,
                                                         present_value,
                                                         none,
                                                         none,
                                                         beams,
                                                         output,
                                                         attn_w,
                                                         attn_score,
                                                         false,
                                                         c.causal,
                                                         1.0F / std::sqrt(static_cast<float>(c.S)),
                                                         k_scale_zp,
                                                         v_scale_zp,
                                                         head_sum,
                                                         c.group,
                                                         c.group,
                                                         c.by_channel,
                                                         none,
                                                         cpu_parallel);
        };
#if OV_THREAD_USE_TBB
        tbb::task_arena(threads).execute([&]() {
            ASSERT_EQ(parallel_get_max_threads(), threads);
            call();
        });
#else
        (void)threads;  // without TBB the host's thread count is used, so only one value path may run
        call();
#endif
        std::vector<float> out(c.B * c.H * c.q_len * c.S);
        for (size_t i = 0; i < out.size(); i++) {
            out[i] = c.q_prec == ov::element::f16 ? static_cast<float>(output.ptr<ov::float16>()[i])
                                                  : output.ptr<float>()[i];
        }
        return out;
    }

    Case c{};
    PlainTensor q_full, query, k_full, v_full, present_key, present_value, k_scale_zp, v_scale_zp, beams;
};

TEST_P(MhaSingleTokenGqaTest, MatchesReference) {
    const auto ref = reference();
    // The plugin object library compiles the kernel once with the default ISA flags, as this file is compiled, so the
    // macro below tells which kernel instantiation runs for f16 queries.
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    // f16 scores, weights and value accumulators, without the NEON_FP16 / SVE f16 loops in this build.
    constexpr double f16_tol = 5e-3;
#else
    // f32 scores and accumulators; only the output is rounded to f16.
    constexpr double f16_tol = 2e-3;
#endif
    // Absolute tolerances on outputs of magnitude below 1: several times the largest error measured on Apple M5, and
    // with f32 accumulators several times the f16 rounding of the output (at most 2.4e-4).
    const double tol = c.q_prec == ov::element::f16 ? f16_tol : 1e-5;
    // With 1 thread every batch has at least as many sequences as threads, which selects the per-sequence value
    // path; with 3 threads, batches of 1 and 2 take the path that splits the cache across threads.
    for (const int threads : {1, 3}) {
        const auto out = run(threads);
        for (size_t i = 0; i < out.size(); i++) {
            ASSERT_NEAR(out[i], ref[i], tol) << threads << " threads, index " << i;
        }
    }
}

const auto f32 = ov::element::f32;
const auto f16 = ov::element::f16;

// clang-format off
const std::vector<Case> cases = {
    //  q    B   H  Hk q_len kv_len  S  group beams  causal by_channel
    // G = H / Hk of 1-5 and 7: tiles of up to 4 heads with remainders of 1, 2 and 3; G = 1 does not group.
    {f32, 1,  2, 2, 1,   129,  128, 128, false, false, false},
    {f32, 1,  4, 2, 1,   129,  128, 128, false, false, false},
    {f32, 1,  6, 2, 1,   129,  128, 128, false, false, false},
    {f32, 1,  8, 2, 1,   129,  128, 128, false, false, false},
    {f32, 1, 10, 2, 1,   129,  128, 128, false, false, false},
    {f32, 1, 14, 2, 1,   129,  128, 128, false, false, false},
    // Several quantization groups, and a group size that leaves a scalar tail.
    {f32, 1,  8, 2, 1,    33,   64,  32, false, false, false},
    {f32, 1,  8, 2, 1,    33,   80,  40, false, false, false},
    // One query head per key/value head over several key groups with a tail: the per-head key path.
    {f32, 1,  2, 2, 1,    33,   80,  40, false, false, false},
    // A single cached token.
    {f32, 1, 14, 2, 1,     1,   64,  64, false, false, false},
    // Query tokens with a past (rows are token and head pairs), beams, and causal truncation.
    {f32, 2,  6, 2, 3,    33,   64,  64, true,  false, false},
    {f32, 4,  6, 2, 3,    33,   64,  64, true,  false, false},
    {f32, 2,  8, 2, 3,    40,   64,  64, false, true,  false},
    // Causal truncation with tiles that span query tokens (G = 3).
    {f32, 2,  6, 2, 3,    40,   64,  64, false, true,  false},
    // Keys quantized per channel keep the per-head key path; values still use the grouped path.
    {f32, 1,  8, 2, 1,    64,   64,  32, false, false, true},
    // f16 queries: converted to f32 for the grouped key path (G = 1 in the third case, which does not group and, with
    // GCC, is the only test of common.hpp's f16 u8 dot product); f16 scores and value rows when the macro above is
    // defined (as with Apple clang), else the f32 path.
    {f16, 2,  6, 2, 3,    33,   64,  64, false, false, false},
    {f16, 4,  6, 2, 3,    33,   64,  64, true,  false, false},
    {f16, 1,  2, 2, 1,    33,   64,  64, false, false, false},
    {f16, 2,  6, 2, 3,    40,   64,  64, false, true,  false},
    // f16 queries over several quantization groups, with an 8-element tail in the 16-wide NEON loops; the f16 value
    // loops of the NEON_FP16 and SVE builds are covered by smoke_ConcatSDPGQAF16Test_ValueGroups.
    {f16, 1,  8, 2, 1,    33,   80,  40, false, false, false},
    {f16, 1,  2, 2, 1,    33,   80,  40, false, false, false},
};
// clang-format on

INSTANTIATE_TEST_SUITE_P(smoke_MhaSingleTokenGqa, MhaSingleTokenGqaTest, ::testing::ValuesIn(cases), case_name);

}  // namespace
