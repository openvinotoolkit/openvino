// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_kv_cache_codec.hpp"

#include <algorithm>
#include <cstring>

#include "codecs/codec_kernels.hpp"
#include "codecs/codecs.hpp"
#include "common.hpp"
#include "mha_kv_cache_reduce.hpp"
#include "nodes/kernels/simd/simd.hpp"
#include "openvino/core/parallel.hpp"
#include "softmax_kernel.hpp"

namespace ov::Extensions::Cpu::XARCH {

// Prefetch lookahead: number of records ahead to prefetch.
// At ~50 cycles/record and ~200 cycle DRAM latency, 4-8 records ahead hides latency.
static constexpr int PREFETCH_AHEAD = 8;

// ---------------------------------------------------------------------------
// mha_kv_cache — fused multi-head attention over raw or quantized KV cache.
// ---------------------------------------------------------------------------

using ov::Extensions::Cpu::CacheCodec;
using ov::intel_cpu::CpuParallelPtr;
using ov::intel_cpu::PlainTensor;

struct NoOpInit {
    void operator()(size_t /*unused*/) const {}
};

struct MhaKvTraversal {
    size_t batch_size = 0;
    size_t num_kv_heads = 0;
    size_t kv_len = 0;
    size_t heads_per_kv_group = 0;
    int nthr = 0;
};

// Generic parallel loop over KV cache tokens.
// Covers both Q·K scoring and V accumulation — both traverse the same [B, num_kv_heads, kv_len]
// cache structure with identical parallelization and work splitting.
//
//   direct_fn(run_len, num_group_heads, head_dim, b, h_group, start_pos, ithr)
//     Processes all tokens in a contiguous run. Computes pointers, handles prefetching,
//     and accesses data via captures.
//   per_thread_init(ithr)
//     Called once per thread before work (e.g. memset accumulators). Default: no-op.
template <typename DirectFn, typename InitFn = NoOpInit>
static void mha_foreach_kv(const MhaKvTraversal& traversal,
                           size_t head_dim,
                           DirectFn&& direct_fn,
                           InitFn per_thread_init = {}) {
    parallel_nt_static(traversal.nthr, [&](const size_t ithr, const size_t nthr) {
        const auto [B, num_kv_heads, kv_len, heads_per_kv_group, traversal_nthr] = traversal;
        (void)traversal_nthr;
        per_thread_init(ithr);

        size_t start{0}, end{0};
        splitter(B * num_kv_heads * kv_len, nthr, ithr, start, end);
        if (start >= end) {
            return;
        }

        size_t pos = 0, b = 0, h_group = 0;
        parallel_it_init(start, h_group, num_kv_heads, b, B, pos, kv_len);

        size_t iwork = start;
        while (iwork < end) {
            size_t run_len = std::min(kv_len - pos, end - iwork);

            direct_fn(run_len, static_cast<int>(heads_per_kv_group), static_cast<int>(head_dim), b, h_group, pos, ithr);

            for (size_t r = 0; r < run_len; r++) {
                parallel_it_step(h_group, num_kv_heads, b, B, pos, kv_len);
            }
            iwork += run_len;
        }
    });
}

static void mha_softmax(const PlainTensor& attn_w,
                        const PlainTensor& alibi_mask,
                        const PlainTensor& attention_mask,
                        const PlainTensor& sink_input,
                        size_t B,
                        size_t num_q_heads,
                        size_t q_len,
                        size_t kv_len,
                        bool auto_causal,
                        float d_scale,
                        const CpuParallelPtr& cpu_parallel) {
    auto precision = ov::element::f32;
    auto softmax_body = [&](size_t b, size_t h, size_t m) {
        auto ncausal = auto_causal ? (kv_len - q_len + m + 1) : kv_len;
        float* alibi_ptr = alibi_mask ? &alibi_mask.at<float>({b, h, m, 0}, true) : nullptr;
        uint8_t* attn_mask_ptr = nullptr;
        auto attn_mask_prec = attention_mask.get_precision();
        if (attention_mask) {
            attn_mask_ptr = &attention_mask.at<uint8_t>({b, h, m, 0}, true);
        }
        float* sink = nullptr;
        if (sink_input) {
            sink = &sink_input.at<float>({b, h, m, 0}, true);
        }
        attn_softmax_kernel<float>(attn_w.ptr<float>(b, h, m),
                                   attn_w.ptr<float>(b, h, m),
                                   d_scale,
                                   alibi_ptr,
                                   attn_mask_ptr,
                                   nullptr,
                                   false,
                                   ncausal,
                                   kv_len,
                                   attn_mask_prec,
                                   precision,
                                   sink);
    };
    if (q_len == 1) {
        cpu_parallel->parallel_for2d(B, num_q_heads, [&](size_t b, size_t h) {
            softmax_body(b, h, 0);
        });
    } else {
        cpu_parallel->parallel_for3d(B, num_q_heads, q_len, softmax_body);
    }
}

// ---------------------------------------------------------------------------
// QK dot product and V accumulation through record codecs.
// ---------------------------------------------------------------------------

template <typename QT, typename View>
static inline float record_qk_dot(const uint8_t* k, const QT* q, int head_dim, const View& view) {
    if constexpr (is_affine_v<View>) {
        // Deferred dequant: raw dot per group, then affine correction.
        const int gd = view.group_dim();
        const int ng = view.n_groups(head_dim);
        float sum = 0.0F;
        for (int g = 0; g < ng; g++) {
            auto plan = DecodePlan{view.group_decoder(g), view.group_params(g)};
            constexpr int bpe = decltype(plan.decoder)::bits;
            const auto* gk = k + static_cast<size_t>(g * gd) * bpe / 8;
            float raw = codec_dot<QT>(gk, q + g * gd, gd, [&](int, auto) {
                return plan;
            });
            sum += view.correct_dot(raw, g);
        }
        return sum;
    } else if constexpr (is_head_grouped_v<View>) {
        const int gd = view.group_dim();
        const int ng = view.n_groups(head_dim);
        float sum = 0.0F;
        for (int g = 0; g < ng; g++) {
            auto plan = DecodePlan{view.group_decoder(g), view.group_params(g)};
            constexpr int bpe = decltype(plan.decoder)::bits;
            const auto* gk = k + static_cast<size_t>(g * gd) * bpe / 8;
            sum += codec_dot<QT>(gk, q + g * gd, gd, [&](int, auto) {
                return plan;
            });
        }
        return sum;
    } else if constexpr (has_params_for_v<View>) {
        // By-channel: plan resolved per-j by the view.
        using D = typename View::decoder_t;
        return codec_dot<QT>(k, q, head_dim, [&view](int j, auto a) {
            return DecodePlan{D{}, view.params_for(j, a)};
        });
    } else {
        // RecordView<RawDecoder<T>>: constant plan, no dequant params.
        auto plan = DecodePlan{view.decoder, NoParams{}};
        return codec_dot<QT>(k, q, head_dim, [&](int, auto) {
            return plan;
        });
    }
}

// Unified per-token V weighted accumulation through record view.
template <typename View>
static inline void record_v_accum(const uint8_t* v,
                                  StridedData<const float> weights,
                                  StridedData<float> accum,
                                  int num_group_heads,
                                  int head_dim,
                                  const View& view) {
    if constexpr (is_head_grouped_v<View>) {
        const int gd = view.group_dim();
        const int ng = view.n_groups(head_dim);
        for (int g = 0; g < ng; g++) {
            auto plan = DecodePlan{view.group_decoder(g), view.group_params(g)};
            constexpr int bpe = decltype(plan.decoder)::bits;
            const auto* gv = v + static_cast<size_t>(g * gd) * bpe / 8;
            StridedData<float> ga{accum.data + g * gd, accum.stride};
            codec_weighted_accum(
                gv,
                gd,
                [&](int, auto) {
                    return plan;
                },
                1.0F,
                weights,
                ga,
                num_group_heads);
        }
    } else if constexpr (has_params_for_v<View>) {
        // By-channel: plan resolved per-j by the view.
        using D = typename View::decoder_t;
        codec_weighted_accum(
            v,
            head_dim,
            [&view](int j, auto a) {
                return DecodePlan{D{}, view.params_for(j, a)};
            },
            1.0F,
            weights,
            accum,
            num_group_heads);
    } else {
        // RecordView<RawDecoder<T>>: constant plan, no dequant params.
        auto plan = DecodePlan{view.decoder, NoParams{}};
        codec_weighted_accum(
            v,
            head_dim,
            [&](int, auto) {
                return plan;
            },
            1.0F,
            weights,
            accum,
            num_group_heads);
    }
}

// ---------------------------------------------------------------------------
// QKScorer — bound QK scorer for Phase 1.
// Owns query head access, view resolution, and the call to record_qk_dot.
// Constructed once per (q_precision, codec) dispatch, passed to score_tokens.
// ---------------------------------------------------------------------------

struct KVEntryContext {
    size_t start_pos;
    size_t h_group;
    int head_dim;
};

template <typename Q, typename RecordView>
struct QKScorer {
    Q q;
    RecordView record_view;
    KVEntryContext ctx;

    float operator()(const uint8_t* k, int g, size_t t, size_t b_kv) const {
        using QT = std::remove_const_t<std::remove_pointer_t<decltype(q.data)>>;
        const QT* q_head = q[g];
        if constexpr (has_for_head_v<RecordView>) {
            auto head_view = record_view.for_head(g);
            auto token_view = head_view.for_token(ctx.start_pos, ctx.h_group, t, b_kv);
            return record_qk_dot<QT>(k, q_head, ctx.head_dim, token_view);
        } else if constexpr (is_token_indexed_v<RecordView>) {
            auto token_view = record_view.for_token(ctx.start_pos, ctx.h_group, t, b_kv);
            return record_qk_dot<QT>(k, q_head, ctx.head_dim, token_view);
        } else {
            return record_qk_dot<QT>(k, q_head, ctx.head_dim, record_view);
        }
    }
};

template <typename Q, typename RecordView>
QKScorer(Q, RecordView, KVEntryContext) -> QKScorer<Q, RecordView>;

// ---------------------------------------------------------------------------
// VAccumulator — bound V accumulator for Phase 3.
// Owns view resolution and the call to record_v_accum.
// Constructed once per codec dispatch, passed to accum_tokens.
// ---------------------------------------------------------------------------

template <typename RecordView>
struct VAccumulator {
    RecordView record_view;
    KVEntryContext ctx;

    void operator()(const uint8_t* v,
                    StridedData<const float> w,
                    StridedData<float> a,
                    int num_group_heads,
                    size_t t,
                    size_t b_kv) const {
        if constexpr (is_token_indexed_v<RecordView>) {
            auto token_view = record_view.for_token(ctx.start_pos, ctx.h_group, t, b_kv);
            record_v_accum(v, w, a, num_group_heads, ctx.head_dim, token_view);
        } else {
            record_v_accum(v, w, a, num_group_heads, ctx.head_dim, record_view);
        }
    }
};

template <typename RecordView>
VAccumulator(RecordView, KVEntryContext) -> VAccumulator<RecordView>;

// Unified per-token QK scoring loop.
// Handles beam table resolution, prefetching, and multi-head scoring.
template <typename TokenScoreFn>
static void score_tokens(const uint8_t* kv_base,
                         size_t stride_batch,
                         size_t stride_pos,
                         const int32_t* beam_tbl,
                         size_t b,
                         StridedData<float> scores,
                         size_t run_len,
                         int num_group_heads,
                         size_t pf_bytes,
                         TokenScoreFn score_fn) {
    for (size_t t = 0; t < run_len; t++) {
        const size_t b_kv = beam_tbl ? static_cast<size_t>(beam_tbl[t]) : b;
        const uint8_t* k_record = kv_base + b_kv * stride_batch + t * stride_pos;
        if (t + PREFETCH_AHEAD < run_len) {
            [[maybe_unused]] const size_t prefetch_t = t + PREFETCH_AHEAD;
            [[maybe_unused]] const size_t prefetch_b = beam_tbl ? static_cast<size_t>(beam_tbl[prefetch_t]) : b;
            [[maybe_unused]] const uint8_t* prefetch_record =
                kv_base + prefetch_b * stride_batch + prefetch_t * stride_pos;
            prefetch_bytes(pf_bytes, _MM_HINT_T0, 0, prefetch_record);
        }
        for (int g = 0; g < num_group_heads; g++) {
            scores[g][t] = score_fn(k_record, g, t, b_kv);
        }
    }
}

// Unified per-token V accumulation loop.
template <typename TokenAccumFn>
static void accum_tokens(const uint8_t* kv_base,
                         size_t stride_batch,
                         size_t stride_pos,
                         const int32_t* beam_tbl,
                         size_t b,
                         StridedData<const float> weights,
                         StridedData<float> accum,
                         int num_group_heads,
                         size_t run_len,
                         size_t pf_bytes,
                         TokenAccumFn accum_fn) {
    for (size_t t = 0; t < run_len; t++) {
        const size_t b_kv = beam_tbl ? static_cast<size_t>(beam_tbl[t]) : b;
        const uint8_t* v_record = kv_base + b_kv * stride_batch + t * stride_pos;
        if (t + PREFETCH_AHEAD < run_len) {
            [[maybe_unused]] const size_t prefetch_t = t + PREFETCH_AHEAD;
            [[maybe_unused]] const size_t prefetch_b = beam_tbl ? static_cast<size_t>(beam_tbl[prefetch_t]) : b;
            [[maybe_unused]] const uint8_t* prefetch_record =
                kv_base + prefetch_b * stride_batch + prefetch_t * stride_pos;
            prefetch_bytes(pf_bytes, _MM_HINT_T0, 0, prefetch_record);
        }
        // Offset weights by t: weights[h][t] is the weight for head h at token t.
        StridedData<const float> weights_at_t{weights.data + t, weights.stride};
        accum_fn(v_record, weights_at_t, accum, num_group_heads, t, b_kv);
    }
}

// ---------------------------------------------------------------------------
// Q precision dispatch helper: resolves runtime Q element type to typed pointer.
// ---------------------------------------------------------------------------
template <typename Fn>
static void dispatch_q_precision(const PlainTensor& q_input,
                                 size_t b,
                                 size_t h_start,
                                 ov::element::Type q_precision,
                                 Fn&& fn,
                                 size_t q_idx = 0) {
    const size_t q_stride = q_input.stride(1);
    if (q_precision == ov::element::f16) {
        fn(StridedData<const ov::float16>{q_input.ptr<ov::float16>(b, h_start, q_idx), q_stride});
    } else if (q_precision == ov::element::bf16) {
        fn(StridedData<const ov::bfloat16>{q_input.ptr<ov::bfloat16>(b, h_start, q_idx), q_stride});
    } else {
        fn(StridedData<const float>{q_input.ptr<float>(b, h_start, q_idx), q_stride});
    }
}

// ---------------------------------------------------------------------------
// Total byte size of one cache record for `codec` at head_dim `head_dim`.
// Used for prefetch sizing.
// ---------------------------------------------------------------------------
template <typename View>
static inline size_t codec_record_bytes(const View& view, int head_dim) {
    (void)view;
    return static_cast<size_t>(head_dim) * View::decoder_t::bits / 8;
}

// ---------------------------------------------------------------------------
// Codec dispatch: maps runtime CacheCodec to concrete HeadCodec type.
// ---------------------------------------------------------------------------
template <typename Fn>
static void dispatch_codec(CacheCodec codec,
                           int head_dim,
                           size_t group_size,
                           const PlainTensor& scale_zp,
                           Fn&& fn,
                           const float* q_group_sums = nullptr,
                           size_t q_group_sums_stride = 0) {
    (void)head_dim;
    switch (codec) {
    case CacheCodec::U8:
        if (q_group_sums) {
            fn(AffineRecordView{scale_zp, group_size, q_group_sums, q_group_sums_stride});
        } else {
            fn(GroupedRecordView<U8Decoder>{scale_zp, group_size});
        }
        break;
    case CacheCodec::U4:
        fn(GroupedRecordView<U4Decoder>{scale_zp, group_size});
        break;
    case CacheCodec::U8_BY_CHANNEL:
        fn(ByChannelRecordView{scale_zp, group_size});
        break;
    case CacheCodec::RAW_F32:
        fn(RecordView<RawDecoder<float>>{{}});
        break;
    case CacheCodec::RAW_F16:
        fn(RecordView<RawDecoder<ov::float16>>{{}});
        break;
    case CacheCodec::RAW_BF16:
        fn(RecordView<RawDecoder<ov::bfloat16>>{{}});
        break;
    }
}

// ---------------------------------------------------------------------------

void mha_kv_cache(PlainTensor& q_input,
                  const PlainTensor& key_cache,
                  const PlainTensor& packed_value,
                  const PlainTensor& alibi_mask,
                  const PlainTensor& attention_mask,
                  const PlainTensor& beams,
                  PlainTensor& output_emb,
                  PlainTensor& buf_attn_w,
                  PlainTensor& buf_attn_score,
                  bool has_out_transpose,
                  float d_scale,
                  CacheCodec k_codec,
                  CacheCodec v_codec,
                  bool auto_causal,
                  const PlainTensor& sink_input,
                  const CpuParallelPtr& cpu_parallel,
                  const PlainTensor& k_scale_zp,
                  size_t key_group_size,
                  const PlainTensor& v_scale_zp,
                  size_t value_group_size,
                  ov::element::Type q_precision,
                  size_t value_head_dim) {
    // ---------------------------------------------------------------------------
    // Setup: dimensions, scratch buffers, precomputed constants.
    // ---------------------------------------------------------------------------
    const auto B = q_input.size(0);
    const auto num_q_heads = q_input.size(1);
    const auto q_len = q_input.size(2);
    const auto S = q_input.size(3);  // K head dimension (= Q head dimension)
    const auto SV = value_head_dim;  // V head dimension (may differ from S)
    const auto num_kv_heads = key_cache.size(1);
    const auto kv_len = key_cache.size(2);
    const size_t heads_per_kv_group = num_q_heads / num_kv_heads;
    const auto nthr = parallel_get_max_threads();
    const MhaKvTraversal kv_traversal{B, num_kv_heads, kv_len, heads_per_kv_group, nthr};

    if (d_scale == 0.0F) {
        d_scale = 1.0F / std::sqrt(static_cast<float>(S));
    }

    buf_attn_w.resize<float>({B, num_q_heads, q_len, (kv_len + 15) / 16 * 16});
    buf_attn_score.resize<float>({static_cast<size_t>(nthr), B, q_len, num_q_heads, SV});

    // Precompute per-group Q sums for affine u8 deferred dequant optimization.
    // q_group_sums[b, h, m, group] = sum(q[b, h, m, group*gs .. (group+1)*gs]).
    // AVX2-only: defer u8 zp subtraction from per-element to per-group-after-dot.
    // AVX-512 has enough throughput to absorb the per-element sub — not worth the overhead.
    constexpr bool is_avx2 = simd::f32::isa_value == simd::isa::avx2;
    const bool use_affine_k = is_avx2 && (k_codec == CacheCodec::U8);
    const size_t n_key_groups = use_affine_k ? S / key_group_size : 0;
    PlainTensor q_group_sums_buf;
    if (use_affine_k) {
        q_group_sums_buf.resize<float>({B, num_q_heads, q_len, n_key_groups});
        cpu_parallel->parallel_for3d(B, num_q_heads, q_len, [&](size_t b, size_t h, size_t m_idx) {
            auto* sums = q_group_sums_buf.ptr<float>(b, h, m_idx);
            dispatch_q_precision(
                q_input,
                b,
                h,
                q_precision,
                [&](auto q) {
                    using QT = std::remove_const_t<std::remove_pointer_t<decltype(q.data)>>;
                    const QT* q_head = q[0];  // single head at (b, h)
                    for (size_t g = 0; g < n_key_groups; g++) {
                        constexpr int W = simd::f32::width;
                        simd::f32 acc;
                        const size_t offset = g * key_group_size;
                        size_t i = 0;
                        for (; i + W <= key_group_size; i += W) {
                            acc = acc + simd::load<simd::f32>(q_head + offset + i);
                        }
                        sums[g] = reduce(acc);
                        for (; i < key_group_size; i++) {
                            sums[g] += static_cast<float>(q_head[offset + i]);
                        }
                    }
                },
                m_idx);
        });
    }

    // For each query position m, phases 1-4 run independently. When q_len=1
    // (single-token decode) these loops execute once. When q_len>1 (fuse_concat
    // prompt), each position gets its own scores, softmax, and accumulation.

    // ---------------------------------------------------------------------------
    // Phase 1: Q·K scores for all query positions.
    // ---------------------------------------------------------------------------
    for (size_t m = 0; m < q_len; m++) {
        mha_foreach_kv(
            kv_traversal,
            S,
            [&, k_codec, m](size_t run_len,
                            int num_group_heads,
                            int head_dim,
                            size_t b,
                            size_t h_group,
                            size_t start_pos,
                            size_t /*ithr*/) {
                const size_t h_start = h_group * heads_per_kv_group;
                const auto* kv_base = static_cast<const uint8_t*>(key_cache.ptr_v(size_t{0}, h_group, start_pos));
                const size_t stride_batch = key_cache.stride_bytes(0);
                const size_t stride_pos = key_cache.stride_bytes(2);
                const bool use_beams = beams && B > 1;
                const int32_t* beam_tbl_ptr = use_beams ? beams.ptr<int32_t>(b) + start_pos : nullptr;
                float* scores_row_base = buf_attn_w.ptr<float>(b, h_start, m) + start_pos;
                StridedData<float> scores{scores_row_base, buf_attn_w.stride(1)};
                const KVEntryContext entry_ctx{start_pos, h_group, head_dim};

                // q_group_sums base for first head in group; stride to step between heads.
                const float* q_group_sums = use_affine_k ? q_group_sums_buf.ptr<float>(b, h_start, m) : nullptr;
                const size_t q_group_sums_stride = use_affine_k ? q_group_sums_buf.stride(1) : 0;
                dispatch_q_precision(
                    q_input,
                    b,
                    h_start,
                    q_precision,
                    [&](auto q) {
                        dispatch_codec(
                            k_codec,
                            head_dim,
                            key_group_size,
                            k_scale_zp,
                            [&](auto record_view) {
                                auto scorer = QKScorer{q, record_view, entry_ctx};
                                score_tokens(kv_base,
                                             stride_batch,
                                             stride_pos,
                                             beam_tbl_ptr,
                                             b,
                                             scores,
                                             run_len,
                                             num_group_heads,
                                             codec_record_bytes(record_view, head_dim),
                                             scorer);
                            },
                            q_group_sums,
                            q_group_sums_stride);
                    },
                    m);
            });
    }

    // ---------------------------------------------------------------------------
    // Phase 2: Softmax — runs over all query positions at once.
    // ---------------------------------------------------------------------------
    mha_softmax(buf_attn_w,
                alibi_mask,
                attention_mask,
                sink_input,
                B,
                num_q_heads,
                q_len,
                kv_len,
                auto_causal,
                d_scale,
                cpu_parallel);

    // ---------------------------------------------------------------------------
    // Phases 3+4: V accumulation + reduce, per query position.
    // ---------------------------------------------------------------------------
    for (size_t m = 0; m < q_len; m++) {
        // Phase 3: V accumulation for query position m.
        mha_foreach_kv(
            kv_traversal,
            SV,
            [&, v_codec, m](size_t run_len,
                            int num_group_heads,
                            int head_dim,
                            size_t b,
                            size_t h_group,
                            size_t start_pos,
                            size_t ithr) {
                const size_t h_start = h_group * heads_per_kv_group;
                const auto* kv_base = static_cast<const uint8_t*>(packed_value.ptr_v(size_t{0}, h_group, start_pos));
                const size_t stride_batch = packed_value.stride_bytes(0);
                const size_t stride_pos = packed_value.stride_bytes(2);
                const bool use_beams = beams && B > 1;
                const int32_t* beam_tbl_ptr = use_beams ? beams.ptr<int32_t>(b) + start_pos : nullptr;
                const float* weights_row_base = buf_attn_w.ptr<float>(b, h_start, m) + start_pos;
                StridedData<const float> weights{weights_row_base, buf_attn_w.stride(1)};
                auto* accum_row_base = buf_attn_score.ptr<float>(ithr, b, m, h_start);
                StridedData<float> accum{accum_row_base, buf_attn_score.stride(3)};
                const KVEntryContext entry_ctx{start_pos, h_group, head_dim};

                dispatch_codec(v_codec, head_dim, value_group_size, v_scale_zp, [&](auto record_view) {
                    auto vaccum = VAccumulator{record_view, entry_ctx};
                    accum_tokens(kv_base,
                                 stride_batch,
                                 stride_pos,
                                 beam_tbl_ptr,
                                 b,
                                 weights,
                                 accum,
                                 num_group_heads,
                                 run_len,
                                 codec_record_bytes(record_view, head_dim),
                                 vaccum);
                });
            },
            [&](size_t ithr) {
                for (size_t b = 0; b < B; ++b) {
                    std::memset(buf_attn_score.ptr<float>(ithr, b, m, 0, 0),
                                0,
                                buf_attn_score.stride(2) * sizeof(float));
                }
            });

        // Phase 4: Reduce for query position m.
        mha_reduce(buf_attn_score,
                   output_emb,
                   has_out_transpose,
                   B,
                   num_q_heads,
                   1,  // reduce one query position at a time
                   SV,
                   nthr,
                   cpu_parallel,
                   m);
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
