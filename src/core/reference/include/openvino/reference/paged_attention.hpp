#pragma once
// paged_attention_kernel.hpp — reference PagedAttention using PagedCacheManager

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_attention.hpp"

// use the renamed manager
#include "openvino/core/paged_cache_manager.hpp"

#ifndef PA_DEBUG
#    define PA_DEBUG 0
#endif

namespace ov::reference {

namespace pa_math {
template <typename T>
inline T compute_dot_product(const T* a, const T* b, size_t n) {
    T s = T(0);
    for (size_t i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}
template <typename T>
inline void compute_softmax_in_place(std::vector<T>& v) {
    const T m = *std::max_element(v.begin(), v.end());
    T sum = T(0);
    for (auto& x : v) {
        x = std::exp(x - m);
        sum += x;
    }
    const T inv = sum ? T(1) / sum : T(0);
    for (auto& x : v)
        x *= inv;
}
}  // namespace pa_math

namespace pa_rotary {
template <typename T>
inline void apply_rotary_embedding_to_vector(T* vec, size_t head_size, const T* trig_lut, size_t trig_index) {
    const size_t half = head_size / 2;
    const T* row = trig_lut + trig_index * head_size;
    for (size_t i = 0; i < half; ++i) {
        const T x0 = vec[2 * i], x1 = vec[2 * i + 1];
        vec[2 * i] = x0 * row[i] - x1 * row[half + i];
        vec[2 * i + 1] = x0 * row[half + i] + x1 * row[i];
    }
}
inline bool block_has_rotary(int32_t block_id,
                             const int32_t* rotated_block_indices,
                             size_t rotated_block_count,
                             int32_t& out_idx) {
    if (!rotated_block_indices)
        return false;
    for (size_t i = 0; i < rotated_block_count; ++i)
        if (rotated_block_indices[i] == block_id) {
            out_idx = (int32_t)i;
            return true;
        }
    return false;
}
inline int32_t compute_trig_row_index(int32_t rotated_index,
                                      int32_t token_offset_in_block,
                                      const int32_t* rotation_deltas,
                                      size_t rotation_deltas_dim,
                                      size_t block_size) {
    if (!rotation_deltas)
        return 0;
    if (rotation_deltas_dim == 1)
        return rotation_deltas[rotated_index];
    return rotation_deltas[rotated_index * (int32_t)block_size + token_offset_in_block];
}
}  // namespace pa_rotary

inline int32_t find_first_negative_index(const int32_t* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (data[i] < 0)
            return (int32_t)i;
    return -1;
}
inline size_t resolve_sequence_index_for_token(size_t token_index, const int32_t* subseq_begins, size_t seq_count) {
    if (!subseq_begins || seq_count <= 1)
        return 0;
    for (size_t s = 0; s < seq_count; ++s)
        if (token_index >= (size_t)subseq_begins[s] && token_index < (size_t)subseq_begins[s + 1])
            return s;
    return 0;
}
inline int32_t acquire_or_recycle_block_for_sequence(size_t seq_idx,
                                                     int32_t* block_indices,
                                                     int32_t* block_indices_begins,
                                                     std::vector<int32_t>& seq_block_count,
                                                     size_t max_blocks) {
    const int32_t begin = block_indices_begins[seq_idx];
    const int32_t count = seq_block_count[seq_idx];
    if (count < std::numeric_limits<int32_t>::max()) {
        const int32_t free_block = find_first_negative_index(block_indices, max_blocks);
        if (free_block != -1) {
            seq_block_count[seq_idx] = count + 1;
            return free_block;
        }
    }
    const int32_t oldest = block_indices[begin];
    for (int32_t i = 0; i + 1 < count; ++i)
        block_indices[begin + i] = block_indices[begin + i + 1];
    return oldest;
}

struct paged_attention_kernel_context {
    const void* query{nullptr};
    const void* key{nullptr};
    const void* value{nullptr};
    void* key_cache_base{nullptr};
    void* value_cache_base{nullptr};
    const int32_t* past_lens{nullptr};
    const int32_t* subsequence_begins{nullptr};
    int32_t* block_indices{nullptr};
    int32_t* block_indices_begins{nullptr};
    const void* alibi_slopes{nullptr};
    const int32_t* rotated_block_indices{nullptr};
    const int32_t* rotation_deltas{nullptr};
    const void* rotation_trig_lut{nullptr};
    std::vector<int32_t> sequence_block_count;
    size_t batch_token_count{0}, sequence_count{0};
    size_t head_count{0}, block_size{0}, block_count{0};
    size_t query_head_size{0}, key_head_size{0}, value_head_size{0};
    size_t query_feature_size{0}, key_feature_size{0}, value_feature_size{0};
    int32_t max_context_length{0}, sliding_window{0};
    size_t rotated_block_count{0}, rotation_lut_rows{0}, rotation_deltas_dim{0};
};

struct cache_manager_adapter {
    ov::internal::PagedCacheManager& cm;
    ov::internal::PagedCacheManager::handle_t h;

    explicit cache_manager_adapter(ov::internal::PagedCacheManager& mgr,
                                   ov::internal::PagedCacheManager::handle_t handle)
        : cm(mgr),
          h(handle) {}

    inline void* get_key_cache_base() const {
        return cm.get_cache_blocks().key_base;
    }
    inline void* get_value_cache_base() const {
        return cm.get_cache_blocks().value_base;
    }

    inline const int32_t* get_subsequence_begins_or_null() const {
        auto sv = cm.get_subsequence_begins(h);
        return sv.data;
    }

    struct inferred_layout {
        size_t num_blocks, num_heads, block_size, key_head_size, value_head_size, query_head_size;
    };

    static size_t gcd_size_t(size_t a, size_t b) {
        while (b) {
            size_t t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    inferred_layout infer_layout_from_shapes(const ov::Shape& q, const ov::Shape& k, const ov::Shape& v) const {
        inferred_layout L{};
        const size_t qf = (size_t)q[1], kf = (size_t)k[1], vf = (size_t)v[1];
        size_t g = gcd_size_t(qf, gcd_size_t(kf, vf));
        size_t best_h = 1;
        for (size_t h = 1; h <= g; ++h)
            if ((qf % h) == 0 && (kf % h) == 0 && (vf % h) == 0)
                best_h = h;
        L.num_heads = best_h;
        L.query_head_size = qf / best_h;
        L.key_head_size = kf / best_h;
        L.value_head_size = vf / best_h;
        L.num_blocks = cm.get_num_pages();
        const size_t elem_bytes = (size_t)cm.get_element_type().size();
        const size_t denom = L.num_heads * std::max(L.key_head_size, L.value_head_size) * elem_bytes;
        L.block_size = denom ? (cm.get_page_bytes() / denom) : 0;
        return L;
    }
};

template <typename T>
inline void copy_token_key_value_into_cache(const paged_attention_kernel_context& ctx,
                                            size_t token_index,
                                            size_t sequence_index) {
    const T* ksrc = static_cast<const T*>(ctx.key);
    const T* vsrc = static_cast<const T*>(ctx.value);
    T* kdst = static_cast<T*>(ctx.key_cache_base);
    T* vdst = static_cast<T*>(ctx.value_cache_base);

    const size_t local_index = token_index - (size_t)ctx.subsequence_begins[sequence_index];
    const size_t off_in_block = local_index % ctx.block_size;

    const int32_t bid =
        acquire_or_recycle_block_for_sequence(sequence_index,
                                              ctx.block_indices,
                                              ctx.block_indices_begins,
                                              const_cast<std::vector<int32_t>&>(ctx.sequence_block_count),
                                              ctx.block_count);

    const int32_t begin = ctx.block_indices_begins[sequence_index];
    const int32_t tail = ctx.sequence_block_count[sequence_index] - 1;
    ctx.block_indices[begin + tail] = bid;

    for (size_t h = 0; h < ctx.head_count; ++h) {
        const size_t k_off = ((size_t)bid * ctx.head_count + h) * ctx.block_size + off_in_block;
        std::memcpy(kdst + k_off * ctx.key_head_size,
                    ksrc + token_index * ctx.key_feature_size + h * ctx.key_head_size,
                    ctx.key_head_size * sizeof(T));
        const size_t v_off = ((size_t)bid * ctx.head_count + h) * ctx.block_size + off_in_block;
        std::memcpy(vdst + v_off * ctx.value_head_size,
                    vsrc + token_index * ctx.value_feature_size + h * ctx.value_head_size,
                    ctx.value_head_size * sizeof(T));
    }
}

inline bool try_resolve_cached_block_and_offset(size_t sequence_index,
                                                int32_t key_pos,
                                                const paged_attention_kernel_context& ctx,
                                                int32_t& out_block,
                                                int32_t& out_off) {
    const int32_t begin = ctx.block_indices_begins[sequence_index];
    const int32_t end = ctx.block_indices_begins[sequence_index + 1];
    const int32_t count = end - begin;
    if (key_pos < count) {
        out_block = ctx.block_indices[begin + key_pos];
        out_off = key_pos % (int32_t)ctx.block_size;
        return true;
    }
    return false;
}

template <typename T>
inline T compute_score_against_cached_key(const T* q_vec,
                                          size_t h,
                                          const paged_attention_kernel_context& ctx,
                                          int32_t block_id,
                                          int32_t off) {
    const T* kc = static_cast<const T*>(ctx.key_cache_base);
    const T* k_vec = kc + (((size_t)block_id * ctx.head_count + h) * ctx.block_size + (size_t)off) * ctx.key_head_size;

    T s = pa_math::compute_dot_product(q_vec, k_vec, ctx.key_head_size);

    int32_t rot_idx;
    if (pa_rotary::block_has_rotary(block_id, ctx.rotated_block_indices, ctx.rotated_block_count, rot_idx)) {
        std::vector<T> tmp(k_vec, k_vec + ctx.key_head_size);
        const int32_t trig_row = pa_rotary::compute_trig_row_index(rot_idx,
                                                                   off,
                                                                   ctx.rotation_deltas,
                                                                   ctx.rotation_deltas_dim,
                                                                   ctx.block_size);
        pa_rotary::apply_rotary_embedding_to_vector(tmp.data(),
                                                    ctx.key_head_size,
                                                    static_cast<const T*>(ctx.rotation_trig_lut),
                                                    (size_t)trig_row);
        s = pa_math::compute_dot_product(q_vec, tmp.data(), ctx.key_head_size);
    }
    return s;
}

template <typename T>
inline T compute_score_against_new_key(const T* q_vec,
                                       size_t h,
                                       int32_t abs_token_idx,
                                       const paged_attention_kernel_context& ctx) {
    const T* ksrc = static_cast<const T*>(ctx.key);
    const T* k_vec = ksrc + (size_t)abs_token_idx * ctx.key_feature_size + h * ctx.key_head_size;
    return pa_math::compute_dot_product(q_vec, k_vec, ctx.key_head_size);
}

template <typename T>
inline void accumulate_value_from_cached_key(size_t h,
                                             const paged_attention_kernel_context& ctx,
                                             int32_t block_id,
                                             int32_t off,
                                             T w,
                                             std::vector<T>& out_vec) {
    const T* vc = static_cast<const T*>(ctx.value_cache_base);
    const T* v_vec =
        vc + (((size_t)block_id * ctx.head_count + h) * ctx.block_size + (size_t)off) * ctx.value_head_size;
    for (size_t i = 0; i < ctx.value_head_size; ++i)
        out_vec[i] += w * v_vec[i];
}

template <typename T>
inline void accumulate_value_from_new_key(int32_t abs_token_idx,
                                          size_t h,
                                          const paged_attention_kernel_context& ctx,
                                          T w,
                                          std::vector<T>& out_vec) {
    const T* vsrc = static_cast<const T*>(ctx.value);
    const T* v_vec = vsrc + (size_t)abs_token_idx * ctx.value_feature_size + h * ctx.value_head_size;
    for (size_t i = 0; i < ctx.value_head_size; ++i)
        out_vec[i] += w * v_vec[i];
}

template <typename T>
void paged_attention(T* out,
                     T* out_scores,
                     const T* query,
                     const T* key,
                     const T* value,
                     const int32_t* past_lens,
                     const int32_t* subseq_begins_opt,
                     int32_t* block_indices,
                     int32_t* block_indices_begins,
                     const T* scale_opt,
                     const int32_t* sliding_window_opt,
                     const T* alibi_slopes_opt,
                     const int32_t* max_context_len_opt,
                     const int32_t* rotated_block_indices_opt,
                     const int32_t* rotation_deltas_opt,
                     const T* rotation_trig_lut_opt,
                     const ov::Shape& query_shape,
                     const ov::Shape& key_shape,
                     const ov::Shape& value_shape,
                     const ov::Shape& past_lens_shape,
                     const ov::Shape& rotated_block_indices_shape,
                     const ov::Shape& rotation_deltas_shape,
                     const ov::Shape& rotation_trig_lut_shape,
                     ov::internal::PagedCacheManager::handle_t cache_handle,
                     const std::shared_ptr<ov::internal::PagedCacheManager>& cache_manager) {
    cache_manager_adapter cm(*cache_manager, cache_handle);
    const auto L = cm.infer_layout_from_shapes(query_shape, key_shape, value_shape);

    paged_attention_kernel_context ctx{};
    ctx.query = query;
    ctx.key = key;
    ctx.value = value;
    ctx.key_cache_base = cm.get_key_cache_base();
    ctx.value_cache_base = cm.get_value_cache_base();
    ctx.past_lens = past_lens;
    const int32_t* subseq_from_cm = cm.get_subsequence_begins_or_null();
    ctx.subsequence_begins = subseq_begins_opt ? subseq_begins_opt : subseq_from_cm;
    ctx.block_indices = block_indices;
    ctx.block_indices_begins = block_indices_begins;
    ctx.alibi_slopes = alibi_slopes_opt;
    ctx.rotated_block_indices = rotated_block_indices_opt;
    ctx.rotation_deltas = rotation_deltas_opt;
    ctx.rotation_trig_lut = rotation_trig_lut_opt;

    ctx.batch_token_count = (size_t)query_shape[0];
    ctx.query_feature_size = (size_t)query_shape[1];
    ctx.key_feature_size = (size_t)key_shape[1];
    ctx.value_feature_size = (size_t)value_shape[1];

    ctx.block_count = L.num_blocks;
    ctx.head_count = L.num_heads;
    ctx.block_size = L.block_size;
    ctx.key_head_size = L.key_head_size;
    ctx.value_head_size = L.value_head_size;
    ctx.query_head_size = L.query_head_size ? L.query_head_size : (ctx.query_feature_size / ctx.head_count);

    ctx.sequence_count = (size_t)past_lens_shape[0];
    ctx.rotated_block_count = rotated_block_indices_shape.empty() ? 0 : (size_t)rotated_block_indices_shape[0];
    ctx.rotation_deltas_dim =
        (rotation_deltas_shape.empty() || ctx.rotated_block_count == 0) ? 0 : (size_t)rotation_deltas_shape[1];
    ctx.rotation_lut_rows = rotation_trig_lut_shape.empty() ? 0 : (size_t)rotation_trig_lut_shape[0];

    ctx.max_context_length = max_context_len_opt ? max_context_len_opt[0] : 0;
    ctx.sliding_window = sliding_window_opt ? sliding_window_opt[0] : 0;

    ctx.sequence_block_count.resize(ctx.sequence_count);
    for (size_t s = 0; s < ctx.sequence_count; ++s) {
        const int32_t b = ctx.block_indices_begins[s];
        const int32_t e = ctx.block_indices_begins[s + 1];
        ctx.sequence_block_count[s] = e - b;
    }

    const T scale = scale_opt ? scale_opt[0] : T(1) / std::sqrt((T)ctx.key_head_size);

    for (size_t tok = 0; tok < ctx.batch_token_count; ++tok) {
        const size_t seq = resolve_sequence_index_for_token(tok, ctx.subsequence_begins, ctx.sequence_count);

        if (ctx.subsequence_begins && tok >= (size_t)ctx.subsequence_begins[seq]) {
            copy_token_key_value_into_cache<T>(ctx, tok, seq);
        }

        for (size_t h = 0; h < ctx.head_count; ++h) {
            const T* q_vec = static_cast<const T*>(ctx.query) + tok * ctx.query_feature_size + h * ctx.query_head_size;

            const int32_t past_cnt = ctx.past_lens ? ctx.past_lens[seq] : 0;
            const int32_t new_cnt =
                ctx.subsequence_begins ? (ctx.subsequence_begins[seq + 1] - ctx.subsequence_begins[seq]) : 0;
            const int32_t total_unclamped = past_cnt + new_cnt;
            const int32_t total = (ctx.max_context_length > 0)
                                      ? std::min<int32_t>(total_unclamped, ctx.max_context_length)
                                      : total_unclamped;

            const int32_t keep_from = (ctx.sliding_window > 0) ? std::max<int32_t>(0, total - ctx.sliding_window) : 0;

            std::vector<T> scores((size_t)total, T(0));

            for (int32_t k = 0; k < total; ++k) {
                if (ctx.sliding_window > 0 && k < keep_from) {
                    scores[k] = -std::numeric_limits<T>::infinity();
                    continue;
                }

                T s = T(0);
                if (k < past_cnt) {
                    int32_t bid, off;
                    if (try_resolve_cached_block_and_offset(seq, k, ctx, bid, off)) {
                        s = compute_score_against_cached_key(q_vec, h, ctx, bid, off);
                    }
                } else {
                    const int32_t abs_idx =
                        ctx.subsequence_begins ? (ctx.subsequence_begins[seq] + (k - past_cnt)) : (k - past_cnt);
                    s = compute_score_against_new_key(q_vec, h, abs_idx, ctx);
                }

                const T alibi = ctx.alibi_slopes ? static_cast<const T*>(ctx.alibi_slopes)[h] : T(0);
                scores[k] = s * scale + alibi * T(-(total - k - 1));
            }

            pa_math::compute_softmax_in_place(scores);

            std::vector<T> out_head(ctx.value_head_size, T(0));
            for (int32_t k = 0; k < total; ++k) {
                const T w = scores[k];
                if (k < past_cnt) {
                    int32_t bid, off;
                    if (try_resolve_cached_block_and_offset(seq, k, ctx, bid, off)) {
                        accumulate_value_from_cached_key(h, ctx, bid, off, w, out_head);
                    }
                } else {
                    const int32_t abs_idx =
                        ctx.subsequence_begins ? (ctx.subsequence_begins[seq] + (k - past_cnt)) : (k - past_cnt);
                    accumulate_value_from_new_key<T>(abs_idx, h, ctx, w, out_head);
                }

                const size_t sidx = (tok * ctx.head_count + h) * (size_t)ctx.max_context_length + (size_t)k;
                out_scores[sidx] = scores[k];
            }

            T* dst = out + tok * ctx.value_feature_size + h * ctx.value_head_size;
            std::memcpy(dst, out_head.data(), ctx.value_head_size * sizeof(T));
        }
    }
}

}  // namespace ov::reference
