// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/paged_attn.hpp"

#include <ctime>
#include <memory>
#include <random>
#include <vector>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <cmath>

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {

// Implementation of PagedAttention
class Memory {
public:
    Memory(const Shape& shape, const element::Type& type, char* data)
        : shape_(shape), type_(type), data_(data) {}

    const Shape& getStaticDims() const { return shape_; }
    char* getData() const { return data_; }

private:
    Shape shape_;
    element::Type type_;
    char* data_;
};

template <typename DATA_TYPE, typename KVCACHE_TYPE>
struct AttentionExecutor : public PagedAttentionExecutor {
    MHAHelper<DATA_TYPE, KVCACHE_TYPE> _helper;
    MHA<DATA_TYPE, KVCACHE_TYPE> _kernel;
    PlainTensor _slot_mapping;

    AttentionExecutor() : _kernel(_helper) {}

    void init(const std::vector<Memory*>& inputs,
              const std::vector<Memory*>& outputs,
              PlainTensor& q,
              PlainTensor& k,
              PlainTensor& v,
              PlainTensor& k_cache,
              PlainTensor& v_cache,
              PlainTensor& past_lens,
              PlainTensor& subsequence_begins,
              PlainTensor& block_indices,
              PlainTensor& block_indices_begins,
              float& scale,
              size_t& sliding_window,
              PlainTensor& alibi_slopes,
              size_t& max_context_len,
              PlainTensor& output_emb,
              PlainTensor& output_score) {
        q.reset(inputs[ID_Q]);  // [B_token, H * S]
        k.reset(inputs[ID_K]);
        v.reset(inputs[ID_V]);
        k_cache.reset(inputs[ID_KCACHE]);                             // [NUM_BLOCKS, H, 32, S]
        v_cache.reset(inputs[ID_VCACHE]);                             // [NUM_BLOCKS, H, 32, S]
        past_lens.reset(inputs[ID_PAST_LENS]);                        // [B_seq]
        subsequence_begins.reset(inputs[ID_SUBSEQUENCE_BEGINS]);      // [B_seq+1]
        block_indices.reset(inputs[ID_BLOCK_INDICES]);                // [num_blocks]
        block_indices_begins.reset(inputs[ID_BLOCK_INDICES_BEGINS]);  // [B_seq+1]
        scale = *inputs[ID_SCALE]->getDataAs<float>();
        sliding_window = static_cast<size_t>(*inputs[ID_SLIDING_WINDOW]->getDataAs<int32_t>());
        if (!inputs[ID_ALIBI_SLOPES]->getShape().hasZeroDims())
            alibi_slopes.reset(inputs[ID_ALIBI_SLOPES]);
        max_context_len = static_cast<size_t>(*inputs[ID_MAX_CONTEXT_LEN]->getDataAs<int32_t>());
        output_emb.reset(outputs[0]);
        if (outputs.size() == 2)
            output_score.reset(outputs[1]);

        auto B_token = q.size(0);
        auto Hk = k_cache.size(1);
        auto S = k_cache.size(3) - (k_cache.m_dt == ov::element::Type_t::u8 ? sizeof(float) * 2 : 0);
        auto SV = v_cache.size(3) - (k_cache.m_dt == ov::element::Type_t::u8 ? sizeof(float) * 2 : 0);
        auto block_size = k_cache.size(2);
        auto H = q.size(1) / S;
        auto h_each_group_len = 1;
        if (Hk != H) {
            h_each_group_len = H / Hk;
        }
        auto B_seq = past_lens.size(0);

        q.assert_dims({B_token, H * S});
        k.assert_dims({B_token, Hk * S});
        v.assert_dims({B_token, Hk * SV});
        q = q.reshape({B_token, H, 1, S});
        k = k.reshape({B_token, Hk, 1, S});
        v = v.reshape({B_token, Hk, 1, SV});
        if (k_cache.m_dt == ov::element::Type_t::u8) {
            k_cache.assert_dims({0, Hk, block_size, S + sizeof(float) * 2}, true);
            v_cache.assert_dims({k_cache.m_dims[0], Hk, block_size, SV + sizeof(float) * 2});
        } else {
            k_cache.assert_dims({0, Hk, block_size, S}, true);
            v_cache.assert_dims({k_cache.m_dims[0], Hk, block_size, SV});
        }
        past_lens.assert_dims({B_seq});
        subsequence_begins.assert_dims({B_seq + 1});
        block_indices.assert_dims({0}, true);
        block_indices_begins.assert_dims({B_seq + 1});
        if (scale == 0.0f)
            scale = 1.0f / sqrt(S);
        if (alibi_slopes) {
            alibi_slopes.assert_dims({H});
        }
        output_emb.assert_dims({B_token, H * SV});
        output_emb = output_emb.reshape({B_token, 1, H * SV});

        OPENVINO_ASSERT(block_size == 32, "CPU: block size must be 32, current: ", block_size);

        _helper.init(H, S, SV, Hk, h_each_group_len, block_size, sliding_window, scale, max_context_len, alibi_slopes);
    }

    void concat_pastkv(const PlainTensor& k,
                       const PlainTensor& v,
                       const PlainTensor& k_cache,
                       const PlainTensor& v_cache,
                       const PlainTensor& past_lens,
                       const PlainTensor& subsequence_begins,
                       const PlainTensor& block_indices,
                       const PlainTensor& block_indices_begins) {
        auto B_token = k.size(0);
        _slot_mapping.resize<int32_t>({B_token});

        size_t idx = 0;
        for (size_t i = 0; i < past_lens.size(0); i++) {
            auto q_len = subsequence_begins.ptr<int32_t>()[i + 1] - subsequence_begins.ptr<int32_t>()[i];
            auto kv_len = past_lens.ptr<int32_t>()[i] + q_len;
            auto block_number_start = block_indices_begins.ptr<int32_t>()[i];
            auto block_offset_start = kv_len - q_len;
            for (int32_t j = 0; j < q_len; j++) {
                auto block_offset = block_offset_start + j;
                auto block_number =
                    block_indices.ptr<int32_t>()[block_number_start + block_offset / _helper._block_size];
                _slot_mapping.ptr<int32_t>()[idx++] =
                    block_number * _helper._block_size + block_offset % _helper._block_size;
            }
        }

        if (k_cache.m_dt == ov::element::Type_t::u8) {
            paged_attn_quantkv(k, v, k_cache, v_cache, _slot_mapping);
        } else {
            paged_attn_memcpy(k, v, k_cache, v_cache, _slot_mapping);
        }
    }

    void execute(const std::vector<Memory*>& inputs, const std::vector<Memory*>& outputs) override {
        PlainTensor q, k, v, k_cache, v_cache;
        PlainTensor past_lens, subsequence_begins, block_indices, block_indices_begins;
        float scale;
        size_t sliding_window;
        PlainTensor alibi_slopes;
        size_t max_context_len;
        PlainTensor output_emb;
        PlainTensor output_score;

        init(inputs,
             outputs,
             q,
             k,
             v,
             k_cache,
             v_cache,
             past_lens,
             subsequence_begins,
             block_indices,
             block_indices_begins,
             scale,
             sliding_window,
             alibi_slopes,
             max_context_len,
             output_emb,
             output_score);
        concat_pastkv(k, v, k_cache, v_cache, past_lens, subsequence_begins, block_indices, block_indices_begins);

        _kernel(q,
                k_cache,
                v_cache,
                output_emb,
                output_score,
                max_context_len,
                past_lens,
                subsequence_begins,
                block_indices,
                block_indices_begins,
                alibi_slopes);
    }
};

void paged_attention(const Shape* out_shape,
                     char* out,
                     const Shape& in_shape,
                     char* in,
                     const element::Type& elem_type,
                     const std::vector<Memory*>& inputs,
                     const std::vector<Memory*>& outputs) {
    // Create input and output memory objects
    Memory input_memory(in_shape, elem_type, in);
    Memory output_memory(*out_shape, elem_type, out);

    // Create executor and execute the attention mechanism
    AttentionExecutor<float, float> executor; // Adjust template parameters as needed
    executor.execute(inputs, outputs);
}

void paged_attn_quantkv(const PlainTensor& k,
                        const PlainTensor& v,
                        const PlainTensor& k_cache,
                        const PlainTensor& v_cache,
                        const PlainTensor& slot_mapping) {
    auto B_token = k.size(0);
    auto Hk = k.size(1);
    auto S = k.size(3);
    auto SV = v.size(3);

    for (size_t b = 0; b < B_token; ++b) {
        for (size_t h = 0; h < Hk; ++h) {
            for (size_t s = 0; s < S; ++s) {
                auto slot = slot_mapping.ptr<int32_t>()[b * Hk * S + h * S + s];
                k_cache.ptr<float>()[slot] = k.ptr<float>()[b * Hk * S + h * S + s];
            }
            for (size_t sv = 0; sv < SV; ++sv) {
                auto slot = slot_mapping.ptr<int32_t>()[b * Hk * SV + h * SV + sv];
                v_cache.ptr<float>()[slot] = v.ptr<float>()[b * Hk * SV + h * SV + sv];
            }
        }
    }
}

void paged_attn_memcpy(const PlainTensor& k,
                       const PlainTensor& v,
                       const PlainTensor& k_cache,
                       const PlainTensor& v_cache,
                       const PlainTensor& slot_mapping) {
    auto B_token = k.size(0);
    auto Hk = k.size(1);
    auto S = k.size(3);
    auto SV = v.size(3);

    for (size_t b = 0; b < B_token; ++b) {
        for (size_t h = 0; h < Hk; ++h) {
            for (size_t s = 0; s < S; ++s) {
                auto slot = slot_mapping.ptr<int32_t>()[b * Hk * S + h * S + s];
                std::memcpy(&k_cache.ptr<float>()[slot], &k.ptr<float>()[b * Hk * S + h * S + s], sizeof(float));
            }
            for (size_t sv = 0; sv < SV; ++sv) {
                auto slot = slot_mapping.ptr<int32_t>()[b * Hk * SV + h * SV + sv];
                std::memcpy(&v_cache.ptr<float>()[slot], &v.ptr<float>()[b * Hk * SV + h * SV + sv], sizeof(float));
            }
        }
    }
}

void paged_attention(char* out,
                     const char* query,		 
                     const char* key, 		
                     const char* value, 
                     const char* key_cache,	
                     const char* value_cache,
                     const ov::element::Type dtype,
                     const ov::Shape& qkv_shape,		
                     const ov::Shape& kv_cache_shape,
                     const int32_t* past_lens,
                     const int32_t* subsequence_begins,	
                     const int32_t* block_indices,		
                     const int32_t* block_indices_begins,                        
                     const int32_t scale,					
                     const int32_t sliding_window, 		
                     const int32_t* alibi_slopes,			
                     const int32_t max_context_len) {
    // Assuming qkv_shape is [batch_size, num_heads, seq_len, head_dim]
    int batch_size = qkv_shape[0];
    int num_heads = qkv_shape[1];
    int seq_len = qkv_shape[2];
    int head_dim = qkv_shape[3];

    // Determine the size of each element based on dtype
    size_t element_size = dtype.size();

    // Cast input buffers to the correct data type
    const float* query_f = reinterpret_cast<const float*>(query);
    const float* key_f = reinterpret_cast<const float*>(key);
    const float* value_f = reinterpret_cast<const float*>(value);
    const float* key_cache_f = reinterpret_cast<const float*>(key_cache);
    const float* value_cache_f = reinterpret_cast<const float*>(value_cache);

    // Initialize output buffer
    std::vector<float> output(batch_size * num_heads * seq_len * head_dim, 0.0f);

    // Iterate over each batch and head
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            // Compute attention scores
            for (int i = 0; i < seq_len; ++i) {
                float score = 0.0f;
                for (int j = 0; j < head_dim; ++j) {
                    int query_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + j;
                    int key_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + j;
                    score += query_f[query_idx] * key_f[key_idx];
                }
                score /= std::sqrt(static_cast<float>(head_dim));
                score *= scale;

                // Apply sliding window and ALiBi slopes
                if (sliding_window > 0) {
                    int window_start = std::max(0, i - sliding_window);
                    int window_end = std::min(seq_len, i + sliding_window + 1);
                    for (int k = window_start; k < window_end; ++k) {
                        score += alibi_slopes[k];
                    }
                }

                // Store the computed score in the output buffer
                int output_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim;
                output[output_idx] = score;
            }
        }
    }

    // Copy the output to the out buffer
    std::memcpy(out, output.data(), output.size() * element_size);
}

}  // namespace reference
}  // namespace ov
