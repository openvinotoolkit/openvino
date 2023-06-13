// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "llm_types.hpp"

namespace llmdnn {

// pattern is:
// query:[batch, num_heads, query_seq_len, head_size]  key:[batch, num_heads, key_seq_len, head_size]
//    \                                                 |
//     \                                           Transpose0: [batch, num_heads, head_size, key_seq_len]
//      \                                              /
//       \                                            /
//        \                                          /
//        MatMul0: [batch, num_heads, query_seq_len, key_seq_len]
//          |
//          |   norm_factor(const): [1]
//          |       /
//       Multiply: [batch, num_heads, query_seq_len, key_seq_len]
//          |
//          |   causal_mask: [1, 1, query_seq_len, key_seq_len]
//          |       /
//       Select(only for 1x300): [batch, num_heads, query_seq_len, key_seq_len]
//          |
//          |   attention_mask:[batch, 1, 1, key_seq_len]
//          |       /
//       Add: [batch, num_heads, query_seq_len, key_seq_len]
//          |
//       SoftMax: [batch, num_heads, query_seq_len, key_seq_len]
//          |
//           \  value:[batch, num_heads, key_seq_len, head_size]
//            \     /
//             MatMul1: [batch, num_heads, query_seq_len, head_size]
//               |
//            Transpose1(only for 1x300): [batch, query_seq_len, num_heads * head_size]
class mha_gpt {
public:
    struct create_param {
        size_t num_heads;
        size_t head_size;
        size_t head_size_aligned;       // better to aligned to 64 bytes for best performance, apply for qkv
        size_t max_seq_len;             // max seq length for computing the size of matmul tmp result
        float normal_factor;
        data_type_t qkv_precision;
        data_type_t dst_precision;
    };
    struct exec_param {
        size_t batch;
        size_t query_seq_len;
        size_t key_seq_len;
        uint8_t* q;                         // q buffer, compact, shape: [batch, num_heads, query_seq_len, head_size]
        uint8_t** k;                        // k buffer, k[N] stands different batch which may be discreted
                                            //      k[0] shape: [batch, num_heads, key_seq_len, head_size]
        uint8_t** v;                        // v buffer, v[N] stands different batch which may be discreted
                                            //      v[0] shape: [batch, num_heads, value_seq_len, head_size]
        float** attention_mask;             // attention mask, attention_mask[N] is the batch
                                            //      attention_mask[0] shape: [1, max_seq_len]
        uint8_t* attn_output;               // output, compact, shape: [batch, query_seq_len, num_heads * head_size]
        size_t head_stride_in_kv;           // kv stride for next head; kv may be preallocated a big buffer
        float q_dequant;
        float k_dequant;
        float v_dequant;
        float qk_quant;
        std::vector<float> qkv_quant;       // per channel
        // float* qk_normal_dq;                // per channel, each item = normal_factor * q_dequant * k_dequant, used for softmax input
        // float* qk_quant;                    // per channel, used for softmax output
        // float* qkv_dq_q;                    // per channel, each item = 1 / qk_quant * v_dequant * qkv_quant, used for matmul2 output
    };

    mha_gpt();
    void create(const create_param& param);
    void exec(const exec_param& param);

private:
    struct Impl;
    std::shared_ptr<Impl> _impl;
};

}
