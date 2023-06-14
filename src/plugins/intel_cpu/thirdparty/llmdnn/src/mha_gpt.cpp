// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "simple_parallel.hpp"
#include "utility.hpp"
#include "utility_avx512.hpp"
#include "mm_kernel_amx.hpp"
#include "softmax_kernel_avx512.hpp"
#include "transpose_kernel_avx512.hpp"
#include "mha_gpt.hpp"

using namespace ov::cpu;

namespace llmdnn {

struct mha_gpt::Impl {
    void create(const create_param& param);
    void exec(const exec_param& param);

    create_param _create_param;

    void mha_bf16(const exec_param &param);
    void mha_i8(const exec_param &param);

    size_t bufferMatMul0OutSize;
    size_t bufferMatMul1OutSize;

    std::shared_ptr<uint8_t> bufferMatMul0Out;
    std::shared_ptr<uint8_t> bufferMatMul1Out;

    std::vector<std::shared_ptr<amx_kernel::MatmulVector<ov::bfloat16, ov::bfloat16>>> gemAvB_BF16xBF16;
    std::vector<std::shared_ptr<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>>> qKtrGemm_BF16xBF16;
    std::vector<std::shared_ptr<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>>> qKVGemm_BF16xBF16;

    std::vector<std::shared_ptr<amx_kernel::Matmul<int8_t, int8_t>>> qKtrGemm_i8xi8;
    std::vector<std::shared_ptr<amx_kernel::Matmul<uint8_t, int8_t>>> qKVGemm_u8xi8;
    std::vector<std::shared_ptr<amx_kernel::MatmulVector<int8_t, int8_t>>> gemAvB_i8xi8;
};

void mha_gpt::Impl::create(const create_param& param) {
    _create_param = param;

    // q: [batch, num_heads, query_seq_len, head_size]
    // k: [batch, num_heads, maxSeqLen(valid: key_seq_len), head_size]
    // v: [batch, num_heads, maxSeqLen(valid: value_seq_len), head_size]
    // attention_mask: [batch, 1, 1, maxSeqLen(valid: key_seq_len)]
    // matmul1: [batch, num_heads, query_seq_len, head_size]
    // attn_output: [batch, query_seq_len, num_heads * head_size]
    size_t numThreads = getTotalThreads();
    if (_create_param.qkv_precision == dnnl_s8) {
        qKtrGemm_i8xi8.resize(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            qKtrGemm_i8xi8[i] = std::make_shared<amx_kernel::Matmul<int8_t, int8_t>>(false, true);
        }
        qKVGemm_u8xi8.resize(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            qKVGemm_u8xi8[i] = std::make_shared<amx_kernel::Matmul<uint8_t, int8_t>>(false, false);
        }
        gemAvB_i8xi8.resize(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            gemAvB_i8xi8[i] = std::make_shared<amx_kernel::MatmulVector<int8_t, int8_t>>();
        }
    } else {
        gemAvB_BF16xBF16.resize(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            gemAvB_BF16xBF16[i] = std::make_shared<amx_kernel::MatmulVector<ov::bfloat16, ov::bfloat16>>();
        }
        qKtrGemm_BF16xBF16.resize(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            qKtrGemm_BF16xBF16[i] = std::make_shared<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>>(false, true);
        }
        qKVGemm_BF16xBF16.resize(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            qKVGemm_BF16xBF16[i] = std::make_shared<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>>(false, false);
        }
    }

    bufferMatMul0OutSize = _create_param.max_seq_len * rndup(_create_param.max_seq_len * sizeof(float), 64);
    bufferMatMul1OutSize = _create_param.max_seq_len * _create_param.head_size_aligned * sizeof(float);

    bufferMatMul0Out = std::shared_ptr<uint8_t>(
                            reinterpret_cast<uint8_t*>(aligned_alloc(64, numThreads * bufferMatMul0OutSize)),
                            [](void * p) { ::free(p); });
    bufferMatMul1Out = std::shared_ptr<uint8_t>(
                            reinterpret_cast<uint8_t*>(aligned_alloc(64, numThreads * bufferMatMul1OutSize)),
                            [](void * p) { ::free(p); });
}

void mha_gpt::Impl::mha_bf16(const exec_param &param) {
    uint8_t* pQIn0 = param.q;
    auto& pKIn0 = param.k;
    auto& attn_masks = param.attention_mask;
    auto& pVIn0 = param.v;
    uint8_t* pout = param.attn_output;

    auto outPrcSize = get_precision_size(_create_param.qkv_precision);
    auto& gemAvB_ops = gemAvB_BF16xBF16;
    auto& qKtrGemm_ops = qKtrGemm_BF16xBF16;
    auto& qKVGemm_ops = qKVGemm_BF16xBF16;
    bool is_vector = param.query_seq_len == 1;
    size_t head_stride_in_q = _create_param.head_size_aligned * param.query_seq_len;
    size_t batch_stride_in_q = head_stride_in_q * _create_param.num_heads;
    size_t head_stride_in_attn = _create_param.head_size;
    size_t batch_stride_in_attn = _create_param.head_size * _create_param.num_heads * param.query_seq_len;
    size_t causal_mask_offset_start = param.key_seq_len - param.query_seq_len;

    if (is_vector) {
        parallel_for2d(param.batch, _create_param.num_heads, [&](size_t threadNum, size_t i0, size_t i1) {
            auto pQIn0_aux = pQIn0 + (i0 * batch_stride_in_q + i1 * head_stride_in_q) * get_precision_size(_create_param.qkv_precision);
            auto pKIn0_aux = pKIn0[i0] + i1 * param.head_stride_in_kv * get_precision_size(_create_param.qkv_precision);
            auto pVIn0_aux = pVIn0[i0] + i1 * param.head_stride_in_kv * get_precision_size(_create_param.qkv_precision);

            auto pAddIn1_aux = attn_masks[i0];

            auto bufferMatMul0Out_local = reinterpret_cast<uint8_t*>(bufferMatMul0Out.get() + threadNum * bufferMatMul0OutSize);
            auto bufferMatMul1Out_local = reinterpret_cast<uint8_t*>(bufferMatMul1Out.get() + threadNum * bufferMatMul1OutSize);
            
            tensor2D<ov::bfloat16> matK(param.key_seq_len, _create_param.head_size, reinterpret_cast<ov::bfloat16*>(pKIn0_aux), _create_param.head_size_aligned * sizeof(ov::bfloat16));
            // N: key_seq_len, K: head_size
            // q[1, K] * transpose(k[N, K])        ==>
            //     k[N, K] * transpose(q[1, K])    ==>
            //     k[N, K] * q[K, 1]
            (*gemAvB_ops[threadNum])(matK, reinterpret_cast<ov::bfloat16*>(pQIn0_aux), reinterpret_cast<float*>(bufferMatMul0Out_local));

            float* pMatMul0Out = reinterpret_cast<float*>(bufferMatMul0Out_local);
            mul_add_f32(pMatMul0Out, pMatMul0Out, _create_param.normal_factor, pAddIn1_aux, param.key_seq_len);
            softmax<ov::bfloat16>(reinterpret_cast<ov::bfloat16*>(pMatMul0Out), pMatMul0Out, param.key_seq_len, nullptr, nullptr, nullptr);
            auto pOut_aux = pout + (i0 * batch_stride_in_attn + i1 * head_stride_in_attn) * outPrcSize;
            tensor2D<ov::bfloat16> matQK(param.query_seq_len, param.key_seq_len, reinterpret_cast<ov::bfloat16*>(bufferMatMul0Out_local), rndup(param.key_seq_len * sizeof(ov::bfloat16), 64));
            tensor2D<ov::bfloat16> matV(param.key_seq_len, _create_param.head_size, reinterpret_cast<ov::bfloat16*>(pVIn0_aux), _create_param.head_size_aligned * sizeof(ov::bfloat16));
            tensor2D<float> matQKV(param.query_seq_len, _create_param.head_size, reinterpret_cast<float*>(bufferMatMul1Out_local), _create_param.head_size_aligned * sizeof(float));
            amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(matQKV);
            (*qKVGemm_ops[threadNum])(matQK, matV, 0, _create_param.head_size, pp);
            memcpy2d_stride<ov::bfloat16>(reinterpret_cast<ov::bfloat16*>(pOut_aux), reinterpret_cast<float*>(bufferMatMul1Out_local), param.query_seq_len,
                _create_param.head_size, _create_param.head_size_aligned * sizeof(float), _create_param.num_heads * _create_param.head_size * sizeof(ov::bfloat16), nullptr);
        });
    } else {
        auto numThreads = getTotalThreads();
        int seq_cout_all = rndup(param.query_seq_len, 32) / 32;
        int work_amount = param.batch * _create_param.num_heads * seq_cout_all;
        parallel_for(numThreads, [&](int threadNum) {
            int i0;
            int i1;
            int seq;
            int start {0}, end {0};
            splitter(work_amount, static_cast<int>(numThreads), threadNum, start, end);
            if (start >= work_amount) return;

            parallel_it_init(start, i0, param.batch, i1, _create_param.num_heads, seq, seq_cout_all);
            uint8_t* prev_k = nullptr;
            uint8_t* prev_v = nullptr;
            for (int iwork = start; iwork < end; ++iwork) {
                int seq_start = seq * 32;
                int seq_end = std::min(static_cast<size_t>(seq_start) + 32, param.query_seq_len);
                int seq_cout = seq_end - seq_start;
                // q: [batch, num_heads, query_seq_len, head_size]
                // k: [batch, num_heads, key_seq_len, head_size]
                // v: [batch, num_heads, value_seq_len, head_size]
                auto pQIn0_aux = pQIn0 + (i0 * batch_stride_in_q + i1 * head_stride_in_q + seq_start * _create_param.head_size_aligned) * get_precision_size(_create_param.qkv_precision);
                auto pKIn0_aux = pKIn0[i0] + i1 * param.head_stride_in_kv * get_precision_size(_create_param.qkv_precision);
                auto pVIn0_aux = pVIn0[i0] + i1 * param.head_stride_in_kv * get_precision_size(_create_param.qkv_precision);

                auto pAddIn1_aux = attn_masks[i0];

                auto bufferMatMul0Out_local = reinterpret_cast<uint8_t*>(bufferMatMul0Out.get() + threadNum * bufferMatMul0OutSize);
                auto bufferMatMul1Out_local = reinterpret_cast<uint8_t*>(bufferMatMul1Out.get() + threadNum * bufferMatMul1OutSize);
                
                tensor2D<ov::bfloat16> matQ(seq_cout, _create_param.head_size, reinterpret_cast<ov::bfloat16*>(pQIn0_aux), _create_param.head_size_aligned * sizeof(ov::bfloat16));
                tensor2D<ov::bfloat16> matK(param.key_seq_len, _create_param.head_size, reinterpret_cast<ov::bfloat16*>(pKIn0_aux), _create_param.head_size_aligned * sizeof(ov::bfloat16));
                tensor2D<float> matQK(seq_cout, param.key_seq_len, reinterpret_cast<float*>(bufferMatMul0Out_local), rndup(param.key_seq_len * sizeof(float), 64));
                amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(matQK);
                (*qKtrGemm_ops[threadNum])(matQ, matK, 0, param.key_seq_len, pp, pKIn0_aux == prev_k);
                prev_k = pKIn0_aux;

                auto pMatMul0Out = bufferMatMul0Out_local;
                // loop along K dimension
                size_t valid_softmax_items = causal_mask_offset_start + seq_start + 1;
                for (size_t m = 0; m < seq_cout; m++) {
                    float* src = reinterpret_cast<float*>(pMatMul0Out + m * rndup(param.key_seq_len * sizeof(float), 64));
                    ov::bfloat16* dst = reinterpret_cast<ov::bfloat16*>(pMatMul0Out + m * rndup(param.key_seq_len * sizeof(ov::bfloat16), 64));
                    mul_add_f32(src, src, _create_param.normal_factor, pAddIn1_aux, valid_softmax_items);
                    softmax<ov::bfloat16>(dst, src, valid_softmax_items, nullptr, nullptr, nullptr);
                    // attn_scores = torch.where(causal_mask, attn_scores, mask_value)
                    if (param.key_seq_len > valid_softmax_items) {
                        auto *invalidPtr = dst + valid_softmax_items;
                        memset(invalidPtr, 0, (param.key_seq_len - valid_softmax_items) * get_precision_size(_create_param.qkv_precision));
                        valid_softmax_items = std::min(valid_softmax_items + 1, param.key_seq_len);
                    }
                }
                auto pOut_aux = pout + (i0 * batch_stride_in_attn + i1 * head_stride_in_attn
                    + seq_start * head_stride_in_attn * _create_param.num_heads) * outPrcSize;
                tensor2D<ov::bfloat16> matQKBF16(seq_cout, param.key_seq_len, reinterpret_cast<ov::bfloat16*>(bufferMatMul0Out_local), rndup(param.key_seq_len * sizeof(ov::bfloat16), 64));
                tensor2D<ov::bfloat16> matV(param.key_seq_len, _create_param.head_size, reinterpret_cast<ov::bfloat16*>(pVIn0_aux), _create_param.head_size_aligned * sizeof(ov::bfloat16));
                tensor2D<float> matQKV(seq_cout, _create_param.head_size, reinterpret_cast<float*>(bufferMatMul1Out_local), _create_param.head_size_aligned * sizeof(float));
                amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp2(matQKV);
                (*qKVGemm_ops[threadNum])(matQKBF16, matV, 0, _create_param.head_size, pp2, prev_v == pVIn0_aux);
                prev_v = pVIn0_aux;
                memcpy2d_stride<ov::bfloat16>(reinterpret_cast<ov::bfloat16*>(pOut_aux), reinterpret_cast<float*>(bufferMatMul1Out_local), seq_cout,
                    _create_param.head_size, _create_param.head_size_aligned * sizeof(float), _create_param.num_heads * _create_param.head_size * sizeof(ov::bfloat16), nullptr);
                parallel_it_step(i0, param.batch, i1, _create_param.num_heads, seq, seq_cout_all);
            }
        });
    }
}

void mha_gpt::Impl::mha_i8(const exec_param &param) {
    uint8_t* pQIn0 = param.q;
    auto& pKIn0 = param.k;
    auto& attn_masks = param.attention_mask;
    auto& pVIn0 = param.v;
    uint8_t* pout = param.attn_output;

    auto outPrcSize = get_precision_size(_create_param.dst_precision);
    auto& gemAvB_ops = gemAvB_i8xi8;
    auto& qKtrGemm_ops = qKtrGemm_i8xi8;
    auto& qKVGemm_ops = qKVGemm_u8xi8;
    bool is_vector = param.query_seq_len == 1;
    // dequant param
    auto mul_scales = _create_param.normal_factor * param.q_dequant * param.k_dequant;
    // prepare for per channel
    auto qkv_quant = param.qkv_quant;
    std::vector<float> qk_quant_vec(_create_param.head_size, param.qk_quant);
    for (size_t i = 0; i < param.qkv_quant.size(); i++) {
        qkv_quant[i] *= param.v_dequant / param.qk_quant;
    }
    size_t head_stride_in_q = _create_param.head_size_aligned * param.query_seq_len;
    size_t batch_stride_in_q = head_stride_in_q * _create_param.num_heads;
    size_t head_stride_in_attn = _create_param.head_size;
    size_t batch_stride_in_attn = _create_param.head_size * _create_param.num_heads * param.query_seq_len;
    size_t causal_mask_offset_start = param.key_seq_len - param.query_seq_len;

    if (is_vector) {
        parallel_for2d(param.batch, _create_param.num_heads, [&](size_t threadNum, size_t i0, size_t i1) {
            auto pQIn0_aux = pQIn0 + (i0 * batch_stride_in_q + i1 * head_stride_in_q) * get_precision_size(_create_param.qkv_precision);
            auto pKIn0_aux = pKIn0[i0] + i1 * param.head_stride_in_kv * get_precision_size(_create_param.qkv_precision);
            auto pVIn0_aux = pVIn0[i0] + i1 * param.head_stride_in_kv * get_precision_size(_create_param.qkv_precision);

            auto pAddIn1_aux = attn_masks[i0];

            auto bufferMatMul0Out_local = reinterpret_cast<uint8_t*>(bufferMatMul0Out.get() + threadNum * bufferMatMul0OutSize);
            auto bufferMatMul1Out_local = reinterpret_cast<uint8_t*>(bufferMatMul1Out.get() + threadNum * bufferMatMul1OutSize);
            
            tensor2D<int8_t> matK(param.key_seq_len, _create_param.head_size, reinterpret_cast<int8_t*>(pKIn0_aux), _create_param.head_size_aligned * sizeof(int8_t));
            // N: key_seq_len, K: head_size
            // q[1, K] * transpose(k[N, K])        ==>
            //     k[N, K] * transpose(q[1, K])    ==>
            //     k[N, K] * q[K, 1]
            (*gemAvB_ops[threadNum])(matK, reinterpret_cast<int8_t*>(pQIn0_aux), reinterpret_cast<int32_t*>(bufferMatMul0Out_local));
            cvt_i32_f32(reinterpret_cast<float*>(bufferMatMul0Out_local), reinterpret_cast<int32_t*>(bufferMatMul0Out_local), param.key_seq_len);

            float* pMatMul0Out = reinterpret_cast<float*>(bufferMatMul0Out_local);
            mul_add_f32(pMatMul0Out, pMatMul0Out, mul_scales, pAddIn1_aux, param.key_seq_len);
            softmax<uint8_t>(reinterpret_cast<uint8_t*>(pMatMul0Out), pMatMul0Out, param.key_seq_len, nullptr, nullptr, qk_quant_vec.data());
            auto pOut_aux = pout + (i0 * batch_stride_in_attn + i1 * head_stride_in_attn) * outPrcSize;
            tensor2D<uint8_t> matQK(param.query_seq_len, param.key_seq_len, reinterpret_cast<uint8_t*>(bufferMatMul0Out_local), rndup(param.key_seq_len * sizeof(uint8_t), 64));
            tensor2D<int8_t> matV(param.key_seq_len, _create_param.head_size, reinterpret_cast<int8_t*>(pVIn0_aux), _create_param.head_size_aligned * sizeof(int8_t));
            tensor2D<float> matQKV(param.query_seq_len, _create_param.head_size, reinterpret_cast<float*>(bufferMatMul1Out_local), _create_param.head_size_aligned * sizeof(float));
            amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(matQKV);
            (*qKVGemm_ops[threadNum])(matQK, matV, 0, _create_param.head_size, pp);
            memcpy2d_stride<int8_t>(reinterpret_cast<int8_t*>(pOut_aux), reinterpret_cast<float*>(bufferMatMul1Out_local), param.query_seq_len,
                _create_param.head_size, _create_param.head_size_aligned * sizeof(float), _create_param.num_heads * _create_param.head_size, qkv_quant.data());
        });
    } else {
        auto numThreads = getTotalThreads();
        int seq_cout_all = rndup(param.query_seq_len, 32) / 32;
        int work_amount = param.batch * _create_param.num_heads * seq_cout_all;
        parallel_for(numThreads, [&](int threadNum) {
            int i0;
            int i1;
            int seq;
            int start {0}, end {0};
            splitter(work_amount, static_cast<int>(numThreads), threadNum, start, end);
            if (start >= work_amount) return;

            parallel_it_init(start, i0, param.batch, i1, _create_param.num_heads, seq, seq_cout_all);
            uint8_t* prev_k = nullptr;
            uint8_t* prev_v = nullptr;
            for (int iwork = start; iwork < end; ++iwork) {
                int seq_start = seq * 32;
                int seq_end = std::min(static_cast<size_t>(seq_start) + 32, param.query_seq_len);
                int seq_cout = seq_end - seq_start;
                // q: [batch, num_heads, query_seq_len, head_size]
                // k: [batch, num_heads, key_seq_len, head_size]
                // v: [batch, num_heads, value_seq_len, head_size]
                auto pQIn0_aux = pQIn0 + (i0 * batch_stride_in_q + i1 * head_stride_in_q + seq_start * _create_param.head_size_aligned);
                auto pKIn0_aux = pKIn0[i0] + i1 * param.head_stride_in_kv;
                auto pVIn0_aux = pVIn0[i0] + i1 * param.head_stride_in_kv;

                auto pAddIn1_aux = attn_masks[i0];

                auto bufferMatMul0Out_local = reinterpret_cast<uint8_t*>(bufferMatMul0Out.get() + threadNum * bufferMatMul0OutSize);
                auto bufferMatMul1Out_local = reinterpret_cast<uint8_t*>(bufferMatMul1Out.get() + threadNum * bufferMatMul1OutSize);
                
                tensor2D<int8_t> matQ(seq_cout, _create_param.head_size, reinterpret_cast<int8_t*>(pQIn0_aux), _create_param.head_size_aligned * sizeof(int8_t));
                tensor2D<int8_t> matK(param.key_seq_len, _create_param.head_size, reinterpret_cast<int8_t*>(pKIn0_aux), _create_param.head_size_aligned * sizeof(int8_t));
                tensor2D<float> matQK(seq_cout, param.key_seq_len, reinterpret_cast<float*>(bufferMatMul0Out_local), rndup(param.key_seq_len * sizeof(float), 64));
                amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(matQK);
                (*qKtrGemm_ops[threadNum])(matQ, matK, 0, param.key_seq_len, pp, prev_k == pKIn0_aux);
                prev_k = pKIn0_aux;

                auto pMatMul0Out = bufferMatMul0Out_local;
                // loop along K dimension
                size_t valid_softmax_items = causal_mask_offset_start + seq_start + 1;
                for (size_t m = 0; m < seq_cout; m++) {
                    float* src = reinterpret_cast<float*>(pMatMul0Out + m * rndup(param.key_seq_len * sizeof(float), 64));
                    uint8_t* dst = reinterpret_cast<uint8_t*>(pMatMul0Out + m * rndup(param.key_seq_len * sizeof(uint8_t), 64));
                    mul_add_f32(src, src, mul_scales, pAddIn1_aux, valid_softmax_items);
                    softmax<uint8_t>(dst, src, valid_softmax_items, nullptr, nullptr, qk_quant_vec.data());
                    // attn_scores = torch.where(causal_mask, attn_scores, mask_value)
                    if (param.key_seq_len > valid_softmax_items) {
                        auto *invalidPtr = dst + valid_softmax_items;
                        memset(invalidPtr, 0, (param.key_seq_len - valid_softmax_items) * get_precision_size(_create_param.qkv_precision));
                        valid_softmax_items = std::min(valid_softmax_items + 1, param.key_seq_len);
                    }
                }
                auto pOut_aux = pout + (i0 * batch_stride_in_attn + i1 * head_stride_in_attn
                    + seq_start * head_stride_in_attn * _create_param.num_heads) * outPrcSize;
                tensor2D<uint8_t> matQKI8(seq_cout, param.key_seq_len, reinterpret_cast<uint8_t*>(bufferMatMul0Out_local), rndup(param.key_seq_len * sizeof(uint8_t), 64));
                tensor2D<int8_t> matV(param.key_seq_len, _create_param.head_size, reinterpret_cast<int8_t*>(pVIn0_aux), _create_param.head_size_aligned * sizeof(int8_t));
                tensor2D<float> matQKV(seq_cout, _create_param.head_size, reinterpret_cast<float*>(bufferMatMul1Out_local), _create_param.head_size_aligned * sizeof(float));
                amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp2(matQKV);
                (*qKVGemm_ops[threadNum])(matQKI8, matV, 0, _create_param.head_size, pp2, prev_v == pVIn0_aux);
                prev_v = pVIn0_aux;
                // matmul1: [batch, num_heads, query_seq_len, head_size]
                // attn_output: [batch, query_seq_len, num_heads * head_size]
                memcpy2d_stride<int8_t>(reinterpret_cast<int8_t*>(pOut_aux), reinterpret_cast<float*>(bufferMatMul1Out_local), seq_cout,
                    _create_param.head_size, _create_param.head_size_aligned * sizeof(float), _create_param.num_heads * _create_param.head_size, qkv_quant.data());
                parallel_it_step(i0, param.batch, i1, _create_param.num_heads, seq, seq_cout_all);
            }
        });
    }
}

void mha_gpt::Impl::exec(const exec_param& param) {
    if (_create_param.qkv_precision == dnnl_f32) {
        assert(false);
    } else if (_create_param.qkv_precision == dnnl_bf16) {
        mha_bf16(param);
    } else if (_create_param.qkv_precision == dnnl_s8) {
        mha_i8(param);
    } else {
        assert(false && "doesn't support provided input precisions");
    }
}

// interface
mha_gpt::mha_gpt(): _impl(std::make_shared<Impl>()) {
}

void mha_gpt::create(const create_param& param) {
    _impl->create(param);
}

void mha_gpt::exec(const exec_param& param) {
    _impl->exec(param);
}

}