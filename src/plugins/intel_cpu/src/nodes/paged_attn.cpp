// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attn.h"

#include "common/arbitrary_order_desc_creator.h"
#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/util/common_util.hpp"
#include "shape_inference/custom/paged_attn.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

#ifdef OV_CPU_WITH_MLAS
#    include "mlas/sgemm.hpp"
#endif

#include "utils/plain_tensor.hpp"
#include "kernels/scaled_attn/softmax.hpp"
#include "kernels/scaled_attn/mha_single_token_pa.hpp"
#include "kernels/scaled_attn/attn_memcpy.hpp"
#include "kernels/scaled_attn/attn_quant.hpp"
#include "kernels/x64/brgemm_kernel.hpp"
#include "nodes/common/cpu_convert.h"

#include <algorithm>
#include <string>
#include <vector>
#include "utils/profiler.hpp"

using namespace ov::Extensions::Cpu::XARCH;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
namespace node {

struct PagedAttentionKey {
    ov::element::Type rtPrecision;

    size_t hash() const;
    bool operator==(const PagedAttentionKey& rhs) const;
};

size_t PagedAttentionKey::hash() const {
    size_t seed = 0;
    seed = hash_combine(seed, rtPrecision.hash());

    return seed;
}

bool PagedAttentionKey::operator==(const PagedAttentionKey& rhs) const {
    auto retVal = rtPrecision == rhs.rtPrecision;

    return retVal;
}

#ifdef OPENVINO_ARCH_X86_64

// w = query * Key
//
// query: [1,      head_size]
// Key  : [block_size, head_size]
// w    : [1, block_size]
//
// head_size is known at compile time
struct TileConfig {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
    void reset(int palette, int _startRow, const std::vector<std::pair<int, int>>& _rows_columnsBytes) {
        palette_id = palette;
        startRow = _startRow;
        unsigned long i;
        for (i = 0; i < 14; i++) {
            reserved[i] = 0;
        }
        for (i = 0; i < _rows_columnsBytes.size(); i++) {
            rows[i] = _rows_columnsBytes[i].first;
            cols[i] = _rows_columnsBytes[i].second;
        }
        for (; i < 16; i++) {
            cols[i] = 0;
            rows[i] = 0;
        }
    }
};

class TileConfiger : public jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(TileConfiger)
    TileConfiger() : jit_generator(jit_name()) {
        create_kernel();
    }
    void generate() override {
        Xbyak::Label release;
        test(abi_param1, abi_param1);
        jz(release);
        ldtilecfg(ptr[abi_param1]);
        ret();
        L(release);
        tilerelease();
        ret();
    }
};

class JitMatMulVecAMX : public jit_generator {
    void operator=(const JitMatMulVecAMX&);

public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(JitMatMulVecAMX)
    int m_head_size;
    int m_block_size;
    TileConfiger m_tile_configer;
    TileConfig m_tile_cfg;
    JitMatMulVecAMX(int head_size, int block_size) : jit_generator(jit_name()), m_head_size(head_size), m_block_size(block_size) {
        create_kernel();
        m_tile_cfg.reset(1,
                         0,
                         {
                             {16, 4},   // C:0   M x 1     (4b)
                             {16, 64},  // A:1   M x 32/64 (64b)
                             {16, 4},   // B:2   32/64 x 1 (4b)
                             {16, 4},   // B:3
                             {16, 4},   // B:4
                             {16, 4},   // B:5
                             {16, 4},   // B:6
                             {16, 4},   // B:7
                         });
    }

    void tile_config() {
        m_tile_configer(&m_tile_cfg);
    }
    void tile_release() {
        m_tile_configer(nullptr);
    }

    // to save push/pop: do not use `abi_save_gpr_regs`
    Xbyak::Reg64 reg_q_addr = abi_param1;
    Xbyak::Reg64 reg_k_addr = abi_param2;
    Xbyak::Reg64 reg_dst_addr = abi_param3;
    Xbyak::Reg64 reg_stride_A = rax;
    Xbyak::Reg64 reg_stride_BC = r9;

    Xbyak::Tmm tmmC = tmm0;
    Xbyak::Tmm tmmA = tmm1;
    Xbyak::Tmm tmmB0 = tmm2;
    Xbyak::Tmm tmmB1 = tmm3;
    Xbyak::Tmm tmmB2 = tmm4;
    Xbyak::Tmm tmmB3 = tmm5;
    Xbyak::Tmm tmmB4 = tmm6;
    Xbyak::Tmm tmmB5 = tmm7;

    void generate() override {
        mov(reg_stride_A, m_head_size * 2);
        mov(reg_stride_BC, 4);
        const int kStep = 32;
        if ((m_head_size % 32) != 0)
            throw std::runtime_error("head size is not multiple of 32");
        if ((m_block_size % 16) != 0)
            throw std::runtime_error("block size is not multiple of 16");
        auto num_B_tiles = m_head_size / kStep;
        if (num_B_tiles > 6)
            throw std::runtime_error("number of B tiles is bigger than 6");

        /*
                                    B(query)    head_size x 1
        A(key) matrix : block_size x head_size  C(dst) block_size x 1
        */
        // load query into B tiles
        for (int i = 0; i < num_B_tiles; i++) {
            tileloadd(Xbyak::Tmm(tmmB0.getIdx() + i), ptr[reg_q_addr + reg_stride_BC + i * 64]);
        }

        for (int m = 0; m < m_block_size; m += 16) {
            tilezero(tmmC);
            for (int i = 0; i < num_B_tiles; i++) {
                tileloadd(tmmA, ptr[reg_k_addr + reg_stride_A + i * 64]);
                tdpbf16ps(tmmC, tmmA, Xbyak::Tmm(tmmB0.getIdx() + i));
            }
            tilestored(ptr[reg_dst_addr + reg_stride_BC + m * sizeof(float)], tmmC);
            add(reg_k_addr, m_head_size * 2 * 16);
        }
        ret();
    }
};

#endif

// default implementation: reference
template <PagedAttention::KernelTypes KType, typename T>
struct MHAKernel {
    const GraphContext::CPtr context;
    MHAKernel() = delete;
    explicit MHAKernel(GraphContext::CPtr ctx) : context(ctx) {}

    template <typename D>
    float dot_product(const D* a, const D* b, int len, int stride_b = 1) {
        float result = 0;
        if (stride_b == 1) {
            for (int i = 0; i < len; i++)
                result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        } else {
            for (int i = 0; i < len; i++)
                result += static_cast<float>(a[i]) * static_cast<float>(b[i * stride_b]);
        }
        return result;
    }

    void softmax(float* a, int len) {
        float max = *std::max_element(a, a + len);
        float sum = 0.0f;
        for (int i = 0; i < len; i++) {
            a[i] = exp(a[i] - max);
            sum += a[i];
        }
        float scale = 1.0f / sum;
        for (int i = 0; i < len; i++) {
            a[i] *= scale;
        }
    }

    template <typename D>
    void accumulate(float* acc, const D* v, int len, float weight = 1.0f) {
        for (int i = 0; i < len; i++) {
            acc[i] += static_cast<float>(v[i]) * weight;
        }
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // output_emb    [B, q_len, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    PlainTensor& output_emb,
                    float d_scale = 0.0f,
                    size_t sliding_window = 0) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        auto Hk = present_key.size(1);
        size_t h_each_group_len = H / Hk;
        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        auto k_stride_s = present_key.stride(3);

        parallel_for2d(B, H, [&](size_t b, size_t h) {
            std::vector<float> attn_score(kv_len);
            std::vector<float> word_vec(head_size, 0.0f);

            for (size_t m = 0; m < q_len; m++) {
                // dot-product to get attention scores
                auto* q = query.ptr<T>(b, h, m, 0);
                // how many key/values can be accessed causally
                // no causall mask is set and it's not fused into attention_mask
                auto ncausal = kv_len - q_len + m + 1;
                for (size_t n = 0; n < ncausal; n++) {
                    auto* k = &present_key.at<T>({b, h / h_each_group_len, n, 0}, true);
                    attn_score[n] = dot_product(q, k, head_size, k_stride_s) * d_scale;
                }

                // softmax
                if (sliding_window) {
                    size_t start_idx = 0;
                    auto new_causal = ncausal;
                    if (ncausal > sliding_window) {
                        start_idx = ncausal - static_cast<size_t>(sliding_window);
                        new_causal = sliding_window;
                    }
                    softmax(&attn_score[start_idx], new_causal);
                    memset(&attn_score[0], 0, sizeof(float) * start_idx);
                } else {
                    softmax(&attn_score[0], ncausal);
                }

                // linearly combine value
                word_vec.assign(head_size, 0.0f);
                for (size_t n = 0; n < ncausal; n++) {
                    auto* v = &present_value.at<T>({b, h / h_each_group_len, n, 0}, true);
                    accumulate(word_vec.data(), v, head_size, attn_score[n]);
                }

                // output [B, L1, H*head_size]
                auto* out = &output_emb.at<T>({b, m, h * head_size});
                std::copy(word_vec.begin(), word_vec.end(), out);
            }
        });
    }
};

template <typename T>
struct MHAKernel<PagedAttention::KT_ONEDNN, T> {
    // q: [B, H, q_len, S]
    // k: [B, H, kv_len, S]
    // v: [B, H, kv_len, S]
    const GraphContext::CPtr context;
    dnnl::memory::desc q_md;
    dnnl::memory::desc k_md;
    dnnl::memory::desc weight_md;
    dnnl::memory::desc v_md;
    dnnl::memory::desc out_md;
    dnnl::memory attn_score;
    dnnl::memory attn_weight;
    PlainTensor fp32_out;
    PlainTensor qk_scratch_a;
    PlainTensor qk_scratch_b;
    PlainTensor wv_scratch_a;
    PlainTensor wv_scratch_b;
    std::vector<size_t> wsp;
    size_t wsp_size_per_thread = 0;
    dnnl::matmul qk_prim;
    dnnl::matmul wv_prim;
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    struct brgemmKey {
        size_t M;
        size_t N;
        size_t K;
        size_t lda;
        size_t ldb;
        size_t ldc;
        bool b_transposed;
        ov::element::Type in_type;
        size_t hash() const {
            using namespace dnnl::impl;
            using namespace dnnl::impl::primitive_hashing;
            size_t seed = 0;
            seed = hash_combine(seed, M);
            seed = hash_combine(seed, N);
            seed = hash_combine(seed, K);
            seed = hash_combine(seed, lda);
            seed = hash_combine(seed, ldb);
            seed = hash_combine(seed, ldc);
            seed = hash_combine(seed, b_transposed);
            seed = hash_combine(seed, in_type.hash());
            return seed;
        }
        bool operator==(const brgemmKey& rhs) const {
            return (rhs.M == M) && (rhs.N == N) && (rhs.K == K) && (rhs.lda == lda) && (rhs.ldb == ldb) &&
                   (rhs.ldc == ldc) && (rhs.b_transposed == b_transposed) && (rhs.in_type == in_type);
        }
    };

    std::shared_ptr<BrgemmKernel> qk_gemm_ptr = nullptr;
    std::shared_ptr<BrgemmKernel> wv_gemm_ptr = nullptr;

    MHAKernel() = delete;
    explicit MHAKernel(GraphContext::CPtr ctx)
        : context(ctx) {}

    dnnl::memory::dims make_dnnl_dims(const std::vector<size_t>& dims) {
        dnnl::memory::dims dnnl_dims(dims.size());
        for (size_t i = 0; i < dims.size(); i++)
            dnnl_dims[i] = static_cast<dnnl::memory::dim>(dims[i]);
        return dnnl_dims;
    }

    void prepare_brgemm_prim(dnnl::stream strm, PlainTensor& query, PlainTensor& present_key) {
        auto in_type = precision_of<T>::value;
        auto qkv_dt = in_type == ov::element::f32 ? dt::f32 : dt::bf16;
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        auto Hk = present_key.size(1);
        brgemmKey qk_key = {q_len, kv_len, head_size, query.stride(2), present_key.stride(2), kv_len, true, in_type};

        auto builder = [](const brgemmKey& key) -> std::shared_ptr<BrgemmKernel> {
            return std::make_shared<BrgemmKernel>(key.M,
                                                  key.N,
                                                  key.K,
                                                  key.lda,
                                                  key.ldb,
                                                  key.ldc,
                                                  key.b_transposed,
                                                  key.in_type);
        };

        auto cache = this->context->getParamsCache();
        auto qk_result = cache->getOrCreate(qk_key, builder);
        if (!qk_result.first) {
            OPENVINO_THROW("PagedAttention 1st token qk gemm creation fails");
        }

        qk_gemm_ptr = qk_result.first;
        dnnl::memory::desc attn_md(make_dnnl_dims({B, H, q_len, kv_len}), dt::f32, tag::abcd);
        weight_md = dnnl::memory::desc(make_dnnl_dims({B, H, q_len, kv_len}), qkv_dt, tag::abcd);
        out_md = dnnl::memory::desc(make_dnnl_dims({B, q_len, H, head_size}), qkv_dt, tag::abcd);

        size_t ldc_index = 1;
        brgemmKey wv_key = {q_len,
                            head_size,
                            kv_len,
                            kv_len,
                            present_key.stride(2),
                            static_cast<size_t>(out_md.get_strides()[ldc_index]),
                            false,
                            in_type};

        auto wv_result = cache->getOrCreate(wv_key, builder);
        if (!wv_result.first) {
            OPENVINO_THROW("PagedAttention 1st token wv gemm creation fails");
        }

        wv_gemm_ptr = wv_result.first;

        size_t nthr = static_cast<size_t>(parallel_get_max_threads());

        // wsp is used to compute beta when K is blocked
        wsp_size_per_thread = wv_gemm_ptr->get_wsp_size();
        wsp.resize(nthr * wsp_size_per_thread);

        // allocate scratch a/b, notice get_scratch_a_size/get_scratch_b_size returns in bytes
        size_t data_size = sizeof(T);
        qk_scratch_a.resize<T>({nthr, qk_gemm_ptr->get_scratch_a_size() / data_size});
        wv_scratch_a.resize<T>({nthr, wv_gemm_ptr->get_scratch_a_size() / data_size});

        qk_scratch_b.resize<T>({B, Hk, qk_gemm_ptr->get_scratch_b_size() / data_size});
        wv_scratch_b.resize<T>({B, Hk, wv_gemm_ptr->get_scratch_b_size() / data_size});
        if (!attn_score || attn_md.get_size() > attn_score.get_desc().get_size()) {
            attn_score = dnnl::memory(attn_md, strm.get_engine());
            attn_weight = dnnl::memory(weight_md, strm.get_engine());
        }
        fp32_out.resize<float>({B, q_len, H, head_size});
        return;
    }

    void execute_brgemm(PlainTensor& query,
                        PlainTensor& present_key,
                        PlainTensor& present_value,
                        PlainTensor& output_emb,
                        float d_scale = 0.0f,
                        size_t sliding_window = 0) {
        const auto B = query.size(0);
        const auto H = query.size(1);
        const auto q_len = query.size(2);
        const auto head_size = query.size(3);
        const auto Hk = present_key.size(1);
        const auto kv_len = present_key.size(2);
        size_t h_each_group_len = H / Hk;
        PlainTensor score, weight;
        score.resize({B, H, q_len, kv_len}, static_cast<float*>(attn_score.get_data_handle()));
        weight.resize({B, H, q_len, kv_len}, static_cast<T*>(attn_weight.get_data_handle()));
        const size_t m_block_size = qk_gemm_ptr->get_mblk_size();
        auto m_blocks = (q_len + m_block_size - 1) / m_block_size;
        bool is_bf16 = precision_of<T>::value == ov::element::bf16;
        PROFILE(_attn, "brgemm_pack");
        // packed k, v
        parallel_for2d(B, Hk, [&](size_t b, size_t h) {
            T* k_ptr = &present_key.at<T>({b, h, 0, 0});
            T* v_ptr = &present_value.at<T>({b, h, 0, 0});
            qk_gemm_ptr->copy_buffer_b(k_ptr, &qk_scratch_b.at<T>({b, h, 0}));
            if (is_bf16)
                wv_gemm_ptr->copy_buffer_b(v_ptr, &wv_scratch_b.at<T>({b, h, 0}));
        });

        _attn = ov::intel_cpu::profilerManagerInstance.startProfile("brgemm_attn");
        // attention
        parallel_for3d(B, H, m_blocks, [&](size_t b, size_t h, size_t m_blk) {
            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_len);
            auto m_cnt = m_end - m_start;
            size_t tid = parallel_get_thread_num();
            T* q_ptr = &query.at<T>({b, h, m_start, 0});
            float* c_ptr = &score.at<float>({b, h, m_start, 0});
            T* k_ptr = &qk_scratch_b.at<T>({b, h / h_each_group_len, 0});
            qk_gemm_ptr->executeGemm(m_cnt < m_block_size,
                                     q_ptr,
                                     k_ptr,
                                     c_ptr,
                                     wsp.data() + tid * wsp_size_per_thread,
                                     qk_scratch_a ? &qk_scratch_a.at<T>({tid, 0}) : nullptr);

            for (size_t m = m_start; m < m_end; m++) {
                // apply attention mask & sofmax
                auto ncausal = (kv_len - q_len + m + 1);
                if (sliding_window) {
                    size_t start_idx = 0;
                    auto new_causal = ncausal;
                    if (ncausal > sliding_window) {
                        start_idx = ncausal - static_cast<size_t>(sliding_window);
                        new_causal = sliding_window;
                    }
                    attn_softmax(&score.at<float>({b, h, m, start_idx}),
                                &weight.at<T>({b, h, m, start_idx}),
                                d_scale,
                                nullptr,
                                nullptr,
                                nullptr,
                                false,
                                new_causal,
                                kv_len - start_idx,
                                precision_of<T>::value,
                                precision_of<T>::value);

                    memset(&weight.at<T>({b, h, m, 0}), 0, sizeof(T) * start_idx);
                } else {
                    attn_softmax(&score.at<float>({b, h, m, 0}),
                                &weight.at<T>({b, h, m, 0}),
                                d_scale,
                                nullptr,
                                nullptr,
                                nullptr,
                                false,
                                ncausal,
                                kv_len,
                                precision_of<T>::value,
                                precision_of<T>::value);
                }
            }
            T* w_ptr = &weight.at<T>({b, h, m_start, 0});
            float* fp32_out_ptr;
            if (is_bf16) {
                fp32_out_ptr = &fp32_out.at<float>({b, m_start, h, 0});
            } else {
                fp32_out_ptr = &output_emb.at<float>({b, m_start, h * head_size});
            }
            T* v_ptr = is_bf16 ? &wv_scratch_b.at<T>({b, h / h_each_group_len, 0})
                               : &present_value.at<T>({b, h / h_each_group_len, 0, 0});
            wv_gemm_ptr->executeGemm(m_cnt < m_block_size,
                                     w_ptr,
                                     v_ptr,
                                     fp32_out_ptr,
                                     wsp.data() + tid * wsp_size_per_thread,
                                     wv_scratch_a ? &wv_scratch_a.at<T>({tid, 0}) : nullptr);
            if (is_bf16) {
                attn_memcpy2d_kernel(&fp32_out.at<float>({b, m_start, h, 0}),
                                        &output_emb.at<T>({b, m_start, h * head_size}),
                                        ov::element::f32,
                                        ov::element::bf16,
                                        fp32_out.stride(1),
                                        output_emb.stride(1),
                                        head_size,
                                        m_cnt);
            }
        });
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi          [B, H, q_len, kv_len]
    // output_emb    [B, L1, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    PlainTensor& output_emb,
                    float d_scale = 0.0f,
                    size_t sliding_window = 0) {
        PROFILE(_attn, "prepare_prim");
        auto head_size = query.size(3);
        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        prepare_brgemm_prim(strm, query, present_key);
        _attn = ov::intel_cpu::profilerManagerInstance.startProfile("exec_qk");
        execute_brgemm(query,
                       present_key,
                       present_value,
                       output_emb,
                       d_scale,
                       sliding_window);
    }
};

#ifdef OV_CPU_WITH_MLAS
template <>
struct MHAKernel<PagedAttention::KT_MLAS, float> {
    const GraphContext::CPtr context;
    size_t m_block_size;
    // buffer to hold qk temp
    std::vector<PlainTensor> qk_buffers;

    MHAKernel() = delete;
    explicit MHAKernel(GraphContext::CPtr ctx): context(ctx) {
        m_block_size = 4;
        qk_buffers.resize(parallel_get_max_threads());
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi
    // output_emb    [B, L1, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    PlainTensor& output_emb,
                    float d_scale = 0.0f,
                    size_t sliding_window = 0) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        auto h_group_num = present_key.size(1);
        size_t h_each_group_len = H / h_group_num;

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);
        auto k_stride_s = present_key.stride(3);

        auto m_blocks = (q_len + m_block_size - 1) / m_block_size;

        parallel_for3d(B, H, m_blocks, [&](size_t b, size_t h, size_t m_blk) {
            auto thread_id = parallel_get_thread_num();
            if (thread_id < 0)
                OPENVINO_THROW("The calling thread isn't initialized!");
            auto& qk_buf = qk_buffers[thread_id];

            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_len);
            auto m_cnt = m_end - m_start;

            auto kv_len_cache_align = (((kv_len * sizeof(float)) + 63) / 64 * 64) / sizeof(float);
            qk_buf.resize<float>({m_block_size, kv_len_cache_align});
            const float* q_ptr = &query.at<float>({b, h, m_start, 0});
            const float* k_ptr = &present_key.at<float>({b, h / h_each_group_len, 0, 0});
            const float* v_ptr = &present_value.at<float>({b, h / h_each_group_len, 0, 0});

            float* qk = &(qk_buf.at<float>({0, 0}));
            auto qk_m_stride = qk_buf.stride(0);

            if (k_stride_s == 1)
                mlas_sgemm("N",
                           "T",
                           m_cnt,
                           kv_len,
                           head_size,
                           1.0f,
                           q_ptr,
                           query.stride(2),
                           k_ptr,
                           present_key.stride(2),
                           0.f,
                           qk,
                           qk_m_stride,
                           1);
            else
                mlas_sgemm("N",
                           "N",
                           m_cnt,
                           kv_len,
                           head_size,
                           1.0f,
                           q_ptr,
                           query.stride(2),
                           k_ptr,
                           present_key.stride(3),
                           0.f,
                           qk,
                           qk_m_stride,
                           1);

            for (size_t m = m_start; m < m_end; m++) {
                // apply attention mask & sofmax
                auto ncausal = (kv_len - q_len + m + 1);
                if (sliding_window) {
                    size_t start_idx = 0;
                    auto new_causal = ncausal;
                    if (ncausal > sliding_window) {
                        start_idx = ncausal - static_cast<size_t>(sliding_window);
                        new_causal = sliding_window;
                    }
                    attn_softmax(qk + (m - m_start) * qk_m_stride + start_idx,
                                qk + (m - m_start) * qk_m_stride + start_idx,
                                d_scale,
                                nullptr,
                                nullptr,
                                nullptr,
                                false,
                                new_causal,
                                kv_len - start_idx,
                                ov::element::f32,
                                ov::element::f32);

                    memset(qk + (m - m_start) * qk_m_stride, 0, sizeof(float) * start_idx);
                } else {
                    attn_softmax(qk + (m - m_start) * qk_m_stride,
                                qk + (m - m_start) * qk_m_stride,
                                d_scale,
                                nullptr,
                                nullptr,
                                nullptr,
                                false,
                                ncausal,
                                kv_len,
                                ov::element::f32,
                                ov::element::f32);
                }
            }
            mlas_sgemm("N",
                       "N",
                       m_cnt,
                       head_size,
                       kv_len,
                       1.0f,
                       qk,
                       qk_m_stride,
                       v_ptr,
                       present_value.stride(2),
                       0.f,
                       &output_emb.at<float>({b, m_start, h * head_size}),
                       output_emb.stride(1),
                       1);
        });
    }
};
#endif

// 2nd token case : only 1 token in query
struct MHASingleToken {
    PlainTensor m_attn_w;
    PlainTensor m_temp;
    PlainTensor m_head_sum;
#ifdef OPENVINO_ARCH_X86_64
    std::shared_ptr<JitMatMulVecAMX> m_gemv;
#endif
    MHASingleToken() {}

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // alibi
    // attention_mask [B, 1, q_len, kv_len]
    // output_emb    [B, L1, H, S]
    void operator()(PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    PlainTensor& output_emb,
                    const PlainTensor& block_tables,
                    size_t max_context_len,
                    const PlainTensor& context_lens,
                    float d_scale) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        bool is_pagedattn = context_lens;
        size_t kv_len;
        if (is_pagedattn) {
            kv_len = max_context_len;
        } else {
            kv_len = present_key.size(2);
        }

        bool fastpath_valid = false;
#ifdef OPENVINO_ARCH_X86_64
        auto S = query.size(3);
        size_t block_size = present_value.size(2);
        fastpath_valid = mayiuse(amx_bf16) && (S % 32 == 0) && (block_size % 16 == 0) && (S <= 32 * 6) && present_key.get_precision() == ov::element::bf16;
        if (fastpath_valid) {
            PROFILE(_attn, "t1_qk_fast");
            m_attn_w.resize<float>({B, H, q_len, (kv_len + block_size - 1) / block_size * block_size});
            if (!m_gemv)
                m_gemv = std::make_shared<JitMatMulVecAMX>(static_cast<int>(S), static_cast<int>(block_size));
            auto h_group_num = present_value.size(1);
            size_t h_each_group_len = 1;
            if (h_group_num != H) {
                h_each_group_len = H / h_group_num;
            }
            auto kv_len_in_blocks = block_tables.m_dims[1];
            auto nthr = static_cast<size_t>(parallel_get_max_threads());
            size_t real_len = 0;
            for (size_t b = 0; b < B; b++)
                real_len += static_cast<size_t>(context_lens.ptr<int32_t>()[b]) / block_size;
            if (real_len > nthr) {
                parallel_for2d_dynamic(B, kv_len_in_blocks, [&](size_t b, size_t pk_in_blocks) {
                    auto context_len = static_cast<size_t>(context_lens.ptr<int32_t>()[b]);
                    // kv_len must be valid
                    auto pk = pk_in_blocks * block_size;
                    if (pk < context_len) {
                        m_gemv->tile_config();
                        auto block_number = block_tables.ptr<int32_t>(b)[pk_in_blocks];
                        for (size_t h_group = 0; h_group < h_group_num; h_group++) {
                            for (size_t pq = 0; pq < q_len; pq++) {
                                for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                                    (*m_gemv)(query.ptr<ov::bfloat16>(b, h, pq), present_key.ptr<ov::bfloat16>(block_number, h_group),
                                        m_attn_w.ptr<float>(b, h, pq) + pk);
                                }
                            }
                        }
                        m_gemv->tile_release();
                    }
                });
            } else {
                parallel_for3d_dynamic(B, kv_len_in_blocks, h_group_num, [&](size_t b, size_t pk_in_blocks, size_t h_group) {
                    auto context_len = static_cast<size_t>(context_lens.ptr<int32_t>()[b]);
                    // kv_len must be valid
                    auto pk = pk_in_blocks * block_size;
                    if (pk < context_len) {
                        m_gemv->tile_config();
                        auto block_number = block_tables.ptr<int32_t>(b)[pk_in_blocks];

                        for (size_t pq = 0; pq < q_len; pq++) {
                            for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                                (*m_gemv)(query.ptr<ov::bfloat16>(b, h, pq), present_key.ptr<ov::bfloat16>(block_number, h_group),
                                    m_attn_w.ptr<float>(b, h, pq) + pk);
                            }
                        }
                        m_gemv->tile_release();
                    }
                });
            }
        }
#endif
        if (!fastpath_valid) {
            // aligned to cache line (64bytes=16*sizeof(float)) to avoid false sharing
            m_attn_w.resize<float>({B, H, q_len, (kv_len + 15) / 16 * 16});
        }
        mha_single_token_pa(query, fastpath_valid ? PlainTensor() : present_key, present_value, block_tables, max_context_len,
            context_lens, output_emb, m_attn_w, m_temp,  d_scale);
    }
};

template <PagedAttention::KernelTypes KType, typename T>
struct PagedAttention::AttentionExecutor : public PagedAttention::Executor {
    GraphContext::CPtr context;
    PlainTensor attn_buf;          // f32[[B|1],[H|1], L1|1, L0+L1]

    MHAKernel<KType, T> kernel;
    MHASingleToken kernel_single_token;

    AttentionExecutor(GraphContext::CPtr ctx) : context(ctx), kernel(context) {}

    void prepare_attn_mask(MemoryPtr attn_input) {
        attn_buf.resize<float>(attn_input->getStaticDims());
        auto p = attn_input->getDataAs<uint8_t>();
        for (size_t i = 0; i < attn_input->getSize(); i++)
            attn_buf.ptr<float>()[i] = p[i] ? 0.0f : -FLT_MAX;
    }

    void execute(dnnl::stream strm, const std::vector<MemoryPtr>& inputs, const MemoryPtr output) override {
        bool is_prompt = false;
        PlainTensor present_key, present_value;
        PlainTensor q_input;           // f32[B, H, L1, S]
        PlainTensor k_input;           // f32[B, H|1, L1, S] / [B, H|1, L0+L1, S]
        PlainTensor v_input;           // f32[B, H|1, L1, S] / [B, H|1, L0+L1, S]
        PlainTensor block_tables;        // i32[B, max_kvLen]
        PlainTensor context_lens;
        PlainTensor output_emb(output);
        float scale_input = 0.0f;
        size_t B, L1, S;
        size_t sliding_window = 0;
        size_t max_context_len = 0;

        PROFILE(_attn, "attn_execute");
        q_input.reset(inputs[0]);
        k_input.reset(inputs[1]);
        v_input.reset(inputs[2]);
        present_key.reset(inputs[ID_KCACHE]);
        present_value.reset(inputs[ID_VCACHE]);

        is_prompt = *inputs[ID_IS_PROMPT]->getDataAs<uint8_t>() == 1;
        max_context_len = static_cast<size_t>(*inputs[ID_MAX_CONTEXT_LEN]->getDataAs<int32_t>());
        context_lens.reset(inputs[ID_CONTEXT_LENS]);
        block_tables.reset(inputs[ID_BLOCK_TABLES]);
        scale_input = *inputs[ID_SCALE]->getDataAs<float>();

        // q: [B, L1, H*S], kv: [B, L1, Hk*S]
        // k_cache: [NUM_BLOCKS, Hk, 16, S]
        // v_cache: [NUM_BLOCKS, Hk, 16, S]
        // context_lens: [B]
        // block_tables: [B, max_block_per_request]
        B = k_input.size(0);
        L1 = k_input.size(1);
        auto Hk = present_key.size(1);
        // The layout for per token per head for u8 kv cache:
        // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
        // The actual size needs to deduct scale and zeropoint.
        S = present_value.size(3) - (present_value.m_dt == ov::element::Type_t::u8 ? sizeof(float) * 2 : 0);
        auto H = q_input.size(2) / S;

        q_input.assert_dims({B, L1, H * S});
        if (!is_prompt) {
            context_lens.assert_dims({B});
            block_tables.assert_dims({B, 0}, true);
        } else {
            sliding_window = static_cast<size_t>(*inputs[ID_SLIDING_WINDOW]->getDataAs<int32_t>());
        }
        output_emb.assert_dims({B, L1, H * S});
        q_input = q_input.reshape({B, L1, H, S}).permute({0, 2, 1, 3});
        k_input = k_input.reshape({B, L1, Hk, S}).permute({0, 2, 1, 3});
        v_input = v_input.reshape({B, L1, Hk, S}).permute({0, 2, 1, 3});

        // second token, or first token with pastkv fusing
        if (is_prompt) {
            char buf[256];
            snprintf(buf, sizeof(buf), "first_BL%ld,%ld", B, L1);
            _attn = ov::intel_cpu::profilerManagerInstance.startProfile(buf);

            // multi-token version
            kernel(strm, q_input, k_input, v_input, output_emb, scale_input, sliding_window);
        } else {
            // 1-token version
            // for second token, using a special AVX2/AVX512 float path:
            //  1, in matrix mutiply, using AMX is not efficency because the M dimension of A will alway be 1
            //  2, using float will save the repack cost which typically is required for bf16/int8 opt
            //  3, using dot product can leverage the SIMD while easily adapt to indirect kv cache
            kernel_single_token(q_input, present_key, present_value,
                output_emb, block_tables, max_context_len, context_lens, scale_input);
        }
    }
};

PagedAttention::PagedAttention(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, PAShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }
}

void PagedAttention::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto rtPrecision = getRuntimePrecision();

    NodeConfig config;
    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    config.inConfs.resize(getOriginalInputsNumber());
    config.outConfs.resize(getOriginalOutputsNumber());
    config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getInputShapeAtPort(0)));
    config.inConfs[1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getInputShapeAtPort(1)));
    config.inConfs[2].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getInputShapeAtPort(2)));

    OPENVINO_ASSERT(getOriginalInputsNumber() == 13, "The input number of PagedAttention should be 13.");
    // kvcache, float, []
    auto past_kv_input_mem_precision = getOriginalInputPrecisionAtPort(ID_KCACHE);
    config.inConfs[ID_KCACHE].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        past_kv_input_mem_precision, getInputShapeAtPort(ID_KCACHE)));
    config.inConfs[ID_VCACHE].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        past_kv_input_mem_precision, getInputShapeAtPort(ID_VCACHE)));
    // is_prompt, bool, []
    config.inConfs[ID_IS_PROMPT].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::u8, getInputShapeAtPort(ID_IS_PROMPT)));
    // slot_mapping, int, [batch_size, max_context_len]
    config.inConfs[ID_SLOT_MAPPING].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::i32, getInputShapeAtPort(ID_SLOT_MAPPING)));
    // max_context_len, int, []
    config.inConfs[ID_MAX_CONTEXT_LEN].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::i32, getInputShapeAtPort(ID_MAX_CONTEXT_LEN)));
    // context_lens, int, [batch_size]
    config.inConfs[ID_CONTEXT_LENS].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::i32, getInputShapeAtPort(ID_CONTEXT_LENS)));
    // block_tables, int, [batch_size, max_block_per_request]
    config.inConfs[ID_BLOCK_TABLES].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::i32, getInputShapeAtPort(ID_BLOCK_TABLES)));
    // scale, float, []
    config.inConfs[ID_SCALE].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::f32, getInputShapeAtPort(ID_SCALE)));
    // alibi_slopes, float, [?] or nullptr
    config.inConfs[ID_ALIBI_SLOPES].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::f32, getInputShapeAtPort(ID_ALIBI_SLOPES)));
    // sliding_window, int, []
    config.inConfs[ID_SLIDING_WINDOW].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::i32, getInputShapeAtPort(ID_SLIDING_WINDOW)));

    config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getOutputShapeAtPort(0)));

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref_any);
}

void PagedAttention::createPrimitive() {
    auto rtPrecision = getRuntimePrecision();

    PagedAttentionKey key = {rtPrecision};

    auto builder = [&](const PagedAttentionKey& key) -> std::shared_ptr<Executor> {
        std::shared_ptr<Executor> executor = nullptr;
        if (rtPrecision == ov::element::bf16) {
#ifdef OPENVINO_ARCH_X86_64
            executor = std::make_shared<AttentionExecutor<KT_ONEDNN, ov::bfloat16>>(context);
#endif
        } else {
#ifdef OV_CPU_WITH_MLAS
            executor = std::make_shared<AttentionExecutor<KT_MLAS, float>>(context);
#elif defined(OPENVINO_ARCH_X86_64)
            if (with_cpu_x86_avx512_core()) {
                executor = std::make_shared<AttentionExecutor<KT_ONEDNN, float>>(context);
            } else {
                executor = std::make_shared<AttentionExecutor<KT_REF, float>>(context);
            }
#else
            executor = std::make_shared<AttentionExecutor<KT_REF, float>>(context);
#endif
        }
        return executor;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    if (!result.first) {
        OPENVINO_THROW("PagedAttention AttentionExecutor creation fails with precision " + rtPrecision.to_string());
    }
    m_executor = result.first;
}

void PagedAttention::execute(dnnl::stream strm) {
    PROFILE(_attn, "pg_concat");
    auto orginInputNumber = getOriginalInputsNumber();
    std::vector<MemoryPtr> inputs(orginInputNumber);
    auto output = getDstMemoryAtPort(0);
    for (size_t i = 0; i < orginInputNumber; i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }

    gatherConcatPastkvForPagedAttn(inputs);

    _attn = ov::intel_cpu::profilerManagerInstance.startProfile("pg_exec");
    m_executor->execute(strm, inputs, output);
}

bool PagedAttention::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        int orgInput = static_cast<int>(op->get_input_size());
        if (op->get_type_name() == std::string("PagedAttentionExtension") && orgInput == ID_SLIDING_WINDOW + 1) {
            return true;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void PagedAttention::gatherConcatPastkvForPagedAttn(const std::vector<MemoryPtr>& inputs) {
    PlainTensor k, v, k_cache, v_cache, slot_mapping;

    k.reset(inputs[ID_K]);                          // [B, L1, H * S]
    v.reset(inputs[ID_V]);
    k_cache.reset(inputs[ID_KCACHE]);               // [NUM_BLOCKS, H, 16, S]
    v_cache.reset(inputs[ID_VCACHE]);               // [NUM_BLOCKS, H, 16, S]
    slot_mapping.reset(inputs[ID_SLOT_MAPPING]);    // [B, max_context_len]

    auto B = k.size(0);
    auto L1 = k.size(1);
    auto H = k_cache.size(1);
    auto S = v_cache.size(3) - (k_cache.m_dt == ov::element::Type_t::u8 ? 8 : 0);

    k.assert_dims({B, L1, H * S});
    v.assert_dims({B, L1, H * S});
    slot_mapping.assert_dims({B, 0}, true);
    k = k.reshape({B, L1, H, S}).permute({0, 2, 1, 3});
    v = v.reshape({B, L1, H, S}).permute({0, 2, 1, 3});
    if (k_cache.m_dt == ov::element::Type_t::u8) {
        k_cache.assert_dims({0, H, 0, S + 8}, true);
        v_cache.assert_dims({k_cache.m_dims[0], H, k_cache.m_dims[2], S + 8});
        paged_attn_quantkv(k, v, k_cache, v_cache, slot_mapping);
    } else {
        k_cache.assert_dims({0, H, 0, S}, true);
        v_cache.assert_dims({k_cache.m_dims[0], H, k_cache.m_dims[2], S});
        paged_attn_memcpy(k, v, k_cache, v_cache, slot_mapping);
    }
}

ov::element::Type PagedAttention::getRuntimePrecision() const {
    auto rtPrecision = getOriginalInputPrecisionAtPort(0);
    // bf16 should be enabled only when platform supports
    if (rtPrecision == ov::element::bf16 && ov::with_cpu_x86_bfloat16()) {
        rtPrecision = ov::element::bf16;
    } else {
        rtPrecision = ov::element::f32;
    }
    return rtPrecision;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
