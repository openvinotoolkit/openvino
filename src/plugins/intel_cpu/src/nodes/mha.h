// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <string>
#include <vector>
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/matmul/brgemm_matmul_copy_utils.hpp>
#include <cpu/x64/matmul/brgemm_matmul_utils.hpp>
#include <cpu/x64/amx_tile_configure.hpp>

namespace ov {
namespace intel_cpu {
namespace node {

struct jit_mul_add_softmax_compile_params {
    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision dst_prc;
    size_t work_amount;
    bool with_mul_scales;
    bool is_mul_first;
    bool with_scales0;
    bool broadcast_scales0;
    bool with_scales1;
    bool broadcast_scales1;
};

struct jit_mul_add_softmax_call_args {
    const void *p_in0;
    const void *p_mul_in1;
    const void *p_add_in1;
    void *p_out;
    void *p_buffer;
    const void *p_scales0;
    const void *p_scales1;
};

struct jit_uni_mul_add_softmax_kernel {
    void (*ker_)(const jit_mul_add_softmax_call_args*);

    void operator()(const jit_mul_add_softmax_call_args* call_args) {
        assert(ker_);
        ker_(call_args);
    }

    explicit jit_uni_mul_add_softmax_kernel(const jit_mul_add_softmax_compile_params& jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_mul_add_softmax_kernel() {}

    virtual void create_ker() = 0;

    jit_mul_add_softmax_compile_params jcp_;
};

struct jit_convert_reorder_compile_params {
    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision dst_prc;
    size_t inner_work_amount;
    bool with_scales;
    bool broadcast_scales;
    size_t src_stride;
    size_t dst_stride;
};

struct jit_convert_reorder_call_args {
    const void *p_in;
    void *p_out;
    const void *p_scales;
    size_t outter_work_amount;
};

struct jit_uni_convert_reorder_kernel {
    void (*ker_)(const jit_convert_reorder_call_args*);

    void operator()(const jit_convert_reorder_call_args* call_args) {
        assert(ker_);
        ker_(call_args);
    }

    explicit jit_uni_convert_reorder_kernel(const jit_convert_reorder_compile_params& jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_convert_reorder_kernel() {}

    virtual void create_ker() = 0;

    jit_convert_reorder_compile_params jcp_;
};

struct jit_convert_transpose_compile_params {
    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision dst_prc;
    size_t inner_work_amount;
    size_t outter_work_amount;
    bool with_scales;
    bool broadcast_scales;
    size_t inner_src_stride;
    size_t outter_src_stride;
    size_t outter_dst_stride;
};

struct jit_convert_transpose_call_args {
    const void *p_in;
    void *p_out;
    const void *p_scales;
};

struct jit_uni_convert_transpose_kernel {
    void (*ker_)(const jit_convert_transpose_call_args*);

    void operator()(const jit_convert_transpose_call_args* call_args) {
        assert(ker_);
        ker_(call_args);
    }

    explicit jit_uni_convert_transpose_kernel(const jit_convert_transpose_compile_params& jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_convert_transpose_kernel() {}

    virtual void create_ker() = 0;

    jit_convert_transpose_compile_params jcp_;
};

#define MHA_BRGEMM_KERNELS_NUM 8

class MHA : public Node {
public:
    MHA(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
    void prepareParams() override;

private:
    struct brgemmCtx {
        size_t M, N, K, LDA, LDB, LDC;
        dnnl_data_type_t dt_in0, dt_in1;
        char palette[64];
        bool is_with_amx;
        bool is_with_comp;
        float beta;
    };

    template <typename in1_type>
    void mhaImpl();

    void init_brgemm(brgemmCtx& ctx, std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& brgKernel, bool use_amx);
    void init_brgemm_copy_a(std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t>& brgCopyKernel,
        size_t K, size_t K_blk, size_t K_tail, size_t LDA, dnnl_data_type_t dt_in0);
    void init_brgemm_copy_b(std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t>& brgCopyKernel,
        size_t N, size_t N_blk, size_t N_tail, size_t LDB, size_t K, bool is_with_amx, dnnl_data_type_t dt_in0, dnnl_data_type_t dt_in1);

    void callBrgemm(brgemmCtx& ctx, std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& brgKernel,
                    const void* pin0, const void* pin1, void* pout, void* wsp);

    size_t getBrgIdx(size_t mIdx, size_t kIdx, size_t nIdx) {
        return mIdx * 4 + kIdx * 2 + nIdx;
    }

    std::vector<InferenceEngine::Precision> inputPrecisions;
    InferenceEngine::Precision accPrecision0;
    InferenceEngine::Precision accPrecision1;

    VectorDims dimsTranspose0In0;
    VectorDims dimsTranspose1In0;
    VectorDims dimsMulIn1;
    VectorDims dimsAddIn1;
    VectorDims dimsTranspose2In0;
    VectorDims dimsOut;

    VectorDims strTranspose0In0;
    VectorDims strTranspose1In0;
    VectorDims strMulIn1;
    VectorDims strAddIn1;
    VectorDims strTranspose2In0;
    VectorDims strOut;

    VectorDims dimsMatMul0In0;
    VectorDims dimsMatMul0In1;
    VectorDims dimsMatMul0Out;
    VectorDims dimsMatMul1In1;
    VectorDims dimsMatMul1Out;

    size_t batch0, batch1;
    size_t M, M_blk, M_tail;
    size_t K0, K0_blk, K0_tail, N0, N0_blk, N0_tail;
    size_t K1, K1_blk, K1_tail, N1, N1_blk, N1_tail;

    size_t bufferMatMul0In0Size;
    size_t bufferMatMul0In1Size;
    size_t bufferMatMul0OutSize;
    size_t bufferMatMul1In1Size;
    size_t bufferMatMul1OutSize;
    size_t bufferCompensation0Size;
    size_t bufferCompensation1Size;
    size_t wsp_size_per_thread = 4 * 1024;

    std::vector<uint8_t> bufferMatMul0In0;
    std::vector<uint8_t> bufferMatMul0In1;
    std::vector<uint8_t> bufferMatMul0Out;
    std::vector<uint8_t> bufferMatMul1In1;
    std::vector<uint8_t> bufferMatMul1Out;
    std::vector<int32_t> bufferCompensation0;
    std::vector<int32_t> bufferCompensation1;
    std::vector<size_t> wsp;

    bool isMulFirst;
    InferenceEngine::Precision fqPrc2;

    std::vector<float> mulScales;
    std::vector<float> fqScales0;
    std::vector<float> fqScales1;
    std::vector<float> fqScales2;
    std::vector<float> fqScales3;

    size_t brg0VnniFactor;
    brgemmCtx brgCtxs0[MHA_BRGEMM_KERNELS_NUM];
    std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgKernels0[MHA_BRGEMM_KERNELS_NUM];
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t> brgCopyAKernel0;
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t> brgCopyBKernel0;

    size_t brg1VnniFactor;
    brgemmCtx brgCtxs1[MHA_BRGEMM_KERNELS_NUM];
    std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgKernels1[MHA_BRGEMM_KERNELS_NUM];
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t> brgCopyBKernel1;

    std::unique_ptr<jit_uni_mul_add_softmax_kernel> mulAddSoftmaxKernel;
    std::unique_ptr<jit_uni_convert_reorder_kernel> convertReorderKernel;
    std::unique_ptr<jit_uni_convert_transpose_kernel> convertTransposeKernel;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
