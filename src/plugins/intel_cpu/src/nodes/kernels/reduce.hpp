// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_kernel_base.hpp"
#include <cpu/x64/injectors/jit_uni_depthwise_injector.hpp>
#include <cpu/x64/injectors/jit_uni_quantization_injector.hpp>
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>
#include "emitters/jit_bf16_emitters.hpp"

namespace ov {
namespace intel_cpu {
namespace kernel {

enum ReduceLayoutType {
    reduce_ncsp,
    reduce_nspc,
    reduce_blocked
};

struct JitReduceConfigParams {
    ReduceLayoutType layout;
    Algorithm reduce_mode;
    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision dst_prc;
};

struct JitReduceCallArgs {
    const void *src;
    const void *idx;
    void *dst;
    size_t work_amount;
    size_t work_batch;
    size_t reduce_w = 2;    // only used in planar layout  [1: reduce width dimension]   [0: reduce other dimension] [other value: N/A]
    size_t reduce_stride;   // only used in planar layout while reducing dimensions except for width
};

struct JitReducePostCallArgs {
    const void *src;
    void *dst;
    size_t work_amount;
    size_t reduce_c = 2;    // only used in blocked layout [1: reduce channel dimension] [0: reduce other dimension] [other value: N/A]
    size_t oc_off;          // offset in byte along channel on output tensor
    size_t channel_size;    // only for post ops fusion of nspc layout
    const void *divisor;    // mean = sum / divisor
    const void** post_op_data;
};


template<typename CallArgs>
struct JitReduceKernelBase : public JitKernelBase {
    void (*kernel_func)(const CallArgs *);

    void operator()(const CallArgs *args) {
        assert(kernel_func);
        kernel_func(args);
    }

    explicit JitReduceKernelBase(const JitReduceConfigParams &jcp, const char *name) : JitKernelBase(name), kernel_func(nullptr), jcp(jcp) {
        if (jcp.src_prc.size() <= 4) {
            exec_prc = InferenceEngine::Precision::FP32;
        } else if (jcp.src_prc.size() == 8) {
            exec_prc = jcp.src_prc;
        }
    }

    virtual ~JitReduceKernelBase() = default;

    dnnl::impl::status_t create_kernel() override {
        const auto code = jit_generator::create_kernel();
        if (code != dnnl::impl::status::success) {
            IE_THROW() << "Could not create kernel. Error code: " << std::to_string(code) << ". " <<
                       "Xbyak error code: " << Xbyak::ConvertErrorToString(Xbyak::GetError());
        }
        kernel_func = (decltype(kernel_func))jit_ker();
        return code;
    }

    const InferenceEngine::Precision &get_exec_prc() {
        return exec_prc;
    }

protected:
    void horiz_ps(const Xbyak::Xmm &xmm, const Xbyak::Operand &op);

    void horiz_pd(const Xbyak::Xmm &xmm, const Xbyak::Operand &op);

    void horiz_qq(const Xbyak::Xmm &xmm, const Xbyak::Operand &op);

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void horiz_reduce_store_ps(const Xbyak::Xmm &vmm_dst, const InferenceEngine::Precision &dst_dt, bool load_embedded = false);

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void horiz_reduce_store_qq(const Xbyak::Xmm &vmm_dst, const InferenceEngine::Precision &dst_dt, bool load_embedded = false);

    JitReduceConfigParams jcp;
    InferenceEngine::Precision exec_prc;

    const Xbyak::Reg64 &reg_dst = r9;

    Xbyak::Xmm xmm_aux1;
    Xbyak::Xmm xmm_aux2;
    Xbyak::Xmm xmm_aux3;

    Xbyak::Ymm ymm_aux1;
};


template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct JitReduceKernel : public JitReduceKernelBase<JitReduceCallArgs> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(JitReduceKernel)

    explicit JitReduceKernel(const JitReduceConfigParams &jcp);

    void generate() override;

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41, Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,  Xbyak::Ymm,
                                                                                             Xbyak::Zmm>::type;
    const size_t vlen = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;
    bool planar_layout = false;

    Xbyak::Address table_val(int index) { return ptr[reg_table + index * vlen]; }

    const Xbyak::Reg64 &reg_src            = r8;
    const Xbyak::Reg64 &reg_idx            = rdx;
    const Xbyak::Reg64 &reg_work_amount    = r10;
    const Xbyak::Reg64 &reg_reduce_w       = r11;
    const Xbyak::Reg64 &reg_reduce_stride  = r12;
    const Xbyak::Reg64 &reg_work_batch     = r13;
    const Xbyak::Reg64 &reg_table          = r14;
    const Xbyak::Reg64 reg_params          = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    const Xbyak::Reg8  &reg_tmp_8          = r15b;
    const Xbyak::Reg32 &reg_tmp_32         = r15d;
    const Xbyak::Reg64 &reg_tmp_64         = r15;

    const Xbyak::Reg64 &reg_src_aux        = rax;
    const Xbyak::Reg64 &reg_work_batch_aux = rbx;

    Vmm vmm_aux     = Vmm(0);
    Vmm vmm_src     = Vmm(1);
    Vmm vmm_dst     = Vmm(2);
    Vmm vmm_zero    = Vmm(3);
    Vmm vmm_dst_aux = Vmm(4);
    Vmm vmm_idx     = Vmm(8);
    Vmm vmm_mask    = Vmm(9);

    Xbyak::Xmm xmm_aux  = Xbyak::Xmm(vmm_aux.getIdx());
    Xbyak::Xmm xmm_src  = Xbyak::Xmm(vmm_src.getIdx());
    Xbyak::Xmm xmm_dst  = Xbyak::Xmm(vmm_dst.getIdx());
    Xbyak::Xmm xmm_zero = Xbyak::Xmm(vmm_zero.getIdx());
    Xbyak::Xmm xmm_idx  = Xbyak::Xmm(vmm_idx.getIdx());

    Xbyak::Ymm ymm_idx  = Xbyak::Ymm(vmm_idx.getIdx());

    const Xbyak::Opmask &k_mask = k1;

    Xbyak::Label l_table;

    std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_eltwise_injector_f32<isa>> exp_injector;

    void reduce_main();

    void reduce_tail();

    void init_reg_reduce_stride();

    void reduce_kernel();

    void reduce_once();

    void reduce_batch();

    void reduce_gather(const Vmm &vmm_dst, int64_t offset);

    void pack_gathered_vector(const Vmm &vmm_val, const Vmm &vmm_index, int64_t offset, const InferenceEngine::Precision &src_dt);

    void reduce_kernel_tail();

    void reduce_once_tail();

    void reduce_batch_tail();

    void reduce_main_loop();

    void reduce_kernel(const Vmm &vmm_src, const Vmm &vmm_dst);

    void reduce_kernel_scalar(const Xbyak::Xmm &xmm_src, const Xbyak::Xmm &xmm_dst);

    void load_dst_vector();

    void store_dst_vector();

    void horiz_reduce_store_pd(const Vmm &vmm_dst, const InferenceEngine::Precision &dst_dt, bool load_embedded = false);

    void horiz_store_pd(const Xbyak::Xmm &xmm_dst, const InferenceEngine::Precision &dst_dt, bool load_embedded);

    void prepare_aux_table();

    const struct aux_vals_type {
        uint32_t float_one = 0x3f800000; // 1.0f
        uint32_t float_abs = 0x7fffffff; // mask to make positive
        uint32_t float_min = 0xff7fffff; // float lowest
        uint32_t float_max = 0x7f7fffff; // float maximum
        uint32_t float_int32_min = 0xcf000000; // -2^31 presented in float
        uint32_t float_int32_max = 0x4effffff; // 2^31-1 presented in float

        uint64_t double_one = 0x3ff0000000000000; // 1.0
        uint64_t double_abs = 0x7fffffffffffffff; // mask to make positive
        uint64_t double_min = 0xffefffffffffffff; // double lowest
        uint64_t double_max = 0x7fefffffffffffff; // double maximum
        uint64_t double_int64_min = 0xc3e0000000000000; // lowest int64 presented in double
        uint64_t double_int64_max = 0x43dfffffffffffff; // max int64 presented in double

        uint64_t int64_one = 0x0000000000000001; // 1
        uint64_t int64_abs = 0x7fffffffffffffff; // mask to make positive
        uint64_t int64_min = 0x0000000000000000; // lowest int64 presented in double
        uint64_t int64_max = 0x7fffffffffffffff; // max int64
    } aux_vals;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct JitReducePostKernel : public JitReduceKernelBase<JitReducePostCallArgs> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(JitReducePostKernel)

    explicit JitReducePostKernel(const JitReduceConfigParams &jcp, const dnnl_primitive_attr &attr);

    void generate() override;

private:
    const dnnl_primitive_attr &attr;

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41, Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,  Xbyak::Ymm,
                                                                                             Xbyak::Zmm>::type;
    const size_t vlen = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;
    bool planar_layout = false;

    const Xbyak::Reg64 &reg_work_amount       = r8;
    const Xbyak::Reg64 &reg_total_work_amount = r10;
    const Xbyak::Reg64 &reg_channel_size      = r11;
    const Xbyak::Reg64 &reg_divisor           = r12;
    const Xbyak::Reg64 &reg_reduce_c          = r13;
    const Xbyak::Reg64 reg_params             = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    const Xbyak::Reg8  &reg_tmp_8             = r14b;
    const Xbyak::Reg32 &reg_tmp_32            = r14d;
    const Xbyak::Reg64 &reg_tmp_64            = r14;

    const Xbyak::Reg64 &reg_oc_off            = rax;
    const Xbyak::Reg64 &reg_d_weights         = rbx;
    const Xbyak::Reg64 &reg_d_bias            = rdx;
    const Xbyak::Reg64 &reg_post_ops_data     = r15;

    Vmm vmm_aux       = Vmm(0);
    Vmm vmm_dst       = Vmm(1);
    Vmm vmm_zero      = Vmm(2);
    Vmm vmm_dst_aux   = Vmm(3);
    Vmm vmm_d_weights = Vmm(7);
    Vmm vmm_d_bias    = Vmm(8);

    Xbyak::Xmm xmm_aux  = Xbyak::Xmm(vmm_aux.getIdx());
    Xbyak::Xmm xmm_dst  = Xbyak::Xmm(vmm_dst.getIdx());

    std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_eltwise_injector_f32<isa>> log_injector;

    std::vector<std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    void reduce_post_main();

    void reduce_post_tail();

    void apply_post_ops(const InferenceEngine::Precision &dst_dt, bool is_broadcast);

    void reduce_map_kernel(const Vmm &vmm_dst);

    void reduce_map_kernel_scalar(const Xbyak::Xmm &xmm_dst);

    void horiz_reduce_store_pd(const Vmm &vmm_dst, const InferenceEngine::Precision &dst_dt, bool load_embedded = false);

    void horiz_store(const Xbyak::Xmm &xmm_dst, const InferenceEngine::Precision &dst_dt, bool load_embedded);
};

}   // namespace kernel
}   // namespace intel_cpu
}   // namespace ov
