// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_kernel_base.hpp"
#include <cpu/x64/injectors/jit_uni_depthwise_injector.hpp>
#include <cpu/x64/injectors/jit_uni_quantization_injector.hpp>
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>
#include <emitters/x64/jit_eltwise_emitters.hpp>

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
    bool fuse_low_precision;
    element::Type src_el_type;
    element::Type dst_el_type;
};

struct JitReduceCallArgs {
    const void* src;
    const void* idx;
    void* dst;
    size_t work_amount;
    size_t work_batch;
    size_t reduce_w = 2;    // only used in planar layout  [1: reduce width dimension]   [0: reduce other dimension] [other value: N/A]
    size_t reduce_stride;   // only used in planar layout while reducing dimensions except for width
    size_t can_divide;      // if apply division in reduce_kernel [1: Yes] [0: No]
    const void* divisor;    // mean = sum / divisor
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
class JitReduceKernelBase : public JitKernel<JitReduceConfigParams, CallArgs> {
public:
    explicit JitReduceKernelBase(const char* name, const JitReduceConfigParams& jcp, dnnl::impl::cpu::x64::cpu_isa_t isa);

    virtual ~JitReduceKernelBase() = default;

    const element::Type &get_exec_prc() const {
        return exec_el_type;
    }

protected:
    void horiz_ps(const Xbyak::Xmm& xmm, const Xbyak::Operand& op);

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void horiz_qq(const Xbyak::Xmm& xmm, const Xbyak::Operand& op);

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void horiz_reduce_store_ps(const Xbyak::Xmm& vmm_dst, const element::Type& dst_dt, bool load_embedded = false);

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void horiz_reduce_store_qq(const Xbyak::Xmm& vmm_dst, const element::Type& dst_dt, bool load_embedded = false);


    RegistersPool::Reg<Xbyak::Reg64> reg_src;
    RegistersPool::Reg<Xbyak::Reg64> reg_dst;
    RegistersPool::Reg<Xbyak::Reg64> reg_work_amount;

    element::Type exec_el_type;
    bool post_reduce = false;
    bool post_ops_fusing = false;
    bool planar_layout = false;
    int loop_step = 1;

    std::shared_ptr<jit_maximum_emitter>  max_emitter;
    std::shared_ptr<jit_minimum_emitter>  min_emitter;
    std::shared_ptr<jit_multiply_emitter> mul_emitter;
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

    Xbyak::Address table_val(int index) { return ptr[reg_table + index * vlen]; }

    const Xbyak::Reg64 reg_params = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    RegistersPool::Reg<Xbyak::Reg64> reg_reduce_w;
    RegistersPool::Reg<Xbyak::Reg64> reg_reduce_stride;
    RegistersPool::Reg<Xbyak::Reg64> reg_work_batch;
    RegistersPool::Reg<Xbyak::Reg64> reg_table;

    RegistersPool::Reg<Vmm> v_src;
    RegistersPool::Reg<Vmm> v_dst;
    RegistersPool::Reg<Vmm> v_zero;
    RegistersPool::Reg<Vmm> v_dst_aux;
    RegistersPool::Reg<Vmm> v_idx;
    RegistersPool::Reg<Vmm> v_ones;
    RegistersPool::Reg<Vmm> v_abs_mask;

    const Xbyak::Opmask &k_mask = k1;

    Xbyak::Label l_table;

    std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_eltwise_injector_f32<isa>> exp_injector;

    void reduce_main();

    void reduce_tail();

    void init_reg_reduce_stride();

    void reduce_kernel();

    void reduce_once();

    void reduce_batch();

    void reduce_gather(const Vmm& vmm_dst, int64_t offset);

    void pack_gathered_vector(const Vmm& vmm_val, const Vmm& vmm_index, int64_t offset, const element::Type& src_dt);

    void reduce_kernel_tail();

    void reduce_once_tail();

    void reduce_batch_tail();

    void reduce_main_loop();

    void reduce_kernel(const Vmm& vmm_src, const Vmm& vmm_dst);

    void reduce_kernel_scalar(const Xbyak::Xmm& xmm_src, const Xbyak::Xmm& xmm_dst);

    void load_dst_vector();

    void store_dst_vector();

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
        // uint64_t int64_min = 0x0000000000000000; // lowest int64
        uint64_t int64_min = 0x8000000000000000; // lowest int64
        uint64_t int64_max = 0x7fffffffffffffff; // max int64
    } aux_vals;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct JitReducePostKernel : public JitReduceKernelBase<JitReducePostCallArgs> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(JitReducePostKernel)

    explicit JitReducePostKernel(const JitReduceConfigParams& jcp, const dnnl_primitive_attr& attr);

    void generate() override;

private:
    const dnnl_primitive_attr &attr;

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41, Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,  Xbyak::Ymm,
                                                                                             Xbyak::Zmm>::type;
    const size_t vlen = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;

    const Xbyak::Reg64 reg_params = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    RegistersPool::Reg<Xbyak::Reg64> reg_divisor;
    RegistersPool::Reg<Xbyak::Reg64> reg_reduce_c;
    RegistersPool::Reg<Xbyak::Reg64> reg_oc_off;
    RegistersPool::Reg<Xbyak::Reg64> reg_d_weights;
    RegistersPool::Reg<Xbyak::Reg64> reg_d_bias;
    RegistersPool::Reg<Xbyak::Reg64> reg_post_ops_data;

    RegistersPool::Reg<Vmm> v_dst;
    RegistersPool::Reg<Vmm> v_d_weights;
    RegistersPool::Reg<Vmm> v_d_bias;
    RegistersPool::Reg<Vmm> v_divisor;

    std::shared_ptr<jit_divide_emitter>  division_emitter;
    std::shared_ptr<jit_sqrt_emitter>    sqrt_emitter;
    std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_eltwise_injector_f32<isa>> log_injector;

    std::vector<std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    void reduce_post_main();

    void reduce_post_tail();

    void apply_post_ops(const element::Type& dst_dt, bool is_broadcast);

    void reduce_map_kernel(const Vmm& vmm_dst);

    void reduce_map_kernel_scalar(const Xbyak::Xmm& xmm_dst);

    void wrap_load_vector(const Vmm& vmm_val, const element::Type& dst_dt, const element::Type& src_dt, size_t offset);

    void wrap_load_scalar(const Xbyak::Xmm& xmm_val, const element::Type& dst_dt, const element::Type& src_dt, size_t offset);

    void horiz_store(const Xbyak::Xmm& xmm_dst, const element::Type& dst_dt, bool load_embedded);
};  // JitReducePostKernel

}   // namespace kernel
}   // namespace intel_cpu
}   // namespace ov
