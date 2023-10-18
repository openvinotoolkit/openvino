// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_kernel_base.hpp"

#if defined(OPENVINO_ARCH_X86_64)

namespace ov {
namespace intel_cpu {
namespace kernel {

struct RandomUniformCompileParams {
    element::Type out_data_type = element::f32;
};

struct RandomUniformCallArgs {
    void* dst_ptr;
    const void* key_ptr;
    const void* counter_ptr;
    const void* n_ptr;
    const void* min_ptr;
    const void* range_ptr;
    uint64_t work_amount = 0lu;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
class RandomUniform : public JitKernel<RandomUniformCompileParams, RandomUniformCallArgs> {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(RandomUniform)

    explicit RandomUniform(const RandomUniformCompileParams& jcp);

    void generate() override;

private:
    using Vmm   = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core, Xbyak::Zmm,
                                                           isa == dnnl::impl::cpu::x64::sse41,       Xbyak::Xmm,
                                                                                                     Xbyak::Ymm>::type;
    using Vmask = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core, Xbyak::Opmask,
                                                           isa == dnnl::impl::cpu::x64::sse41,       Xbyak::Xmm,
                                                                                                     Xbyak::Ymm>::type;

    RegistersPool::Reg<Xbyak::Reg64> r64_dst;
    RegistersPool::Reg<Xbyak::Reg64> r64_work_amount;
    RegistersPool::Reg<Xbyak::Reg64> r64_n_inc;
    RegistersPool::Reg<Xbyak::Reg64> r64_convert_0;
    RegistersPool::Reg<Xbyak::Reg64> r64_convert_1;
    RegistersPool::Reg<Xbyak::Reg64> r64_min;
    RegistersPool::Reg<Xbyak::Reg64> r64_f64_pow_52;

    const Xbyak::Reg64 r64_params = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    // Vector registers.
    RegistersPool::Reg<Vmm> v_max_mul_n_64;
    RegistersPool::Reg<Vmm> v_max_mul_c_64;
    RegistersPool::Reg<Vmm> v_add_low_k;
    RegistersPool::Reg<Vmm> v_add_up_k;
    RegistersPool::Reg<Vmm> v_convert_0;
    RegistersPool::Reg<Vmm> v_convert_1;
    RegistersPool::Reg<Vmm> v_convert_2;
    RegistersPool::Reg<Vmm> v_n_inc;
    RegistersPool::Reg<Vmm> v_key_64;
    RegistersPool::Reg<Vmm> v_counter_64;
    RegistersPool::Reg<Vmm> v_n_64;
    RegistersPool::Reg<Vmm> v_min;
    RegistersPool::Reg<Vmm> v_range;
    RegistersPool::Reg<Vmm> v_res_perm;
    RegistersPool::Reg<Vmm> v_perm_16;

    void initVectors();

    void process();

    void runPhilox(const std::vector<Vmm>& vmm_res, const Vmm& vmm_key, const Vmm& vmm_counter, const Vmm& vmm_n);

    void calculateRound(const Vmm& vmm_k_0, const Vmm& vmm_k_1, const Vmm& vmm_c_0, const Vmm& vmm_c_1,
                        const Vmm& vmm_n_0, const Vmm& vmm_n_1, const Vmm& vmm_aux_0, const Vmm& vmm_aux_1);

    void raiseKey(const Vmm& vmm_k_0, const Vmm& vmm_k_1);

    void convert(const std::vector<Vmm>& vmm_dst, const std::vector<Vmm>& vmm_src);

    void tail(const std::vector<Vmm>& vmm_dst);

    static constexpr uint64_t ROUNDS_NUMBER = 10lu;
    static constexpr uint32_t CRUSH_RESISTANCE_CONST_LOWER_VALUE = 0x9E3779B9;
    static constexpr uint32_t CRUSH_RESISTANCE_CONST_UPPER_VALUE = 0xBB67AE85;
    static constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_N = 0xD2511F53;
    static constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER = 0xCD9E8D57;
};

}   // namespace kernel
}   // namespace intel_cpu
}   // namespace ov

#endif // OPENVINO_ARCH_X86_64
