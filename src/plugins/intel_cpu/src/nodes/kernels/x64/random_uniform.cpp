// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "random_uniform.hpp"

#include <immintrin.h>

using namespace dnnl::impl::cpu;

namespace ov::intel_cpu::kernel::random_uniform {

#define GET_PHILOX_OFFSET(field) offsetof(PhiloxGeneratorCallArgs, field)

#define GET_MERSENNE_OFFSET(field) offsetof(MersenneTwisterGeneratorCallArgs, field)

#define BROADCAST_CONSTANT(func, vector, aux_register, constant) \
    mov(aux_register, constant);                                 \
    func(vector, aux_register);

#define BROADCAST_PARAM(func, vector, aux_register, param_args_offset) \
    mov(aux_register, ptr[r64_params + (param_args_offset)]);          \
    func(vector, ptr[aux_register]);

#define INIT_ARR(A, V, R, T)                                                                 \
    static const T A[8] = {V, V, V, V, V, V, V, V};                                          \
    if (isa == x64::avx2) {                                                                  \
        mov(R, reinterpret_cast<uintptr_t>(A));                                              \
    } else {                                                                                 \
        static const T* A##_aligned = (A) + (reinterpret_cast<int64_t>(A) % 16) / sizeof(T); \
        mov(R, reinterpret_cast<uintptr_t>(A##_aligned));                                    \
    }

union FloatAsBits {
    float f;
    uint32_t u;
};

////////////// PHILOX GENERATOR /////////////////////////

template <x64::cpu_isa_t isa>
PhiloxGenerator<isa>::PhiloxGenerator(const PhiloxGeneratorCompileParams& jcp) : JitKernel(jit_name(), jcp, isa) {}

template <x64::cpu_isa_t isa>
void PhiloxGenerator<isa>::generate() {
    this->preamble();
    registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

    r64_dst = getReg64();
    r64_work_amount = getReg64();

    mov(r64_work_amount, ptr[r64_params + GET_PHILOX_OFFSET(work_amount)]);
    mov(r64_dst, ptr[r64_params + GET_PHILOX_OFFSET(dst_ptr)]);

    initVectors();
    process();

    registersPool.reset();
    this->postamble();
}

template <>
void PhiloxGenerator<x64::avx512_core>::initVectors() {
    const auto r64_aux = getReg64();
    const auto r32_aux = Xbyak::Reg32(r64_aux.getIdx());
    const auto r16_aux = Xbyak::Reg16(r64_aux.getIdx());

    v_max_mul_n_64 = getVmm();
    v_max_mul_c_64 = getVmm();
    v_add_low_k = getVmm();
    v_add_up_k = getVmm();
    v_n_inc = getVmm();
    v_range = getVmm();
    v_min = getVmm();
    v_key_64 = getVmm();
    v_counter_64 = getVmm();
    v_n_64 = getVmm();
    v_res_perm = getVmm();

    if (m_jcp.out_data_type.is_real()) {
        v_convert_0 = getVmm();
        v_convert_1 = getVmm();
    }

    // Initialize constants
    BROADCAST_CONSTANT(vpbroadcastq, v_max_mul_n_64, r64_aux, STATISTIC_MAXIMIZING_MULTIPLIER_N)
    BROADCAST_CONSTANT(vpbroadcastq, v_max_mul_c_64, r64_aux, STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER)
    BROADCAST_CONSTANT(vpbroadcastd, v_add_low_k, r32_aux, CRUSH_RESISTANCE_CONST_LOWER_VALUE)
    BROADCAST_CONSTANT(vpbroadcastd, v_add_up_k, r32_aux, CRUSH_RESISTANCE_CONST_UPPER_VALUE)
    BROADCAST_CONSTANT(vpbroadcastq, v_n_inc, r64_aux, 0x00000008)

    if (m_jcp.out_data_type == element::f32) {
        BROADCAST_CONSTANT(vpbroadcastd, v_convert_0, r32_aux, 0x3f800000)
        BROADCAST_CONSTANT(vpbroadcastd, v_convert_1, r32_aux, 0x007fffff)
        BROADCAST_PARAM(vpbroadcastd, v_range, r64_aux, GET_PHILOX_OFFSET(range_ptr))
        BROADCAST_PARAM(vpbroadcastd, v_min, r64_aux, GET_PHILOX_OFFSET(min_ptr))
    } else if (m_jcp.out_data_type == element::f16 && x64::mayiuse(x64::avx512_core_fp16)) {
        BROADCAST_CONSTANT(vpbroadcastw, v_convert_0, r16_aux, 0x3c00)
        BROADCAST_CONSTANT(vpbroadcastw, v_convert_1, r16_aux, 0x03ff)
        BROADCAST_PARAM(vpbroadcastw, v_range, r64_aux, GET_PHILOX_OFFSET(range_ptr))
        BROADCAST_PARAM(vpbroadcastw, v_min, r64_aux, GET_PHILOX_OFFSET(min_ptr))
    } else if (m_jcp.out_data_type == element::bf16 && x64::mayiuse(x64::avx512_core_bf16)) {
        v_convert_2 = getVmm();
        const auto ymm_min = Xbyak::Ymm(v_min.getIdx());
        const auto ymm_range = Xbyak::Ymm(v_range.getIdx());

        BROADCAST_CONSTANT(vpbroadcastw, v_convert_0, r16_aux, 0x3f80)
        BROADCAST_CONSTANT(vpbroadcastw, v_convert_1, r16_aux, 0x007f)
        BROADCAST_CONSTANT(vpbroadcastd, v_convert_2, r32_aux, 0x3f800000)

        BROADCAST_PARAM(vpbroadcastw, v_range, r64_aux, GET_PHILOX_OFFSET(range_ptr))
        vpmovzxwd(v_range, ymm_range);
        uni_vpslld(v_range, v_range, 16);

        BROADCAST_PARAM(vpbroadcastw, v_min, r64_aux, GET_PHILOX_OFFSET(min_ptr))
        vpmovzxwd(v_min, ymm_min);
        uni_vpslld(v_min, v_min, 16);
    } else if (m_jcp.out_data_type == element::i32) {
        const auto ymm_range = Xbyak::Ymm(v_range.getIdx());

        BROADCAST_PARAM(vpbroadcastd, v_range, r64_aux, GET_PHILOX_OFFSET(range_ptr))
        BROADCAST_PARAM(vpbroadcastd, v_min, r64_aux, GET_PHILOX_OFFSET(min_ptr))

        uni_vcvtdq2pd(v_range, ymm_range);
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    }

    // Initialize inputs.
    BROADCAST_PARAM(vpbroadcastq, v_key_64, r64_aux, GET_PHILOX_OFFSET(key_ptr))
    BROADCAST_PARAM(vpbroadcastq, v_counter_64, r64_aux, GET_PHILOX_OFFSET(counter_ptr))
    BROADCAST_PARAM(vpbroadcastq, v_n_64, r64_aux, GET_PHILOX_OFFSET(n_ptr))

    if (m_jcp.out_data_type.size() <= 4) {
        static const uint64_t n_inc_arr[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr));
    } else {
        static const uint64_t n_inc_arr[8] = {0, 1, 2, 3, 4, 5, 6, 7};  // TODO: i64
        mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr));
    }
    uni_vpaddq(v_n_64, v_n_64, ptr[r64_aux]);

    // Initialize auxiliary vectors.
    static const uint32_t res_perm_mask[16] = {0b00000000,
                                               0b00010000,
                                               0b00001000,
                                               0b00011000,
                                               0b00000010,
                                               0b00010010,
                                               0b00001010,
                                               0b00011010,
                                               0b00000100,
                                               0b00010100,
                                               0b00001100,
                                               0b00011100,
                                               0b00000110,
                                               0b00010110,
                                               0b00001110,
                                               0b00011110};
    mov(r64_aux, reinterpret_cast<uintptr_t>(res_perm_mask));
    uni_vmovups(v_res_perm, ptr[r64_aux]);

    if (m_jcp.out_data_type == element::f16 && x64::mayiuse(x64::avx512_core_fp16)) {
        v_perm_16 = getVmm();
        static const uint16_t perm_16[32] = {
            0b00000000, 0b00000010, 0b00000100, 0b00000110, 0b00001000, 0b00001010, 0b00001100, 0b00001110,
            0b00010000, 0b00010010, 0b00010100, 0b00010110, 0b00011000, 0b00011010, 0b00011100, 0b00011110,
            0b00100000, 0b00100010, 0b00100100, 0b00100110, 0b00101000, 0b00101010, 0b00101100, 0b00101110,
            0b00110000, 0b00110010, 0b00110100, 0b00110110, 0b00111000, 0b00111010, 0b00111100, 0b00111110};
        mov(r64_aux, reinterpret_cast<uintptr_t>(perm_16));
        uni_vmovups(v_perm_16, ptr[r64_aux]);
    }
}

template <x64::cpu_isa_t isa>  // Works for AVX2, SSE41
void PhiloxGenerator<isa>::initVectors() {
    const auto r64_aux = getReg64();

    v_max_mul_n_64 = getVmm();
    v_max_mul_c_64 = getVmm();
    v_add_low_k = getVmm();
    v_add_up_k = getVmm();
    v_range = getVmm();
    v_key_64 = getVmm();
    v_counter_64 = getVmm();
    v_n_64 = getVmm();

    r64_n_inc = getReg64();
    r64_min = getReg64();

    // Initialize constants.
    INIT_ARR(max_mul_n_64, STATISTIC_MAXIMIZING_MULTIPLIER_N, r64_aux, uint64_t);
    uni_vmovups(v_max_mul_n_64, ptr[r64_aux]);

    INIT_ARR(max_mul_c_64, STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER, r64_aux, uint64_t);
    uni_vmovups(v_max_mul_c_64, ptr[r64_aux]);

    INIT_ARR(add_low_k, CRUSH_RESISTANCE_CONST_LOWER_VALUE, r64_aux, uint32_t);
    uni_vmovups(v_add_low_k, ptr[r64_aux]);

    INIT_ARR(add_up_k, CRUSH_RESISTANCE_CONST_UPPER_VALUE, r64_aux, uint32_t);
    uni_vmovups(v_add_up_k, ptr[r64_aux]);

    INIT_ARR(n_inc_step, isa == x64::avx2 ? 4 : 2, r64_n_inc, uint64_t);

    if (m_jcp.out_data_type == element::f32) {
        r64_convert_0 = getReg64();
        r64_convert_1 = getReg64();

        INIT_ARR(convert_0, 0x3f800000, r64_convert_0, uint32_t);
        INIT_ARR(convert_1, 0x007fffff, r64_convert_1, uint32_t);

        mov(r64_aux, ptr[r64_params + GET_PHILOX_OFFSET(range_ptr)]);
        uni_vpbroadcastd(v_range, ptr[r64_aux]);

        auto v_aux = getVmm();
        mov(r64_aux, ptr[r64_params + GET_PHILOX_OFFSET(min_ptr)]);
        uni_vpbroadcastd(v_aux, ptr[r64_aux]);
        static uint32_t min_arr[8];
        mov(r64_min, reinterpret_cast<uintptr_t>(min_arr));
        uni_vmovups(ptr[r64_min], v_aux);
    } else if (m_jcp.out_data_type == element::i32) {
        r64_f64_pow_52 = getReg64();
        const auto v_aux = getVmm();
        const auto xmm_range = Xbyak::Xmm(v_range.getIdx());

        INIT_ARR(f64_pow_52, 0x4330000000000000, r64_f64_pow_52, uint64_t);

        mov(r64_aux, ptr[r64_params + GET_PHILOX_OFFSET(range_ptr)]);
        uni_vpbroadcastd(v_range, ptr[r64_aux]);

        mov(r64_aux, ptr[r64_params + GET_PHILOX_OFFSET(min_ptr)]);
        uni_vpbroadcastd(v_aux, ptr[r64_aux]);
        static uint32_t min_arr[8];
        mov(r64_min, reinterpret_cast<uintptr_t>(min_arr));
        uni_vmovups(ptr[r64_min], v_aux);

        uni_vcvtdq2pd(v_range, xmm_range);
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    }

    // Initialize inputs.
    mov(r64_aux, ptr[r64_params + GET_PHILOX_OFFSET(key_ptr)]);
    uni_vpbroadcastq(v_key_64, ptr[r64_aux]);

    mov(r64_aux, ptr[r64_params + GET_PHILOX_OFFSET(counter_ptr)]);
    uni_vpbroadcastq(v_counter_64, ptr[r64_aux]);

    mov(r64_aux, ptr[r64_params + GET_PHILOX_OFFSET(n_ptr)]);
    uni_vpbroadcastq(v_n_64, ptr[r64_aux]);

    if (m_jcp.out_data_type.size() <= 4) {
        if (isa == x64::avx2) {
            static const uint64_t n_inc_arr[4] = {0, 1, 2, 3};
            mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr));
        } else {
            static uint64_t n_inc_arr[4];
            static uint64_t* n_inc_arr_aligned =
                n_inc_arr + (reinterpret_cast<int64_t>(n_inc_arr) % 16) / sizeof(uint64_t);
            n_inc_arr_aligned[0] = 0;
            n_inc_arr_aligned[1] = 1;
            mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr_aligned));
        }
    } else {
        static const uint64_t n_inc_arr[4] = {0, 1, 2, 3};  // TODO: i64
        mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr));
    }

    uni_vpaddq(v_n_64, v_n_64, ptr[r64_aux]);
}

template <x64::cpu_isa_t isa>
void PhiloxGenerator<isa>::process() {
    auto v_dst_0 = getVmm();
    auto v_dst_1 = getVmm();
    std::vector<Vmm> v_res{v_dst_0, v_dst_1};

    auto step = vlen;
    if (one_of(m_jcp.out_data_type.size(), 2lu, 4lu)) {
        step = vlen * 2 / sizeof(uint32_t);
    } else if (m_jcp.out_data_type.size() == 8) {
        step = vlen / sizeof(uint32_t);
    }

    Xbyak::Label l_loop, l_tail;
    L(l_loop);
    {
        cmp(r64_work_amount, step);
        jl(l_tail, T_NEAR);

        runPhilox(v_res, v_key_64, v_counter_64, v_n_64);
        convert(v_res, v_res);

        uni_vmovups(ptr[r64_dst], v_dst_0);
        add(r64_dst, vlen);
        if (one_of(m_jcp.out_data_type.size(), 4lu, 8lu)) {
            uni_vmovups(ptr[r64_dst], v_dst_1);
            add(r64_dst, vlen);
        }

        if (isa == x64::avx512_core) {
            uni_vpaddd(v_n_64, v_n_64, v_n_inc);
        } else {
            uni_vpaddd(v_n_64, v_n_64, ptr[r64_n_inc]);
        }

        sub(r64_work_amount, step);
        jmp(l_loop, T_NEAR);
    }

    L(l_tail);
    tail(v_res);
}

template <x64::cpu_isa_t isa>
void PhiloxGenerator<isa>::calculateRound(const Vmm& vmm_k_0,
                                          const Vmm& vmm_k_1,
                                          const Vmm& vmm_c_0,
                                          const Vmm& vmm_c_1,
                                          const Vmm& vmm_n_0,
                                          const Vmm& vmm_n_1,
                                          const Vmm& vmm_aux_0,
                                          const Vmm& vmm_aux_1) {
    uni_vpmuludq(vmm_aux_0, vmm_n_0, v_max_mul_n_64);  // {p0,p1,p0,p1} = {n0,_,n0,_} * {m0,_,m0,_}
    uni_vpmuludq(vmm_aux_1, vmm_c_0, v_max_mul_c_64);  // {r0,r1,r0,r1} = {c0,_,c0,_} * {m0,_,m0,_}

    uni_vpshufd(vmm_c_0, vmm_aux_0, 0b10110001);  // {p1,p0,p1,p0} = shuf {p0,p1,p0,p1}
    uni_vxorps(vmm_c_0, vmm_c_0, vmm_c_1);        // {c0,_,c0,_} = {p1,_,p1,_} ^ {c1,_,c1,_}
    uni_vxorps(vmm_c_0, vmm_c_0, vmm_k_1);        // {c0,_,c0,_} = {c0,_,c0,_} ^ {k1,_,k1,_}

    uni_vpshufd(vmm_n_0, vmm_aux_1, 0b10110001);  // {r1,r0,r1,r0} = shuf {r0,r1,r0,r1}
    uni_vxorps(vmm_n_0, vmm_n_0, vmm_n_1);        // {n0,_,n0,_} = {r1,_,r1,_} ^ {n1,_,n1,_}
    uni_vxorps(vmm_n_0, vmm_n_0, vmm_k_0);        // {n0,_,n0,_} = {n0,_,n0,_} ^ {k0,_,k0,_}
}

template <x64::cpu_isa_t isa>
void PhiloxGenerator<isa>::runPhilox(const std::vector<Vmm>& vmm_dst,
                                     const Vmm& vmm_key,
                                     const Vmm& vmm_counter,
                                     const Vmm& vmm_n) {
    auto vmm_k_0 = getVmm();
    auto vmm_k_1 = getVmm();
    auto vmm_n_0 = getVmm();
    auto vmm_n_1 = vmm_dst[0];
    auto vmm_c_0 = getVmm();
    auto vmm_c_1 = getVmm();
    auto vmm_aux_0 = getVmm();
    auto vmm_aux_1 = vmm_dst[1];

    uni_vmovups(vmm_k_0, vmm_key);              // {k0,k1,k0,k1} -> {k0,_,k0,_}
    uni_vpshufd(vmm_k_1, vmm_key, 0b10110001);  // {k0,k1,k0,k1} -> {k1,_,k1,_}

    uni_vpmuludq(vmm_aux_0, vmm_n, v_max_mul_n_64);        // {p0,p1,p0,p1} = {n0,_,n0,_} * {m0,_,m0,_}
    uni_vpmuludq(vmm_aux_1, vmm_counter, v_max_mul_c_64);  // {r0,r1,r0,r1} = {c0,_,c0,_} * {m0,_,m0,_}

    uni_vxorps(vmm_c_0, vmm_aux_0, vmm_counter);  // {_,c0,_,c0} = {_,p1,_,p1} ^ {_,c1,_,c1}
    uni_vxorps(vmm_c_0, vmm_c_0, vmm_key);        // {_,c0,_,c0} = {_,c0,_,c0} ^ {_,k1,_,k1}
    uni_vpshufd(vmm_c_0, vmm_c_0, 0b10110001);    // {_,c0,_,c0} -> {c0,_,c0,_}

    uni_vxorps(vmm_n_0, vmm_aux_1, vmm_n);      // {_,n0,_,n0} = {_,r1,_,r1} ^ {_,n1,_,n1}
    uni_vpshufd(vmm_n_0, vmm_n_0, 0b10110001);  // {_,n0,_,n0} -> {n0,_,n0,_}
    uni_vxorps(vmm_n_0, vmm_n_0, vmm_key);      // {n0,_,n0,_} = {n0,_,n0,_} ^ {k0,_,k0,_}

    for (size_t i = 0lu; i < ROUNDS_NUMBER - 1; i++) {
        raiseKey(vmm_k_0, vmm_k_1);

        std::swap(vmm_c_1, vmm_aux_0);
        std::swap(vmm_n_1, vmm_aux_1);
        calculateRound(vmm_k_0, vmm_k_1, vmm_c_0, vmm_c_1, vmm_n_0, vmm_n_1, vmm_aux_0, vmm_aux_1);
    }
    std::swap(vmm_c_1, vmm_aux_0);
    std::swap(vmm_n_1, vmm_aux_1);

    if (isa == x64::avx512_core) {
        vpermt2d(vmm_n_0, v_res_perm, vmm_n_1);             // {n0,n1,n0,n1} = perm {n0,_,n0,_} {n1,_,n1,_}
        vpermt2d(vmm_c_0, v_res_perm, vmm_c_1);             // {c0,c1,c0,c1} = perm {c0,_,c0,_} {c1,_,c1,_}
        vshufpd(vmm_dst[0], vmm_n_0, vmm_c_0, 0b00000000);  // {n0,n1,c0,c1} = shuf {n0,n1,n0,n1} {c0,c1,c0,c1}
        vshufpd(vmm_dst[1], vmm_n_0, vmm_c_0, 0b11111111);  // {n0,n1,c0,c1} = shuf {n0,n1,n0,n1} {c0,c1,c0,c1}
    } else if (isa == x64::avx2) {
        auto ymm_dst_0 = Xbyak::Ymm(vmm_dst[0].getIdx());
        auto ymm_dst_1 = Xbyak::Ymm(vmm_dst[1].getIdx());
        auto ymm_c_0 = Xbyak::Ymm(vmm_c_0.getIdx());

        uni_vshufps(vmm_n_0, vmm_n_0, vmm_n_1, 0b10001000);    // {n0,n0,n1,n1} = shuf {n0,_,n0,_} {n1,_,n1,_}
        uni_vshufps(vmm_c_0, vmm_c_0, vmm_c_1, 0b10001000);    // {c0,c0,c1,c1} = shuf {c0,_,c0,_} {c1,_,c1,_}
        uni_vshufps(ymm_dst_1, vmm_n_0, vmm_c_0, 0b10001000);  // {n0,n1,c0,c1} = shuf {n0,n0,n1,n1} {c0,c0,c1,c1}
        uni_vshufps(vmm_c_0, vmm_n_0, vmm_c_0, 0b11011101);    // {n0,n1,c0,c1} = shuf {n0,n0,n1,n1} {c0,c0,c1,c1}
        vperm2f128(ymm_dst_0, ymm_dst_1, ymm_c_0, 0b00100000);
        vperm2f128(ymm_dst_1, ymm_dst_1, ymm_c_0, 0b00110001);
    } else {
        uni_vshufps(vmm_n_0, vmm_n_0, vmm_n_1, 0b10001000);
        uni_vshufps(vmm_c_0, vmm_c_0, vmm_c_1, 0b10001000);
        uni_vshufps(vmm_dst[0], vmm_n_0, vmm_c_0, 0b10001000);
        uni_vshufps(vmm_dst[1], vmm_n_0, vmm_c_0, 0b11011101);
    }
}

template <x64::cpu_isa_t isa>
void PhiloxGenerator<isa>::raiseKey(const Vmm& vmm_k_0, const Vmm& vmm_k_1) {
    uni_vpaddd(vmm_k_0, vmm_k_0, v_add_low_k);  // {k0,_,k0,_} + {l0,_,l0,_}
    uni_vpaddd(vmm_k_1, vmm_k_1, v_add_up_k);   // {k1,_,k1,_} + {u0,_,u0,_}
}

template <>
void PhiloxGenerator<x64::avx512_core>::convert(const std::vector<Vmm>& v_dst, const std::vector<Vmm>& v_src) {
    if (m_jcp.out_data_type.size() == 4) {
        for (size_t i = 0lu; i < v_src.size(); i++) {
            const auto& vmm_src = v_src[i];
            const auto& vmm_dst = v_dst[i];

            if (m_jcp.out_data_type == element::f32) {
                uni_vandps(vmm_dst, vmm_src, v_convert_1);
                uni_vorps(vmm_dst, vmm_dst, v_convert_0);
                uni_vsubps(vmm_dst, vmm_dst, v_convert_0);
                vfmadd132ps(vmm_dst, v_min, v_range);
            } else if (m_jcp.out_data_type == element::i32) {
                // x % (max - min) + min
                const auto v_aux_0 = getVmm();
                const auto v_aux_1 = getVmm();
                const auto ymm_src = Xbyak::Ymm(vmm_src.getIdx());
                const auto ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
                const auto ymm_aux_1 = Xbyak::Ymm(v_aux_1.getIdx());

                // Divide in the f64 due to the f32 loses accuracy here.
                vcvtudq2pd(v_aux_0, ymm_src);
                uni_vdivpd(v_aux_1, v_aux_0, v_range);
                uni_vroundpd(v_aux_1, v_aux_1, 3);
                vfnmadd132pd(v_aux_1, v_aux_0, v_range);

                vextractf64x4(ymm_dst, vmm_src, 1);
                vcvtudq2pd(v_aux_0, ymm_dst);
                uni_vcvtpd2dq(ymm_dst, v_aux_1);
                uni_vdivpd(v_aux_1, v_aux_0, v_range);
                uni_vroundpd(v_aux_1, v_aux_1, 3);
                vfnmadd132pd(v_aux_1, v_aux_0, v_range);
                uni_vcvtpd2dq(ymm_aux_1, v_aux_1);
                vshuff64x2(vmm_dst, vmm_dst, v_aux_1, 0b01000100);

                uni_vpaddd(vmm_dst, vmm_dst, v_min);
            } else {
                OPENVINO_THROW("RandomUniform kernel does not support precision ",
                               m_jcp.out_data_type,
                               " for ",
                               x64::get_isa_info());
            }
        }
    } else if (m_jcp.out_data_type.size() == 2) {
        if (m_jcp.out_data_type == element::f16 && x64::mayiuse(x64::avx512_core_fp16)) {
            const auto& vmm_dst = v_dst[0];

            if (v_src[0].getIdx() != vmm_dst.getIdx()) {
                uni_vmovups(vmm_dst, v_src[0]);
            }
            vpermt2w(vmm_dst, v_perm_16, v_src[1]);

            uni_vandps(vmm_dst, vmm_dst, v_convert_1);
            uni_vorps(vmm_dst, vmm_dst, v_convert_0);
            vsubph(vmm_dst, vmm_dst, v_convert_0);
            vfmadd132ph(vmm_dst, v_min, v_range);
        } else if (m_jcp.out_data_type == element::bf16 && x64::mayiuse(x64::avx512_core_bf16)) {
            for (size_t i = 0lu; i < v_src.size(); i++) {
                const auto& vmm_dst = v_dst[i];

                uni_vandps(vmm_dst, v_src[i], v_convert_1);
                uni_vorps(vmm_dst, vmm_dst, v_convert_0);
                uni_vpslld(vmm_dst, vmm_dst, 16);

                uni_vsubps(vmm_dst, vmm_dst, v_convert_2);
                vfmadd132ps(vmm_dst, v_min, v_range);
            }

            vcvtne2ps2bf16(v_dst[0], v_dst[1], v_dst[0]);
        } else {
            OPENVINO_THROW("RandomUniform kernel does not support precision ",
                           m_jcp.out_data_type,
                           " for ",
                           x64::get_isa_info());
        }
    } else if (m_jcp.out_data_type.size() == 8) {
        if (m_jcp.out_data_type == element::i64) {
            // TODO: in scope of i64 enabling.
        }
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    }
}

template <x64::cpu_isa_t isa>  // Works for AVX2, SSE41
void PhiloxGenerator<isa>::convert(const std::vector<Vmm>& v_dst, const std::vector<Vmm>& v_src) {
    if (m_jcp.out_data_type.size() == 4) {
        for (size_t i = 0lu; i < v_src.size(); i++) {
            auto vmm_src = v_src[i];
            auto vmm_dst = v_dst[i];

            if (m_jcp.out_data_type == element::f32) {
                uni_vandps(vmm_dst, vmm_src, ptr[r64_convert_1]);
                uni_vorps(vmm_dst, vmm_dst, ptr[r64_convert_0]);
                uni_vsubps(vmm_dst, vmm_dst, ptr[r64_convert_0]);
                if (isa == x64::avx2) {
                    vfmadd213ps(vmm_dst, v_range, ptr[r64_min]);
                } else {
                    uni_vmulps(vmm_dst, vmm_dst, v_range);
                    uni_vaddps(vmm_dst, vmm_dst, ptr[r64_min]);
                }
            } else if (m_jcp.out_data_type == element::i32) {
                // x % (max - min) + min
                const auto v_aux_0 = getVmm();
                const auto v_aux_1 = getVmm();
                const auto xmm_dst = Xbyak::Xmm(vmm_dst.getIdx());
                const auto ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
                const auto xmm_aux_1 = Xbyak::Xmm(v_aux_1.getIdx());

                // Convert u32->f64. TODO: move to convert emitter after i64 enabling.
                uni_vpmovzxdq(v_aux_0, xmm_dst);
                uni_vorps(v_aux_0, v_aux_0, ptr[r64_f64_pow_52]);
                uni_vsubpd(v_aux_0, v_aux_0, ptr[r64_f64_pow_52]);

                // Divide in the f64 due to the f32 loses accuracy here.
                uni_vdivpd(v_aux_1, v_aux_0, v_range);
                uni_vroundpd(v_aux_1, v_aux_1, 3);
                if (isa == x64::avx2) {
                    vfnmadd132pd(v_aux_1, v_aux_0, v_range);
                } else {
                    uni_vmulpd(v_aux_1, v_aux_1, v_range);
                    uni_vsubpd(v_aux_0, v_aux_0, v_aux_1);
                    uni_vmovups(v_aux_1, v_aux_0);
                }

                if (isa == x64::avx2) {
                    vperm2f128(ymm_dst, ymm_dst, ymm_dst, 0b00000001);
                } else {
                    uni_vshufpd(vmm_dst, vmm_dst, vmm_dst, 0b00000001);
                }
                // Convert u32->f64. TODO: move to convert emitter after i64 enabling.
                uni_vpmovzxdq(v_aux_0, xmm_dst);
                uni_vorps(v_aux_0, v_aux_0, ptr[r64_f64_pow_52]);
                uni_vsubpd(v_aux_0, v_aux_0, ptr[r64_f64_pow_52]);

                uni_vcvtpd2dq(xmm_dst, v_aux_1);
                uni_vdivpd(v_aux_1, v_aux_0, v_range);
                uni_vroundpd(v_aux_1, v_aux_1, 3);
                if (isa == x64::avx2) {
                    vfnmadd132pd(v_aux_1, v_aux_0, v_range);
                } else {
                    uni_vmulpd(v_aux_1, v_aux_1, v_range);
                    uni_vsubpd(v_aux_0, v_aux_0, v_aux_1);
                    uni_vmovups(v_aux_1, v_aux_0);
                }
                uni_vcvtpd2dq(xmm_aux_1, v_aux_1);
                if (isa == x64::avx2) {
                    vperm2f128(ymm_dst, ymm_dst, v_aux_1, 0b00100000);
                } else {
                    uni_vshufpd(vmm_dst, vmm_dst, v_aux_1, 0b00000000);
                }

                uni_vpaddd(vmm_dst, vmm_dst, ptr[r64_min]);
            } else {
                OPENVINO_THROW("RandomUniform kernel does not support precision ",
                               m_jcp.out_data_type,
                               " for ",
                               x64::get_isa_info());
            }
        }
    } else if (m_jcp.out_data_type.size() == 8) {
        if (m_jcp.out_data_type == element::i64) {
            // TODO: in scope of i64 enabling.
        }
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    }
}

template <>
void PhiloxGenerator<x64::avx512_core>::tail(const std::vector<Vmm>& vmm_dst) {
    Xbyak::Label l_end;
    const auto k_rest_mask = getMask();

    cmp(r64_work_amount, 0);
    jle(l_end, T_NEAR);

    runPhilox(vmm_dst, v_key_64, v_counter_64, v_n_64);
    convert(vmm_dst, vmm_dst);

    if (m_jcp.out_data_type.size() == 4) {
        Xbyak::Label l_0;
        const auto step = vlen / sizeof(uint32_t);

        cmp(r64_work_amount, step);
        jl(l_0, T_NEAR);

        uni_vmovups(ptr[r64_dst], vmm_dst[0]);
        add(r64_dst, vlen);
        sub(r64_work_amount, step);
        fillRestWorkMask(k_rest_mask, r64_work_amount);
        uni_vmovups(ptr[r64_dst] | k_rest_mask, vmm_dst[1]);
        jmp(l_end, T_NEAR);

        L(l_0);
        fillRestWorkMask(k_rest_mask, r64_work_amount);
        uni_vmovups(ptr[r64_dst] | k_rest_mask, vmm_dst[0]);
    } else if (m_jcp.out_data_type.size() == 2) {
        fillRestWorkMask(k_rest_mask, r64_work_amount);
        vmovdqu16(ptr[r64_dst] | k_rest_mask, vmm_dst[0]);
    }

    L(l_end);
}

template <>
void PhiloxGenerator<x64::avx2>::tail(const std::vector<Vmm>& vmm_dst) {
    Xbyak::Label l_0, l_end;
    const auto step = vlen / sizeof(uint32_t);

    cmp(r64_work_amount, 0);
    jle(l_end, T_NEAR);

    runPhilox(vmm_dst, v_key_64, v_counter_64, v_n_64);
    convert(vmm_dst, vmm_dst);
    const auto v_rest_mask = getVmm();

    cmp(r64_work_amount, step);
    jl(l_0, T_NEAR);

    uni_vmovups(ptr[r64_dst], vmm_dst[0]);
    add(r64_dst, vlen);
    sub(r64_work_amount, step);
    fillRestWorkMask(v_rest_mask, r64_work_amount, m_jcp.out_data_type.size());
    vmaskmovps(ptr[r64_dst], v_rest_mask, vmm_dst[1]);
    jmp(l_end, T_NEAR);

    L(l_0);
    fillRestWorkMask(v_rest_mask, r64_work_amount, m_jcp.out_data_type.size());
    vmaskmovps(ptr[r64_dst], v_rest_mask, vmm_dst[0]);

    L(l_end);
}

template <x64::cpu_isa_t isa>
void PhiloxGenerator<isa>::tail(const std::vector<Vmm>& vmm_dst) {
    Xbyak::Label l_0, l_end;
    const auto step = vlen / sizeof(uint32_t);

    cmp(r64_work_amount, 0);
    jle(l_end, T_NEAR);

    runPhilox(vmm_dst, v_key_64, v_counter_64, v_n_64);
    convert(vmm_dst, vmm_dst);

    cmp(r64_work_amount, step);
    jl(l_0, T_NEAR);

    uni_vmovups(ptr[r64_dst], vmm_dst[0]);
    add(r64_dst, vlen);
    sub(r64_work_amount, step);
    store(ptr[r64_dst], vmm_dst[1], r64_work_amount, m_jcp.out_data_type.size());
    jmp(l_end, T_NEAR);

    L(l_0);
    store(ptr[r64_dst], vmm_dst[0], r64_work_amount, m_jcp.out_data_type.size());

    L(l_end);
}

//////////////// MERSENNE TWISTER GENERATOR ////////////////////

template <x64::cpu_isa_t isa>
MersenneTwisterGenerator<isa>::MersenneTwisterGenerator(const MersenneTwisterGeneratorCompileParams& jcp)
    : JitKernel(jit_name(), jcp, isa) {}

template <x64::cpu_isa_t isa>
void MersenneTwisterGenerator<isa>::generate() {
    this->preamble();
    registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

    r64_dst = getReg64();
    r64_state = getReg64();
    r64_output_idx = getReg64();
    r64_max_output_idx = getReg64();
    r64_storage_capacity = getReg64();
    r64_elements_to_generate = getReg64();
    r64_state_accesses_count = getReg64();

    mov(r64_dst, ptr[r64_params + GET_MERSENNE_OFFSET(dst_ptr)]);
    mov(r64_state, ptr[r64_params + GET_MERSENNE_OFFSET(state_ptr)]);
    mov(r64_output_idx, ptr[r64_params + GET_MERSENNE_OFFSET(output_idx)]);
    mov(r64_max_output_idx, ptr[r64_params + GET_MERSENNE_OFFSET(max_output_idx)]);
    mov(r64_elements_to_generate, ptr[r64_params + GET_MERSENNE_OFFSET(elements_to_generate)]);
    mov(r64_state_accesses_count, ptr[r64_params + GET_MERSENNE_OFFSET(state_accesses_count)]);
    mov(r64_storage_capacity, static_cast<uint64_t>(vlen / sizeof(uint32_t)));

    initVectors();
    process();

    registersPool.reset();
    this->postamble();
}

template <>
void MersenneTwisterGenerator<x64::avx512_core>::initVectors() {
    const auto r64_aux = getReg64();
    const auto r32_aux = Xbyak::Reg32(r64_aux.getIdx());

    v_min = getVmm();
    v_range = getVmm();
    v_state = getVmm();
    v_result = getVmm();
    v_const_1 = getVmm();
    v_const_2 = getVmm();

    if (m_jcp.out_data_type.is_real()) {
        v_mask = getVmm();
        v_divisor = getVmm();
    }

    // Initialize state
    uni_vmovdqu(v_state, ptr[r64_state]);

    // Initialize constants
    BROADCAST_CONSTANT(vpbroadcastd, v_const_1, r32_aux, MT_CONST_1)
    BROADCAST_CONSTANT(vpbroadcastd, v_const_2, r32_aux, MT_CONST_2)

    // Initialize constants based on the requested data type
    if (m_jcp.out_data_type == element::f32) {
        FloatAsBits val;
        val.f = 1.0f / (1 << 24);
        BROADCAST_CONSTANT(vpbroadcastd, v_divisor, r32_aux, val.u);
        BROADCAST_CONSTANT(vpbroadcastd, v_mask, r32_aux, static_cast<uint32_t>((1 << 24) - 1))

        BROADCAST_PARAM(vpbroadcastd, v_range, r64_aux, GET_MERSENNE_OFFSET(range_ptr))
        BROADCAST_PARAM(vpbroadcastd, v_min, r64_aux, GET_MERSENNE_OFFSET(min_ptr))
    } else if (m_jcp.out_data_type == element::f16 && x64::mayiuse(x64::avx512_core_fp16)) {
        FloatAsBits val;
        val.f = 1.0f / (1 << 11);
        BROADCAST_CONSTANT(vpbroadcastd, v_divisor, r32_aux, val.u);
        BROADCAST_CONSTANT(vpbroadcastd, v_mask, r32_aux, static_cast<uint32_t>((1 << 11) - 1))

        // Note: two times too many values in Zmm
        BROADCAST_PARAM(vpbroadcastw, v_range, r64_aux, GET_MERSENNE_OFFSET(range_ptr))
        BROADCAST_PARAM(vpbroadcastw, v_min, r64_aux, GET_MERSENNE_OFFSET(min_ptr))
    } else if (m_jcp.out_data_type == element::bf16 && x64::mayiuse(x64::avx512_core_bf16)) {
        FloatAsBits val;
        val.f = 1.0f / (1 << 8);
        BROADCAST_CONSTANT(vpbroadcastd, v_divisor, r32_aux, val.u);
        BROADCAST_CONSTANT(vpbroadcastd, v_mask, r32_aux, static_cast<uint32_t>((1 << 8) - 1))

        // Note: two times too many values in Zmm
        BROADCAST_PARAM(vpbroadcastw, v_range, r64_aux, GET_MERSENNE_OFFSET(range_ptr))
        BROADCAST_PARAM(vpbroadcastw, v_min, r64_aux, GET_MERSENNE_OFFSET(min_ptr))
    } else if (m_jcp.out_data_type == element::i32) {
        BROADCAST_PARAM(vpbroadcastd, v_range, r64_aux, GET_MERSENNE_OFFSET(range_ptr))
        BROADCAST_PARAM(vpbroadcastd, v_min, r64_aux, GET_MERSENNE_OFFSET(min_ptr))
    } else if (m_jcp.out_data_type == element::i64) {
        // Same as in Philox - in scope of i64 enabling
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    }
}

template <x64::cpu_isa_t isa>  // Works for AVX2, SSE41
void MersenneTwisterGenerator<isa>::initVectors() {
    const auto r64_aux = getReg64();

    v_min = getVmm();
    v_range = getVmm();
    v_state = getVmm();
    v_result = getVmm();
    v_const_1 = getVmm();
    v_const_2 = getVmm();

    // Initialize state
    uni_vmovdqu(v_state, ptr[r64_state]);

    // Initialize constants.
    INIT_ARR(const_1, MT_CONST_1, r64_aux, uint32_t);
    uni_vmovdqu(v_const_1, ptr[r64_aux]);

    INIT_ARR(const_2, MT_CONST_2, r64_aux, uint32_t);
    uni_vmovdqu(v_const_2, ptr[r64_aux]);

    if (m_jcp.out_data_type == element::f32) {
        v_mask = getVmm();
        v_divisor = getVmm();

        INIT_ARR(mask, static_cast<uint32_t>((1 << 24) - 1), r64_aux, uint32_t);
        uni_vmovups(v_mask, ptr[r64_aux]);

        INIT_ARR(divisor, static_cast<float>(1.0f / (1 << 24)), r64_aux, float);
        uni_vmovups(v_divisor, ptr[r64_aux]);

        mov(r64_aux, ptr[r64_params + GET_MERSENNE_OFFSET(range_ptr)]);
        uni_vpbroadcastd(v_range, ptr[r64_aux]);

        mov(r64_aux, ptr[r64_params + GET_MERSENNE_OFFSET(min_ptr)]);
        uni_vpbroadcastd(v_min, ptr[r64_aux]);
    } else if (m_jcp.out_data_type == element::i32) {
        mov(r64_aux, ptr[r64_params + GET_MERSENNE_OFFSET(range_ptr)]);
        uni_vpbroadcastd(v_range, ptr[r64_aux]);

        mov(r64_aux, ptr[r64_params + GET_MERSENNE_OFFSET(min_ptr)]);
        uni_vpbroadcastd(v_min, ptr[r64_aux]);
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    }
}

template <x64::cpu_isa_t isa>
void MersenneTwisterGenerator<isa>::process() {
    // Generate random numbers
    generateRandomNumbers();

    // Convert to output type, store result
    convertToOutputTypeMersenne();

    // Store results
    storeResults();
}

template <>
void MersenneTwisterGenerator<x64::avx512_core>::generateRandomNumbers() {
    // Difference between this implementation and SSE41/AVX2 is
    // uni_vmovdqu -> vmovdqu32, uni fails with incorrect EVEX
    // Other functions work just fine
    const auto v_aux = getVmm();

    // Load values from memory, copy
    vmovdqu32(v_result, v_state);  // x = state

    // Apply Mersenne Twister transformations
    vmovdqu32(v_aux, v_result);  // tmp = x

    // x ^= (x >> 11);
    vpsrld(v_aux, v_aux, 11);           // tmp >>= 11
    vpxord(v_result, v_result, v_aux);  // x ^= tmp

    // x ^= (x << 7) & const_1;
    vmovdqu32(v_aux, v_result);         // tmp = x
    vpslld(v_aux, v_aux, 7);            // tmp <<= 7
    vpandd(v_aux, v_aux, v_const_1);    // tmp &= const_1
    vpxord(v_result, v_result, v_aux);  // x ^= tmp

    // x ^= (x << 15) & const_2;
    vmovdqu32(v_aux, v_result);         // tmp = x
    vpslld(v_aux, v_aux, 15);           // tmp <<= 15
    vpandd(v_aux, v_aux, v_const_2);    // tmp &= const_2
    vpxord(v_result, v_result, v_aux);  // x ^= tmp

    // x ^= (x >> 18);
    vmovdqu32(v_aux, v_result);         // tmp = x
    vpsrld(v_aux, v_aux, 18);           // tmp >>= 18
    vpxord(v_result, v_result, v_aux);  // x ^= tmp
}

template <x64::cpu_isa_t isa>  // Works for SSE41, AVX2
void MersenneTwisterGenerator<isa>::generateRandomNumbers() {
    const auto v_aux = getVmm();

    // Load values from memory, copy
    uni_vmovdqu(v_result, v_state);  // x = state

    // Apply Mersenne Twister transformations
    uni_vmovdqu(v_aux, v_result);  // tmp = x

    // x ^= (x >> 11);
    uni_vpsrld(v_aux, v_aux, 11);          // tmp >>= 11
    uni_vpxor(v_result, v_result, v_aux);  // x ^= tmp

    // x ^= (x << 7) & const_1;
    uni_vmovdqu(v_aux, v_result);          // tmp = x
    uni_vpslld(v_aux, v_aux, 7);           // tmp <<= 7
    uni_vpand(v_aux, v_aux, v_const_1);    // tmp &= const_1
    uni_vpxor(v_result, v_result, v_aux);  // x ^= tmp

    // x ^= (x << 15) & const_2;
    uni_vmovdqu(v_aux, v_result);          // tmp = x
    uni_vpslld(v_aux, v_aux, 15);          // tmp <<= 15
    uni_vpand(v_aux, v_aux, v_const_2);    // tmp &= const_2
    uni_vpxor(v_result, v_result, v_aux);  // x ^= tmp

    // x ^= (x >> 18);
    uni_vmovdqu(v_aux, v_result);          // tmp = x
    uni_vpsrld(v_aux, v_aux, 18);          // tmp >>= 18
    uni_vpxor(v_result, v_result, v_aux);  // x ^= tmp
}

template <>
void MersenneTwisterGenerator<x64::avx512_core>::convertToOutputTypeMersenne() {
    if (m_jcp.out_data_type == element::f32) {
        // Apply mask and divisor
        // No need to do int32's voodoo with double since mask ensures
        // that most significant bit is 0
        vpandd(v_result, v_result, v_mask);
        vcvtdq2ps(v_result, v_result);
        vmulps(v_result, v_result, v_divisor);

        // Scale and shift
        vmulps(v_result, v_result, v_range);
        vaddps(v_result, v_result, v_min);
    } else if (m_jcp.out_data_type == element::f16) {
        // Apply mask and divisor
        vpandd(v_result, v_result, v_mask);
        vcvtdq2ps(v_result, v_result);
        vmulps(v_result, v_result, v_divisor);

        vcvtps2ph(v_result, v_result, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // View only half as v_result shrunk from 32->16 conversion
        auto ymm_result = Xbyak::Ymm(v_result);
        auto ymm_range = Xbyak::Ymm(v_range);
        auto ymm_min = Xbyak::Ymm(v_min);

        // Scale and shift
        vmulph(ymm_result, ymm_result, ymm_range);
        vaddph(v_result, ymm_result, ymm_min);
    } else if (m_jcp.out_data_type == element::bf16) {
        // Apply mask and divisor
        vpandd(v_result, v_result, v_mask);
        vcvtdq2ps(v_result, v_result);
        vmulps(v_result, v_result, v_divisor);

        // Pseudoconvert to f32 by bitshift
        vpslld(v_range, v_range, 16);
        vpslld(v_min, v_min, 16);

        // Scale and shift
        vmulps(v_result, v_range);
        vaddps(v_result, v_min);

        vcvtneps2bf16(v_result, v_result);
    } else if (m_jcp.out_data_type == element::i32) {
        // Split result before converting 32 -> 64 to fit new bits
        const auto v_result_high_double = getVmm();
        const auto v_result_low_double = getVmm();
        const auto v_range_double = getVmm();

        const auto y_result_high_double = Xbyak::Ymm(v_result_high_double.getIdx());
        const auto y_result_low_double = Xbyak::Ymm(v_result_low_double.getIdx());
        const auto y_range_double = Xbyak::Ymm(v_range_double.getIdx());

        vextracti32x8(y_result_high_double, v_result, 1);
        vextracti32x8(y_result_low_double, v_result, 0);
        vextracti32x8(y_range_double, v_range, 0);

        // Extract the most significant bit (MSB) using bitshift
        const auto v_msb_high_double = getVmm();
        const auto v_msb_low_double = getVmm();
        const auto y_msb_high_double = Xbyak::Ymm(v_msb_high_double.getIdx());
        const auto y_msb_low_double = Xbyak::Ymm(v_msb_low_double.getIdx());

        vpsrld(y_msb_high_double, y_result_high_double, 31);
        vpsrld(y_msb_low_double, y_result_low_double, 31);

        // Remove most significant digit from result by bitshift
        // One left (removes msb)
        vpslld(y_result_high_double, y_result_high_double, 1);
        vpslld(y_result_low_double, y_result_low_double, 1);

        // One right (shifts back, sets 0 at the front)
        vpsrld(y_result_high_double, y_result_high_double, 1);
        vpsrld(y_result_low_double, y_result_low_double, 1);

        // Create a double value of 2^31 for the most significant digit instead of -1
        const auto r64_multiplier_double = getReg64();
        const auto v_multiplier_double = getVmm();

        mov(r64_multiplier_double, 0x41E0000000000000);  // 2^31 in IEEE 754 double format
        vpbroadcastq(v_multiplier_double, r64_multiplier_double);

        // Convert most significant digit to double (either 0 or 1)
        vcvtdq2pd(v_msb_high_double, y_msb_high_double);
        vcvtdq2pd(v_msb_low_double, y_msb_low_double);

        // Multiply (0/1) * 2^31
        vmulpd(v_msb_high_double, v_msb_high_double, v_multiplier_double);
        vmulpd(v_msb_low_double, v_msb_low_double, v_multiplier_double);

        // Convert uint32_t to double for accuracy
        vcvtdq2pd(v_result_high_double, y_result_high_double);
        vcvtdq2pd(v_result_low_double, y_result_low_double);
        vcvtdq2pd(v_range_double, y_range_double);

        // Add sign as 2^31 if was present, correctly converting uint32_t to double
        vaddpd(v_result_high_double, v_result_high_double, v_msb_high_double);
        vaddpd(v_result_low_double, v_result_low_double, v_msb_low_double);

        // Compute approximate division
        const auto v_aprox_result_high_double = getVmm();
        const auto v_aprox_result_low_double = getVmm();
        vdivpd(v_aprox_result_high_double, v_result_high_double, v_range_double);  // value / range = (aux = aux / aux2)
        vdivpd(v_aprox_result_low_double, v_result_low_double, v_range_double);    // value / range = (aux = aux / aux2)

        // Floor the result to nearest int (biggest multiple of divisor)
        vrndscalepd(v_aprox_result_high_double, v_aprox_result_high_double, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        vrndscalepd(v_aprox_result_low_double, v_aprox_result_low_double, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

        // Compute closest divisible value by multiplying back
        vmulpd(v_aprox_result_high_double,
               v_aprox_result_high_double,
               v_range_double);  // aux = floor(double(result) / double(range)) * double(range)
        vmulpd(v_aprox_result_low_double,
               v_aprox_result_low_double,
               v_range_double);  // aux = floor(double(result) / double(range)) * double(range)

        // Compute remainder by subtracting approximation from the original
        vsubpd(v_result_high_double, v_result_high_double, v_aprox_result_high_double);
        vsubpd(v_result_low_double, v_result_low_double, v_aprox_result_low_double);

        // Convert 64 -> 32, always possible as 0 < result < range
        vcvtpd2dq(y_result_high_double, v_result_high_double);  // value - closest_div_value = remainder (modulo)
        vcvtpd2dq(y_result_low_double, v_result_low_double);    // value - closest_div_value = remainder (modulo)

        // Concatenate them back, now result holds all remainders (modulos)
        vinserti32x8(v_result, v_result, y_result_high_double, 1);
        vinserti32x8(v_result, v_result, y_result_low_double, 0);

        // Add minimum
        vpaddd(v_result, v_result, v_min);  // remainder + min
    } else if (m_jcp.out_data_type == element::i64 && m_jcp.optimized) {
        // Same as in Philox - in scope of i64 enabling
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    } else if (m_jcp.out_data_type == element::i64 && !m_jcp.optimized) {
        // Same as in Philox - in scope of i64 enabling
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    }
}

template <>
void MersenneTwisterGenerator<x64::avx2>::convertToOutputTypeMersenne() {
    if (m_jcp.out_data_type == element::f32) {
        // Apply mask and divisor
        // No need to do int32's voodoo with double since mask ensures
        // that most significant bit is 0
        vpand(v_result, v_result, v_mask);
        vcvtdq2ps(v_result, v_result);
        vmulps(v_result, v_divisor);

        // Scale and shift
        vmulps(v_result, v_range);
        vaddps(v_result, v_min);
    } else if (m_jcp.out_data_type == element::i32) {
        // Split result before converting 32 -> 64 to fit new bits
        const auto v_result_high_double = getVmm();
        const auto v_result_low_double = getVmm();
        const auto v_range_double = getVmm();

        const auto x_result_high_double = Xbyak::Xmm(v_result_high_double.getIdx());
        const auto x_result_low_double = Xbyak::Xmm(v_result_low_double.getIdx());
        const auto x_range_double = Xbyak::Xmm(v_range_double.getIdx());

        vextracti128(x_result_high_double, v_result, 1);
        vextracti128(x_result_low_double, v_result, 0);
        vextracti128(x_range_double, v_range, 0);

        // Extract the most significant bit (MSB) using bitshift
        const auto v_msb_high_double = getVmm();
        const auto v_msb_low_double = getVmm();
        const auto x_msb_high_double = Xbyak::Xmm(v_msb_high_double.getIdx());
        const auto x_msb_low_double = Xbyak::Xmm(v_msb_low_double.getIdx());

        vpsrld(x_msb_high_double, x_result_high_double, 31);
        vpsrld(x_msb_low_double, x_result_low_double, 31);

        // Remove most significant digit from result by bitshift
        // One left (removes msb)
        vpslld(x_result_high_double, x_result_high_double, 1);
        vpslld(x_result_low_double, x_result_low_double, 1);

        // One right (shifts back, sets 0 at the front)
        vpsrld(x_result_high_double, x_result_high_double, 1);
        vpsrld(x_result_low_double, x_result_low_double, 1);

        // Create a double value of 2^31 for the most significant digit instead of -1
        const auto r64_multiplier_double = getReg64();
        const auto v_multiplier_double = getVmm();
        const auto x_multiplier_double = Xbyak::Xmm(v_multiplier_double.getIdx());

        mov(r64_multiplier_double, 0x41E0000000000000);  // 2^31 in IEEE 754 double format
        vmovq(x_multiplier_double, r64_multiplier_double);
        vbroadcastsd(v_multiplier_double, x_multiplier_double);

        // Convert most significant digit to double (either 0 or 1)
        vcvtdq2pd(v_msb_high_double, x_msb_high_double);
        vcvtdq2pd(v_msb_low_double, x_msb_low_double);

        // Multiply (0/1) * 2^31
        vmulpd(v_msb_high_double, v_msb_high_double, v_multiplier_double);
        vmulpd(v_msb_low_double, v_msb_low_double, v_multiplier_double);

        // Convert uint32_t to double for accuracy
        vcvtdq2pd(v_result_high_double, x_result_high_double);
        vcvtdq2pd(v_result_low_double, x_result_low_double);
        vcvtdq2pd(v_range_double, x_range_double);

        // Add sign as 2^31 if was present, correctly converting uint32_t to double
        vaddpd(v_result_high_double, v_result_high_double, v_msb_high_double);
        vaddpd(v_result_low_double, v_result_low_double, v_msb_low_double);

        // Compute approximate division
        const auto v_aprox_result_high_double = getVmm();
        const auto v_aprox_result_low_double = getVmm();
        vdivpd(v_aprox_result_high_double, v_result_high_double, v_range_double);  // value / range = (aux = aux / aux2)
        vdivpd(v_aprox_result_low_double, v_result_low_double, v_range_double);    // value / range = (aux = aux / aux2)

        // Floor the result to nearest int (biggest multiple of divisor)
        vroundpd(v_aprox_result_high_double, v_aprox_result_high_double, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        vroundpd(v_aprox_result_low_double, v_aprox_result_low_double, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

        // Compute closest divisible value by multiplying back
        vmulpd(v_aprox_result_high_double,
               v_aprox_result_high_double,
               v_range_double);  // aux = floor(double(result) / double(range)) * double(range)
        vmulpd(v_aprox_result_low_double,
               v_aprox_result_low_double,
               v_range_double);  // aux = floor(double(result) / double(range)) * double(range)

        // Compute remainder by subtracting approximation from the original
        vsubpd(v_result_high_double, v_result_high_double, v_aprox_result_high_double);
        vsubpd(v_result_low_double, v_result_low_double, v_aprox_result_low_double);

        // Convert 64 -> 32, always possible as 0 < result < range
        vcvtpd2dq(x_result_high_double, v_result_high_double);  // value - closest_div_value = remainder (modulo)
        vcvtpd2dq(x_result_low_double, v_result_low_double);    // value - closest_div_value = remainder (modulo)

        // Concatenate them back, now result holds all remainders (modulos)
        vinserti128(v_result, v_result, x_result_high_double, 1);
        vinserti128(v_result, v_result, x_result_low_double, 0);

        // Add minimum
        vpaddd(v_result, v_result, v_min);  // remainder + min
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    }
}

template <x64::cpu_isa_t isa>  // Works for SSE41
void MersenneTwisterGenerator<isa>::convertToOutputTypeMersenne() {
    if (m_jcp.out_data_type == element::f32) {
        // Apply mask and divisor
        // No need to do int32's voodoo with double since mask ensures
        // that most significant bit is 0
        pand(v_result, v_mask);
        cvtdq2ps(v_result, v_result);
        mulps(v_result, v_divisor);

        // Scale and shift
        mulps(v_result, v_range);
        addps(v_result, v_min);
    } else if (m_jcp.out_data_type == element::i32) {
        // Split result before converting 32 -> 64 to fit new bits
        const auto r64_result_high_double = getReg64();
        const auto r64_result_low_double = getReg64();
        const auto r64_range_double = getReg64();

        pextrq(r64_result_high_double, v_result, 1);
        pextrq(r64_result_low_double, v_result, 0);
        pextrq(r64_range_double, v_range, 0);

        // Extract the most significant bit (MSB)
        const auto v_msb_high_double = getVmm();
        const auto v_msb_low_double = getVmm();
        const auto r64_aux = getReg64();
        const auto v_result_aux = getVmm();

        mov(r64_aux, 8000000080000000);  // Bitmask to check for leading 1

        movq(v_result_aux, r64_aux);
        movq(v_msb_high_double, r64_result_high_double);
        movq(v_msb_low_double, r64_result_low_double);

        and_(v_msb_high_double, v_result_aux);
        psrld(v_msb_high_double, 31);

        and_(v_msb_low_double, v_result_aux);
        psrld(v_msb_low_double, 31);

        // Convert most significant digit to double (either 0 or 1)
        cvtdq2pd(v_msb_high_double, v_msb_high_double);
        cvtdq2pd(v_msb_low_double, v_msb_low_double);

        // Remove most significant digit from result
        mov(r64_aux, 0x7FFFFFFF7FFFFFFF);
        and_(r64_result_high_double, r64_aux);
        and_(r64_result_low_double, r64_aux);

        // Create a double value of 2^31 for the most significant digit instead of -1
        mov(r64_aux, 0x41E0000000000000);  // 2^31 in IEEE 754 double format
        movq(v_result_aux, r64_aux);       // v_result_aux reused to store multiplier
        pshufd(v_result_aux, v_result_aux, 0x11);

        // Multiply (0/1) * 2^31
        mulpd(v_msb_high_double, v_result_aux);
        mulpd(v_msb_low_double, v_result_aux);

        // Convert uint32_t to double for accuracy
        const auto v_result_high_double = getVmm();
        const auto v_result_low_double = getVmm();
        const auto v_range_double = getVmm();

        movq(v_result_high_double, r64_result_high_double);
        movq(v_result_low_double, r64_result_low_double);
        movq(v_range_double, r64_range_double);

        cvtdq2pd(v_result_high_double, v_result_high_double);
        cvtdq2pd(v_result_low_double, v_result_low_double);
        cvtdq2pd(v_range_double, v_range_double);

        // Add sign as 2^31 if was present, correctly converting uint32_t to double
        addpd(v_result_high_double, v_msb_high_double);
        addpd(v_result_low_double, v_msb_low_double);

        // Compute approximate division
        const auto v_aprox_result_high_double = getVmm();
        const auto v_aprox_result_low_double = getVmm();

        movups(v_aprox_result_high_double, v_result_high_double);
        movups(v_aprox_result_low_double, v_result_low_double);
        divpd(v_aprox_result_high_double, v_range_double);  // value / range = (aux = aux / aux2)
        divpd(v_aprox_result_low_double, v_range_double);   // value / range = (aux = aux / aux2)

        // Floor the result to nearest int (biggest multiple of divisor)
        roundpd(v_aprox_result_high_double, v_aprox_result_high_double, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        roundpd(v_aprox_result_low_double, v_aprox_result_low_double, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

        // Compute closest divisible value by multiplying back
        mulpd(v_aprox_result_high_double,
              v_range_double);  // aux = floor(double(result) / double(range)) * double(range)
        mulpd(v_aprox_result_low_double,
              v_range_double);  // aux = floor(double(result) / double(range)) * double(range)

        // Compute remainder by subtracting approximation from the original
        subpd(v_result_high_double, v_aprox_result_high_double);
        subpd(v_result_low_double, v_aprox_result_low_double);

        // Convert 64 -> 32, always possible as 0 < result < range
        cvtpd2dq(v_result_high_double, v_result_high_double);  // value - closest_div_value = remainder (modulo)
        cvtpd2dq(v_result_low_double, v_result_low_double);    // value - closest_div_value = remainder (modulo)

        movq(r64_result_high_double, v_result_high_double);
        movq(r64_result_low_double, v_result_low_double);

        // Concatenate them back, now result holds all remainders (modulos)
        pinsrq(v_result, r64_result_high_double, 1);
        pinsrq(v_result, r64_result_low_double, 0);

        // Add minimum
        paddd(v_result, v_min);  // remainder + min
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    }
}

template <>
void MersenneTwisterGenerator<x64::avx512_core>::storeResults() {
    const auto r64_aux = getReg64();
    const auto v_rest_mask = getMask();

    if (m_jcp.out_data_type.size() == sizeof(uint32_t)) {
        // Find minimum count from elements_to_generate and storage_capacity
        mov(r64_aux, r64_elements_to_generate);
        cmp(r64_aux, r64_storage_capacity);
        cmovg(r64_aux, r64_storage_capacity);

        fillRestWorkMask(v_rest_mask, r64_aux);
        vmovdqu32(ptr[r64_dst] | v_rest_mask, v_result);
    } else if (m_jcp.out_data_type.size() == sizeof(uint16_t)) {
        mov(r64_aux, r64_elements_to_generate);
        cmp(r64_aux, r64_storage_capacity);
        cmovg(r64_aux, r64_storage_capacity);

        // Store only the bottom half of the register
        auto ymm_result = Xbyak::Ymm(v_result);
        fillRestWorkMask(v_rest_mask, r64_aux);
        vmovdqu16(ptr[r64_dst] | v_rest_mask, ymm_result);
    } else if (m_jcp.out_data_type.size() == sizeof(uint64_t)) {
        // i64 enablement
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    }
}

template <>
void MersenneTwisterGenerator<x64::avx2>::storeResults() {
    const auto r64_aux = getReg64();
    auto v_rest_mask = getMask();

    if (m_jcp.out_data_type.size() == sizeof(uint32_t)) {
        // Find minimum count from elements_to_generate and storage_capacity
        mov(r64_aux, r64_elements_to_generate);
        cmp(r64_aux, r64_storage_capacity);
        cmovg(r64_aux, r64_storage_capacity);

        fillRestWorkMask(v_rest_mask, r64_aux, m_jcp.out_data_type.size());
        vmaskmovps(ptr[r64_dst], v_rest_mask, v_result);
    } else if (m_jcp.out_data_type.size() == sizeof(uint16_t)) {
        // AVX2 does not support 16 bit value transfer
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    } else if (m_jcp.out_data_type.size() == sizeof(uint64_t)) {
        // i64 enablement
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    }
}

template <x64::cpu_isa_t isa>  // Works for SSE41
void MersenneTwisterGenerator<isa>::storeResults() {
    const auto r64_aux = getReg64();

    if (m_jcp.out_data_type.size() == sizeof(uint32_t)) {
        // Find minimum count from elements_to_generate and storage_capacity
        auto v_rest_mask = getMask();
        mov(r64_aux, r64_elements_to_generate);
        cmp(r64_aux, r64_storage_capacity);
        cmovg(r64_aux, r64_storage_capacity);
        store(ptr[r64_dst], v_result, r64_aux, m_jcp.out_data_type.size());
    } else if (m_jcp.out_data_type.size() == sizeof(uint16_t)) {
        // SSE41 does not support 16 bit value transfer
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    } else if (m_jcp.out_data_type.size() == sizeof(uint64_t)) {
        // i64 enablement
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ",
                       m_jcp.out_data_type,
                       " for ",
                       x64::get_isa_info());
    }
}

template class PhiloxGenerator<x64::avx512_core>;
template class PhiloxGenerator<x64::avx2>;
template class PhiloxGenerator<x64::sse41>;

template class MersenneTwisterGenerator<x64::avx512_core>;
template class MersenneTwisterGenerator<x64::avx2>;
template class MersenneTwisterGenerator<x64::sse41>;

#undef INIT_ARR
#undef BROADCAST_PARAM
#undef BROADCAST_CONSTANT
#undef GET_MERSENNE_OFFSET
#undef GET_PHILOX_OFFSET

}  // namespace ov::intel_cpu::kernel::random_uniform
