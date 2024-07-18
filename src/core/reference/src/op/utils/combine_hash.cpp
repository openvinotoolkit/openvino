// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The CRC computation is used for x86.
// The calculations were taken from the article
// "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction - Intel (December, 2009)".

#include "openvino/core/visibility.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/reference/utils/combine_hash.hpp"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include "openvino/reference/utils/registers_pool.hpp"
#endif // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

#include <cstring>

namespace ov {
namespace runtime {

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
namespace jit {

#define GET_OFF(field) offsetof(CombineHashCallArgs, field)
#define getReg64() RegistersPool::Reg<Xbyak::Reg64>(registersPool)
#define getVmm()   RegistersPool::Reg<Vmm>(registersPool)
#define getXmm()   RegistersPool::Reg<Xbyak::Xmm>(registersPool)

struct CombineHashCompileParams {
};

struct CombineHashCallArgs {
    const void* src_ptr;
    void* dst_ptr;
    uint64_t work_amount = 0lu;
    uint64_t make_64_fold = 0lu;
};

typedef void (*fn_t)(const CombineHashCallArgs*);

template <cpu_isa_t isa>
class CombineHash : public Generator {
public:
    explicit CombineHash(const CombineHashCompileParams& jcp) :
            m_jcp(jcp) {
        if (isa == avx512_core) {
            vlen = zmm_len;
        } else if (isa == avx2) {
            vlen = ymm_len;
        } else {
            OPENVINO_THROW("Unsupported isa: ", isa);
        }
        if (!mayiuse(cpu_isa_t::pclmulqdq)) {
            OPENVINO_THROW("The current CPU does not support pclmulqdq instruction, which is required for the CRC algorithm.");
        }
        if (mayiuse(cpu_isa_t::vpclmulqdq)) {
            is_vpclmulqdq = true;
        }

        generate();
    }

    void generate() {
        this->preamble();
        registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

        r64_src = getReg64();
        r64_dst = getReg64();
        r64_work_amount  = getReg64();
        r64_make_64_fold = getReg64();

        mov(r64_src, ptr[r64_params + GET_OFF(src_ptr)]);
        mov(r64_dst, ptr[r64_params + GET_OFF(dst_ptr)]);
        mov(r64_work_amount, ptr[r64_params + GET_OFF(work_amount)]);
        mov(r64_make_64_fold, ptr[r64_params + GET_OFF(make_64_fold)]);

        initVectors();
        bulkFold(v_dst);
        restFold(v_dst);
        tailFold(v_dst);

        registersPool.reset();
        this->postamble();
    }

    static fn_t get() {
        static const CombineHashCompileParams params;
        static CombineHash<isa> kernel(params);

        return (fn_t)kernel.getCode();
    }

    void fillRestWorkMask(const Xbyak::Opmask& k_dst_mask,
                          const Xbyak::Reg64& r64_work_rest) {
        Xbyak::Label l_mv_mask;
        auto rOnes = getReg64();

        mov(rOnes, 0xFFFFFFFFFFFFFFFF);
        cmp(r64_work_rest, 0x3f);
        jg(l_mv_mask);

        shlx(rOnes, rOnes, r64_work_rest);
        not_(rOnes);

        L(l_mv_mask);
        kmovq(k_dst_mask, rOnes);
    }

    void partialLoad(const Xbyak::Xmm&     xmm_dst,
                     const Xbyak::Address& src_addr,
                     const Xbyak::Reg64&   r64_load_num) {
        Xbyak::Label l_partial, l_end;

        cmp(r64_load_num, xmm_len);
        jl(l_partial, T_NEAR);
        vmovdqu(xmm_dst, ptr[src_addr.getRegExp()]);
        jmp(l_end, T_NEAR);

        L(l_partial); {
            size_t offset = xmm_len;

            for (size_t j = 0lu; j < xmm_len - 1; j++) {
                pinsrb(xmm_dst, ptr[src_addr.getRegExp() + offset], j);
                cmp(r64_load_num, ++offset);
                jle(l_end, T_NEAR);
            }
        }

        L(l_end);
    }

    void partialLoad(const Xbyak::Ymm&     ymm_dst,
                     const Xbyak::Address& src_addr,
                     const Xbyak::Reg64&   r64_load_num) {
        Xbyak::Label l_xmm, l_partial, l_end;
        auto xmm_dst = Xbyak::Xmm(ymm_dst.getIdx());

        cmp(r64_load_num, ymm_len);
        jl(l_xmm, T_NEAR);
        vmovdqu(ymm_dst, ptr[src_addr.getRegExp()]);
        jmp(l_end, T_NEAR);

        L(l_xmm);
        vpxorq(ymm_dst, ymm_dst, ymm_dst);
        cmp(r64_load_num, xmm_len);
        jl(l_partial, T_NEAR);
        vmovdqu(xmm_dst, ptr[src_addr.getRegExp()]);
        je(l_end, T_NEAR);

        {
            Xbyak::Label l_rest_loop, l_perm;
            size_t offset = xmm_len;

            vperm2f128(ymm_dst, ymm_dst, ymm_dst, 0x1);
            for (size_t j = 0lu; j < xmm_len - 1; j++) {
                pinsrb(xmm_dst, ptr[src_addr.getRegExp() + offset], j);
                cmp(r64_load_num, ++offset);
                jle(l_perm, T_NEAR);
            }
            L(l_perm);
            vperm2f128(ymm_dst, ymm_dst, ymm_dst, 0x1);
        }
        jmp(l_end, T_NEAR);

        L(l_partial); {
            size_t offset = xmm_len;

            for (size_t j = 0lu; j < xmm_len - 1; j++) {
                pinsrb(xmm_dst, ptr[src_addr.getRegExp() + offset], j);
                cmp(r64_load_num, ++offset);
                jle(l_end, T_NEAR);
            }
        }

        L(l_end);
    }

private:
    static constexpr uint64_t CHUNK_SIZE = 32;
    static const uint64_t CRC_VAL;
    static const uint64_t CONST_K[12];
    static const uint8_t SHUF_MASK[16];

    using Vmm = typename std::conditional<isa == avx512_core, Xbyak::Zmm, Xbyak::Ymm>::type;
    size_t vlen = xmm_len;
    bool is_vpclmulqdq = false;

    CombineHashCompileParams m_jcp;
    RegistersPool::Ptr registersPool;

    RegistersPool::Reg<Xbyak::Reg64> r64_src;
    RegistersPool::Reg<Xbyak::Reg64> r64_dst;
    RegistersPool::Reg<Xbyak::Reg64> r64_work_amount;
    RegistersPool::Reg<Xbyak::Reg64> r64_make_64_fold;

    const Xbyak::Reg64 r64_params = abi_param1;

    // Vector registers
    RegistersPool::Reg<Vmm> v_dst;
    RegistersPool::Reg<Vmm> v_k_1_2;
    RegistersPool::Reg<Vmm> v_k_4_5;
    RegistersPool::Reg<Vmm> v_k_8_9;
    RegistersPool::Reg<Vmm> v_k_16_17;
    RegistersPool::Reg<Vmm> v_shuf_mask;

    size_t getVlen() {
        return vlen;
    }

    void initVectors();

    void bulkFold(const Vmm& v_dst);

    void restFold(const Vmm& v_dst) {
        Xbyak::Label l_fold_loop, l_end;
        cmp(r64_work_amount, xmm_len);
        jl(l_end, T_NEAR);

        auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
        auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());
        auto xmm_src = getXmm();
        auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
        auto xmm_aux = getXmm();

        L(l_fold_loop); {
            vmovdqu64(xmm_src, ptr[r64_src]);
            vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

            vpclmulqdq(xmm_aux, xmm_dst, xmm_k_1_2, 0b00000000);
            vpclmulqdq(xmm_dst, xmm_dst, xmm_k_1_2, 0b00010001);
            vpxorq(xmm_dst, xmm_dst, xmm_aux);
            vpxorq(xmm_dst, xmm_dst, xmm_src);

            add(r64_src, xmm_len);
            sub(r64_work_amount, xmm_len);
            cmp(r64_work_amount, xmm_len);
            jge(l_fold_loop, T_NEAR);
        }

        L(l_end);
    }

    void tailFold(const Vmm& v_dst);
};

template <>
void CombineHash<avx512_core>::initVectors() {
    auto r64_aux = getReg64();

    v_k_1_2 = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K));
    vbroadcasti64x2(v_k_1_2, ptr[r64_aux]);
    v_k_8_9 = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 6));
    vbroadcasti64x2(v_k_8_9, ptr[r64_aux]);

    v_shuf_mask = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(SHUF_MASK));
    vbroadcasti64x2(v_shuf_mask, ptr[r64_aux]);

    v_dst = getVmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_aux = getXmm();
    auto k_rest_mask = RegistersPool::Reg<Xbyak::Opmask>(registersPool);
    // Initial CRC
    mov(r64_aux, CRC_VAL);
    vpxorq(v_dst, v_dst, v_dst);
    vpinsrq(xmm_dst, xmm_dst, r64_work_amount, 0x0);
    vpinsrq(xmm_dst, xmm_dst, r64_aux, 0x1);
    // First xor with source
    fillRestWorkMask(k_rest_mask, r64_work_amount);
    vmovdqu8(Xbyak::Xmm(xmm_aux.getIdx()) | k_rest_mask | T_z, ptr[r64_src]);
    vpshufb(xmm_aux, xmm_aux, xmm_shuf_mask);
    vpxorq(xmm_dst, xmm_dst, xmm_aux);
    sub(r64_work_amount, xmm_len);
    add(r64_src, xmm_len);
}

template <cpu_isa_t isa>
void CombineHash<isa>::initVectors() {
    auto r64_aux = getReg64();

    v_k_1_2 = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K));
    vbroadcasti128(v_k_1_2, ptr[r64_aux]);
    v_k_8_9 = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 6));
    vbroadcasti128(v_k_8_9, ptr[r64_aux]);

    v_shuf_mask = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(SHUF_MASK));
    vbroadcasti128(v_shuf_mask, ptr[r64_aux]);

    v_dst = getVmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_aux = getXmm();
    auto k_rest_mask = RegistersPool::Reg<Xbyak::Opmask>(registersPool);
    // Initial CRC
    mov(r64_aux, CRC_VAL);
    vpxorq(v_dst, v_dst, v_dst);
    vpinsrq(xmm_dst, xmm_dst, r64_aux, 0x1);
    // First xor with source
    partialLoad(xmm_aux, ptr[r64_src], r64_work_amount);
    vpshufb(xmm_aux, xmm_aux, xmm_shuf_mask);
    vpxorq(xmm_dst, xmm_dst, xmm_aux);
    sub(r64_work_amount, xmm_len);
}

template <>
void CombineHash<avx512_core>::bulkFold(const Vmm& v_dst) {
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, zmm_len + 3 * xmm_len);
    jl(l_end, T_NEAR);

    auto r64_aux = getReg64();

    auto v_src_0 = getVmm();
    auto v_dst_0 = getVmm();
    auto v_dst_1 = getVmm();
    auto v_dst_2 = getVmm();
    auto& v_dst_3 = v_dst;
    auto v_aux_0 = getVmm();

    auto xmm_k_8_9 = Xbyak::Xmm(v_k_8_9.getIdx());
    auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());
    auto xmm_src_0 = Xbyak::Xmm(v_src_0.getIdx());
    auto xmm_src_1 = getXmm();
    auto xmm_dst_0 = Xbyak::Xmm(v_dst_0.getIdx());
    auto xmm_dst_1 = Xbyak::Xmm(v_dst_1.getIdx());
    auto xmm_dst_2 = Xbyak::Xmm(v_dst_2.getIdx());
    auto xmm_dst_3 = Xbyak::Xmm(v_dst_3.getIdx());
    auto xmm_aux_0 = Xbyak::Xmm(v_aux_0.getIdx());

    vmovdqu64(v_dst_0, v_dst_3);

    if (!is_vpclmulqdq) {
        prefetchnta(ptr[r64_src + 3 * xmm_len]);
        vmovdqu64(xmm_dst_1, ptr[r64_src + 0 * xmm_len]);
        vmovdqu64(xmm_dst_2, ptr[r64_src + 1 * xmm_len]);
        vmovdqu64(xmm_dst_3, ptr[r64_src + 2 * xmm_len]);
    }

    add(r64_src, 3 * xmm_len);
    sub(r64_work_amount, zmm_len + 3 * xmm_len);

    L(l_fold_loop); {
        vmovdqu64(v_src_0, ptr[r64_src]);
        vpshufb(v_src_0, v_src_0, v_shuf_mask);

        if (is_vpclmulqdq) {
            vpclmulqdq(v_aux_0, v_dst_0, v_k_8_9, 0b00000000);
            vpclmulqdq(v_dst_0, v_dst_0, v_k_8_9, 0b00010001);
            vpxorq(v_aux_0, v_aux_0, v_src_0);
            vpxorq(v_dst_0, v_dst_0, v_aux_0);
        } else {
            // 0
            vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_8_9, 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_8_9, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            vpxorq(xmm_dst_0, xmm_dst_0, xmm_aux_0);
            // 1
            vextracti64x2(xmm_src_1, v_src_0, 0x1);
            vpclmulqdq(xmm_aux_0, xmm_dst_1, xmm_k_8_9, 0b00000000);
            vpclmulqdq(xmm_dst_1, xmm_dst_1, xmm_k_8_9, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);
            // 2
            vextracti64x2(xmm_src_1, v_src_0, 0x2);
            vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_8_9, 0b00000000);
            vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_8_9, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            vpxorq(xmm_dst_2, xmm_dst_2, xmm_aux_0);
            // 3
            vextracti64x2(xmm_src_1, v_src_0, 0x3);
            vpclmulqdq(xmm_aux_0, xmm_dst_3, xmm_k_8_9, 0b00000000);
            vpclmulqdq(xmm_dst_3, xmm_dst_3, xmm_k_8_9, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        }

        add(r64_src, zmm_len);
        sub(r64_work_amount, zmm_len);
        jge(l_fold_loop, T_NEAR);
    }
    add(r64_work_amount, zmm_len);

    if (is_vpclmulqdq) {
        auto ymm_dst_0 = Xbyak::Ymm(v_dst_0.getIdx());
        auto ymm_dst_1 = Xbyak::Ymm(v_dst_1.getIdx());
        auto ymm_aux_0 = Xbyak::Ymm(v_aux_0.getIdx());

        vextracti64x4(ymm_dst_1, v_dst_0, 0x1);
        mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 2));
        vpclmulqdq(ymm_aux_0, ymm_dst_0, ptr[r64_aux], 0b00000000);
        vpclmulqdq(ymm_dst_0, ymm_dst_0, ptr[r64_aux], 0b00010001);
        vpxorq(ymm_dst_1, ymm_dst_1, ymm_aux_0);
        vpxorq(ymm_dst_0, ymm_dst_0, ymm_dst_1);

        vextracti64x2(xmm_dst_3, ymm_dst_0, 0x1);
        vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_1_2, 0b00000000);
        vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_1_2, 0b00010001);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);
    } else {
        mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 4));
        vpclmulqdq(xmm_aux_0, xmm_dst_0, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_dst_0, xmm_dst_0, ptr[r64_aux], 0b00010001);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);

        mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 2));
        vpclmulqdq(xmm_aux_0, xmm_dst_1, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_dst_1, xmm_dst_1, ptr[r64_aux], 0b00010001);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_1);

        vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_1_2, 0b00000000);
        vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_1_2, 0b00010001);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_2);
    }

    L(l_end);
}

template <>
void CombineHash<avx2>::bulkFold(const Vmm& v_dst) {
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, 2 * vlen - xmm_len);
    jl(l_end, T_NEAR);

    auto r64_aux = getReg64();

    auto v_src_0 = getVmm();
    auto v_dst_0 = getVmm();
    auto v_dst_1 = getVmm();
    auto v_dst_2 = getVmm();
    auto& v_dst_3 = v_dst;
    auto v_aux_0 = getVmm();

    auto xmm_k_4_5 = Xbyak::Xmm(v_k_4_5.getIdx());
    auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());
    auto xmm_src_0 = Xbyak::Xmm(v_src_0.getIdx());
    auto xmm_src_1 = getXmm();
    auto xmm_dst_0 = Xbyak::Xmm(v_dst_0.getIdx());
    auto xmm_dst_1 = Xbyak::Xmm(v_dst_1.getIdx());
    auto xmm_dst_2 = Xbyak::Xmm(v_dst_2.getIdx());
    auto xmm_dst_3 = Xbyak::Xmm(v_dst_3.getIdx());
    auto xmm_aux_0 = Xbyak::Xmm(v_aux_0.getIdx());

    if (!is_vpclmulqdq) {
        vmovdqu64(xmm_dst_1, ptr[r64_src + 0 * xmm_len]);
    }

    add(r64_src, vlen - xmm_len);
    sub(r64_work_amount, 2 * vlen - xmm_len);

    L(l_fold_loop); {
        vmovdqu64(v_src_0, ptr[r64_src]);
        vpshufb(v_src_0, v_src_0, v_shuf_mask);

        if (is_vpclmulqdq) {
            vpclmulqdq(v_aux_0, v_dst_0, v_k_4_5, 0b00000000);
            vpclmulqdq(v_dst_0, v_dst_0, v_k_4_5, 0b00010001);
            vpxorq(v_aux_0, v_aux_0, v_src_0);
            vpxorq(v_dst_0, v_dst_0, v_aux_0);
        } else {
            // 0
            vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_4_5, 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_4_5, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            vpxorq(xmm_dst_0, xmm_dst_0, xmm_aux_0);
            // 1
            vextracti128(xmm_src_1, v_src_0, 0x1);
            vpclmulqdq(xmm_aux_0, xmm_dst_1, xmm_k_4_5, 0b00000000);
            vpclmulqdq(xmm_dst_1, xmm_dst_1, xmm_k_4_5, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);
        }

        add(r64_src, vlen);
        sub(r64_work_amount, vlen);
        jge(l_fold_loop, T_NEAR);
    }
    add(r64_work_amount, vlen);

    if (is_vpclmulqdq) {
        auto ymm_dst_0 = Xbyak::Ymm(v_dst_0.getIdx());
        auto ymm_dst_1 = Xbyak::Ymm(v_dst_1.getIdx());
        auto ymm_aux_0 = Xbyak::Ymm(v_aux_0.getIdx());

        vextracti128(xmm_dst_3, ymm_dst_0, 0x1);
        vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_1_2, 0b00000000);
        vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_1_2, 0b00010001);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);
    } else {
        vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_1_2, 0b00000000);
        vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_1_2, 0b00010001);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_2);
    }

    L(l_end);
}


template <>
void CombineHash<avx512_core>::tailFold(const Vmm& v_dst) {
    Xbyak::Label l_fold_to_64, l_save_128, l_end;
    cmp(r64_work_amount, 0);
    jle(l_fold_to_64, T_NEAR);

    auto r64_aux = getReg64();
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());
    auto xmm_src = getXmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux = getXmm();
    auto xmm_aux_1 = getXmm();
    auto xmm_aux_2 = getXmm();
    auto k_rest_mask = RegistersPool::Reg<Xbyak::Opmask>(registersPool);

    fillRestWorkMask(k_rest_mask, r64_work_amount);

    vpxorq(xmm_src, xmm_src, xmm_src);
    vmovdqu8(Xbyak::Xmm(xmm_src.getIdx()) | k_rest_mask | T_z, ptr[r64_src]);
    vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

    vpclmulqdq(xmm_aux, xmm_dst, xmm_k_1_2, 0b00000000);
    vpclmulqdq(xmm_dst, xmm_dst, xmm_k_1_2, 0b00010001);
    vpxorq(xmm_aux, xmm_aux, xmm_src);
    vpxorq(xmm_dst, xmm_dst, xmm_aux);

    L(l_fold_to_64);
    cmp(r64_make_64_fold, 0);
    je(l_save_128, T_NEAR);

    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 8));
    vpclmulqdq(xmm_aux, xmm_dst, ptr[r64_aux], 0b00000001);
    vpslldq(xmm_dst, xmm_dst, 0x8);
    vpxorq(xmm_dst, xmm_dst, xmm_aux);

    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 10));
    vmovdqu64(xmm_aux_2, ptr[r64_aux]);
    vpclmulqdq(xmm_aux, xmm_dst, xmm_aux_2, 0b00000001);
    mov(r64_aux, 0x0);
    vpinsrq(xmm_aux_1, xmm_dst, r64_aux, 0x0);
    vpxorq(xmm_aux, xmm_aux, xmm_aux_1);
    vpinsrq(xmm_aux_1, xmm_aux, r64_aux, 0x0);
    vpclmulqdq(xmm_aux, xmm_aux, xmm_aux_2, 0b00010001);
    vpxorq(xmm_aux, xmm_aux, xmm_aux_1);
    vpxorq(xmm_dst, xmm_dst, xmm_aux);

    vpextrq(ptr[r64_dst], xmm_dst, 0x0);
    jmp(l_end, T_NEAR);


    L(l_save_128);
    vmovdqu64(ptr[r64_dst], xmm_dst);

    L(l_end);
}

template <>
void CombineHash<avx2>::tailFold(const Vmm& v_dst) {
}

template <cpu_isa_t isa>
const uint64_t CombineHash<isa>::CRC_VAL = 0xffffffffffffffff;

// P(x) = 0x42F0E1EBA9EA3693
template <cpu_isa_t isa>
const uint64_t CombineHash<isa>::CONST_K[12] = { 0x05f5c3c7eb52fab6, 0x4eb938a7d257740e,  // x^(64*1), x^(64*2)
                                                 0x571bee0a227ef92b, 0x44bef2a201b5200c,  // x^(64*3), x^(64*4)
                                                 0x54819d8713758b2c, 0x4a6b90073eb0af5a,  // x^(64*5), x^(64*6)
                                                 0x5f6843ca540df020, 0xddf4b6981205b83f,  // x^(64*7), x^(64*8)
                                                 0x05f5c3c7eb52fab6, 0x0000000000000000,  // x^(64*1), x^(64*1) mod P(x)
                                                 0x578d29d06cc4f872, 0x42f0e1eba9ea3693   // floor(x^128/P(x)) - x^64, P(x) - x^64
                                                };

template <cpu_isa_t isa>
const uint8_t CombineHash<isa>::SHUF_MASK[] = { 0b00001111, 0b00001110, 0b00001101, 0b00001100, 0b00001011, 0b00001010, 0b00001001, 0b00001000,
                                                0b00000111, 0b00000110, 0b00000101, 0b00000100, 0b00000011, 0b00000010, 0b00000001, 0b00000000 };

} // namespace jit
#endif // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

size_t combine_hash(const void* src, size_t size) {
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    jit::fn_t kernel;

    if (jit::Generator::mayiuse(jit::avx512_core)) {
        kernel = jit::CombineHash<jit::avx512_core>::get();
    } else if (jit::Generator::mayiuse(jit::avx2)) {
        kernel = jit::CombineHash<jit::avx2>::get();
    }

    if (kernel) {
        size_t res = 0lu;

        static const size_t block_size = 2lu * jit::Generator::zmm_len;
        // There is no sense to perform parallel execution if there are less than 2 blocks.
        if (size >= 2lu * block_size) {
            const auto nthr = parallel_get_max_threads() / 2; // TODO: WA for Hyper Threading
            std::vector<uint64_t> intermediate(nthr * 2); // xmm_len * nthr
            const uint64_t blocks = size / block_size;
            const uint64_t el_per_thread = block_size * ((blocks + nthr - 1) / nthr);

            parallel_nt(nthr, [&](const int ithr, const int nthr) {
                uint64_t start = ithr * el_per_thread;
                if (start >= size) {
                    return;
                }
                uint64_t work_amount = (el_per_thread + start > size) ? size - start : el_per_thread;

                size_t res = 0lu;
                jit::CombineHashCallArgs args;

                args.src_ptr = reinterpret_cast<const uint8_t *>(src) + start;
                args.dst_ptr = &intermediate[ithr * 2];
                args.work_amount = work_amount;
                args.make_64_fold = 0lu;
                kernel(&args);
            });


            jit::CombineHashCallArgs args;
            args.src_ptr = intermediate.data();
            args.dst_ptr = &res;
            args.work_amount = ((size + el_per_thread - 1) / el_per_thread) * jit::Generator::xmm_len;
            args.make_64_fold = 1lu;
            kernel(&args);
        } else {
            jit::CombineHashCallArgs args;
            args.src_ptr = src;
            args.dst_ptr = &res;
            args.work_amount = size;
            args.make_64_fold = 1lu;
            kernel(&args);
        }
        return res;
    }
#endif // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

    constexpr auto cel_size = sizeof(size_t);
    auto seed = static_cast<size_t>(size);
    const auto data = static_cast<const size_t*>(src);
    const auto d_end = std::next(data, size / cel_size);
    // The constant value used as a magic number has been
    // traditionally used e.g. in boost library's hash_combine.
    // It happens to be derived from the golden ratio.
    for (auto d = data; d != d_end; ++d) {
        seed ^= *d + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    size_t last_bytes{0};
    std::memcpy(&last_bytes, d_end, size % cel_size);
    seed ^= last_bytes + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

}   // namespace runtime
}   // namespace ov
