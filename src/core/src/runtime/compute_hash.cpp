// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The CRC computation is used for x86.
// The calculations were taken from the article
// "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction - Intel (December, 2009)".

#include "openvino/runtime/compute_hash.hpp"

#include <cmath>
#include <cstring>
#include <iterator>

#include "openvino/core/visibility.hpp"

#if !defined(OS_CHROMEOS) && (defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64))
#    define OV_CORE_USE_XBYAK_JIT
#endif

#ifdef OV_CORE_USE_XBYAK_JIT
#    include "openvino/core/parallel.hpp"
#    include "openvino/reference/utils/registers_pool.hpp"
#    include "openvino/util/common_util.hpp"
#endif  // OV_CORE_USE_XBYAK_JIT

namespace ov {
namespace runtime {

#ifdef OV_CORE_USE_XBYAK_JIT

using namespace ov::reference::jit;

namespace jit {

#    define GET_OFF(field) offsetof(ComputeHashCallArgs, field)
#    define getReg64()     RegistersPool::Reg<Xbyak::Reg64>(m_registers_pool)
#    define getVmm()       RegistersPool::Reg<Vmm>(m_registers_pool)
#    define getXmm()       RegistersPool::Reg<Xbyak::Xmm>(m_registers_pool)

enum KernelType { SINGLE_THREAD = 0, FIRST_THREAD, N_THREAD, FINAL_FOLD };

struct ComputeHashCompileParams {
    KernelType type;
};

struct ComputeHashCallArgs {
    const void* src_ptr = nullptr;
    void* dst_ptr = nullptr;
    const void* k_ptr = nullptr;
    void* intermediate_ptr = nullptr;
    uint64_t work_amount = 0lu;
    uint64_t size = 0lu;
    uint64_t threads_num = 1lu;
};

typedef void (*hash_kernel)(const ComputeHashCallArgs*);

static const uint8_t SHUF_MASK[16] = {0b00001111,
                                      0b00001110,
                                      0b00001101,
                                      0b00001100,
                                      0b00001011,
                                      0b00001010,
                                      0b00001001,
                                      0b00001000,
                                      0b00000111,
                                      0b00000110,
                                      0b00000101,
                                      0b00000100,
                                      0b00000011,
                                      0b00000010,
                                      0b00000001,
                                      0b00000000};

constexpr uint64_t CRC_VAL = 0xffffffffffffffff;

// POLYNOM(x) = 0x42F0E1EBA9EA3693
constexpr uint64_t K_2 = 0x05f5c3c7eb52fab6;  // x^(64*2)
constexpr uint64_t P_1 = 0x578d29d06cc4f872;  // floor(x^128/P(x))-x^64
constexpr uint64_t P_2 = 0x42f0e1eba9ea3693;  // P(x)-x^64
static const uint64_t K_PULL[] = {
    K_2,                 // x^(64*2)
    0x4eb938a7d257740e,  // x^(64*3)
    0x571bee0a227ef92b,  // x^(64*4)
    0x44bef2a201b5200c,  // x^(64*5)
    0x54819d8713758b2c,  // x^(64*6)
    0x4a6b90073eb0af5a,  // x^(64*7)
    0x5f6843ca540df020,  // x^(64*8)
    0xddf4b6981205b83f,  // x^(64*9)
    0x097c516e98bd2e73,  // x^(64*10)
    0x0b76477b31e22e7b,  // x^(64*11)
    0x9af04e1eff82d0dd,  // x^(64*12)
    0x6e82e609297f8fe8,  // x^(64*13)
    0xe464f4df5fb60ac1,  // x^(64*14)
    0xb649c5b35a759cf2,  // x^(64*15)
    0x05cf79dea9ac37d6,  // x^(64*16)
    0x001067e571d7d5c2   // x^(64*17)
};

constexpr uint64_t K_2_3_OFF = 0lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_4_5_OFF = 1lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_6_7_OFF = 2lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_8_9_OFF = 3lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_10_11_OFF = 4lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_12_13_OFF = 5lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_14_15_OFF = 6lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_16_17_OFF = 7lu * 2lu * sizeof(uint64_t);

class HashBase : public Generator {
protected:
    void (*ker_fn)(const ComputeHashCallArgs*);

public:
    HashBase(cpu_isa_t isa) : Generator(isa) {}

    virtual void generate() = 0;

    void operator()(const ComputeHashCallArgs* args) {
        ker_fn(args);
    }

    virtual void create_kernel() {
        generate();
        ker_fn = (decltype(ker_fn))getCode();
        OPENVINO_ASSERT(ker_fn, "[ CORE ] Could not generate kernel code.");
    }
};

template <cpu_isa_t isa>
class ComputeHash : public HashBase {
public:
    explicit ComputeHash(const ComputeHashCompileParams& jcp) : HashBase(isa), m_jcp(jcp) {
        if (!mayiuse(cpu_isa_t::pclmulqdq)) {
            OPENVINO_THROW(
                "The current CPU does not support pclmulqdq instruction, which is required for the CRC algorithm.");
        }
        if (mayiuse(cpu_isa_t::vpclmulqdq)) {
            is_vpclmulqdq = true;
        }
    }

    void generate() override {
        m_registers_pool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

        r64_src_ptr = getReg64();
        r64_dst_ptr = getReg64();
        r64_work_amount = getReg64();
        r64_k_ptr = getReg64();
        r64_aux = getReg64();
        v_k_2_3 = getVmm();
        v_shuf_mask = getVmm();
        auto v_dst = getVmm();

        this->preamble();

        initialize(v_dst);
        bulk_fold(v_dst);
        join(v_dst);
        fold_to_128(v_dst);
        fold_to_64(v_dst);

        this->postamble();
        m_registers_pool.reset();
    }

    static std::shared_ptr<HashBase> create(const ComputeHashCompileParams& params) {
        auto kernel = std::make_shared<ComputeHash>(params);
        OPENVINO_ASSERT(kernel, "[ CORE ] Could not create ComputeHash kernel.");
        kernel->create_kernel();

        return kernel;
    }

private:
    using Vmm = typename std::conditional<isa == avx512_core, Xbyak::Zmm, Xbyak::Ymm>::type;
    bool is_vpclmulqdq = false;

    ComputeHashCompileParams m_jcp;
    RegistersPool::Ptr m_registers_pool;

    const Xbyak::Reg64 r64_params = abi_param1;

    RegistersPool::Reg<Xbyak::Reg64> r64_src_ptr;
    RegistersPool::Reg<Xbyak::Reg64> r64_dst_ptr;
    RegistersPool::Reg<Xbyak::Reg64> r64_work_amount;
    RegistersPool::Reg<Xbyak::Reg64> r64_k_ptr;
    RegistersPool::Reg<Xbyak::Reg64> r64_aux;

    // Vector registers
    RegistersPool::Reg<Vmm> v_k_2_3;
    RegistersPool::Reg<Vmm> v_shuf_mask;

    void initialize(const Vmm& v_dst);

    void bulk_fold(const Vmm& v_dst);

    void join(const Vmm& v_dst);

    void fold_to_128(const Vmm& v_dst);

    void fold_to_64(const Vmm& v_dst);

    void uni_vpxorq(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src_0, const Xbyak::Xmm& v_src_1);

    void uni_vmovdqu64(const Xbyak::Xmm& v_dst, const Xbyak::Operand& v_src_0);

    void uni_vmovdqu64(const Xbyak::Address& v_dst, const Xbyak::Xmm& v_src_0);

    void uni_vbroadcasti64x2(const Xbyak::Ymm& v_dst, const Xbyak::Address& v_src_0);

    void partial_load(const Xbyak::Xmm& xmm_dst, const Xbyak::Address& src_addr, const Xbyak::Reg64& r64_load_num);

    void partial_load(const Xbyak::Ymm& ymm_dst, const Xbyak::Address& src_addr, const Xbyak::Reg64& r64_load_num);
};

template <>
void ComputeHash<avx512_core>::uni_vpxorq(const Xbyak::Xmm& v_dst,
                                          const Xbyak::Xmm& v_src_0,
                                          const Xbyak::Xmm& v_src_1) {
    vpxorq(v_dst, v_src_0, v_src_1);
}
template <cpu_isa_t isa>
void ComputeHash<isa>::uni_vpxorq(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src_0, const Xbyak::Xmm& v_src_1) {
    vpxor(v_dst, v_src_0, v_src_1);
}
template <>
void ComputeHash<avx512_core>::uni_vmovdqu64(const Xbyak::Xmm& v_dst, const Xbyak::Operand& v_src_0) {
    vmovdqu64(v_dst, v_src_0);
}
template <cpu_isa_t isa>
void ComputeHash<isa>::uni_vmovdqu64(const Xbyak::Xmm& v_dst, const Xbyak::Operand& v_src_0) {
    vmovdqu(v_dst, v_src_0);
}
template <>
void ComputeHash<avx512_core>::uni_vmovdqu64(const Xbyak::Address& v_dst, const Xbyak::Xmm& v_src_0) {
    vmovdqu64(v_dst, v_src_0);
}
template <cpu_isa_t isa>
void ComputeHash<isa>::uni_vmovdqu64(const Xbyak::Address& v_dst, const Xbyak::Xmm& v_src_0) {
    vmovdqu(v_dst, v_src_0);
}
template <>
void ComputeHash<avx512_core>::uni_vbroadcasti64x2(const Xbyak::Ymm& v_dst, const Xbyak::Address& v_src_0) {
    vbroadcasti64x2(v_dst, v_src_0);
}
template <cpu_isa_t isa>
void ComputeHash<isa>::uni_vbroadcasti64x2(const Xbyak::Ymm& v_dst, const Xbyak::Address& v_src_0) {
    vbroadcasti128(v_dst, v_src_0);
}
template <>
void ComputeHash<avx512_core>::partial_load(const Xbyak::Xmm& xmm_dst,
                                            const Xbyak::Address& src_addr,
                                            const Xbyak::Reg64& r64_load_num) {
    Xbyak::Label l_mv_mask;
    auto rOnes = getReg64();
    auto k_load_mask = RegistersPool::Reg<Xbyak::Opmask>(m_registers_pool);

    mov(rOnes, 0xFFFFFFFFFFFFFFFF);
    cmp(r64_load_num, 0x3f);
    jg(l_mv_mask);

    shlx(rOnes, rOnes, r64_load_num);
    not_(rOnes);

    L(l_mv_mask);
    kmovq(k_load_mask, rOnes);

    vmovdqu8(Vmm(xmm_dst.getIdx()) | k_load_mask | T_z, ptr[r64_src_ptr]);
}
template <cpu_isa_t isa>
void ComputeHash<isa>::partial_load(const Xbyak::Xmm& xmm_dst,
                                    const Xbyak::Address& src_addr,
                                    const Xbyak::Reg64& r64_load_num) {
    Xbyak::Label l_partial, l_end;

    cmp(r64_load_num, xmm_len);
    jl(l_partial, T_NEAR);
    uni_vmovdqu64(xmm_dst, ptr[src_addr.getRegExp()]);
    jmp(l_end, T_NEAR);

    L(l_partial);
    {
        uni_vpxorq(xmm_dst, xmm_dst, xmm_dst);
        for (size_t j = 0lu; j < xmm_len - 1; j++) {
            cmp(r64_load_num, static_cast<uint32_t>(j));
            jle(l_end, T_NEAR);
            pinsrb(xmm_dst, ptr[src_addr.getRegExp() + j], static_cast<uint8_t>(j));
        }
    }

    L(l_end);
}
template <>
void ComputeHash<avx512_core>::partial_load(const Xbyak::Ymm& xmm_dst,
                                            const Xbyak::Address& src_addr,
                                            const Xbyak::Reg64& r64_load_num) {
    partial_load(Xbyak::Xmm(xmm_dst.getIdx()), src_addr, r64_load_num);
}
template <cpu_isa_t isa>
void ComputeHash<isa>::partial_load(const Xbyak::Ymm& ymm_dst,
                                    const Xbyak::Address& src_addr,
                                    const Xbyak::Reg64& r64_load_num) {
    Xbyak::Label l_xmm, l_partial, l_end;
    auto xmm_dst = Xbyak::Xmm(ymm_dst.getIdx());

    cmp(r64_load_num, ymm_len);
    jl(l_xmm, T_NEAR);
    uni_vmovdqu64(ymm_dst, ptr[src_addr.getRegExp()]);
    jmp(l_end, T_NEAR);

    L(l_xmm);
    uni_vpxorq(ymm_dst, ymm_dst, ymm_dst);
    cmp(r64_load_num, xmm_len);
    jl(l_partial, T_NEAR);
    uni_vmovdqu64(xmm_dst, ptr[src_addr.getRegExp()]);
    je(l_end, T_NEAR);

    {
        Xbyak::Label l_rest_loop, l_perm;

        vperm2i128(ymm_dst, ymm_dst, ymm_dst, 0x1);
        for (size_t j = 0lu; j < xmm_len - 1lu; j++) {
            cmp(r64_load_num, static_cast<uint32_t>(xmm_len + j));
            jle(l_perm, T_NEAR);
            pinsrb(xmm_dst, ptr[src_addr.getRegExp() + xmm_len + j], static_cast<uint8_t>(j));
        }
        L(l_perm);
        vperm2i128(ymm_dst, ymm_dst, ymm_dst, 0x1);
    }
    jmp(l_end, T_NEAR);

    L(l_partial);
    {
        for (size_t j = 0lu; j < xmm_len - 1; j++) {
            cmp(r64_load_num, static_cast<uint32_t>(j));
            jle(l_end, T_NEAR);
            pinsrb(xmm_dst, ptr[src_addr.getRegExp() + j], static_cast<uint8_t>(j));
        }
    }

    L(l_end);
}

template <cpu_isa_t isa>
void ComputeHash<isa>::initialize(const Vmm& v_dst) {
    mov(r64_src_ptr, ptr[r64_params + GET_OFF(src_ptr)]);
    mov(r64_dst_ptr, ptr[r64_params + GET_OFF(dst_ptr)]);
    mov(r64_k_ptr, ptr[r64_params + GET_OFF(k_ptr)]);
    mov(r64_work_amount, ptr[r64_params + GET_OFF(work_amount)]);

    uni_vbroadcasti64x2(v_k_2_3, ptr[r64_k_ptr + K_2_3_OFF]);

    mov(r64_aux, reinterpret_cast<uintptr_t>(SHUF_MASK));
    uni_vbroadcasti64x2(v_shuf_mask, ptr[r64_aux]);

    if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FIRST_THREAD) {
        auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
        auto xmm_aux = getXmm();

        // Initial CRC
        mov(r64_aux, ptr[r64_params + GET_OFF(size)]);
        vpinsrq(xmm_aux, xmm_aux, r64_aux, 0x0);
        mov(r64_aux, CRC_VAL);
        vpinsrq(xmm_aux, xmm_aux, r64_aux, 0x1);

        // First xor with source.
        partial_load(v_dst, ptr[r64_src_ptr], r64_work_amount);
        vpshufb(v_dst, v_dst, v_shuf_mask);
        pxor(xmm_dst, xmm_aux);  // The SSE version is used to avoid zeroing out the rest of the Vmm.
        if (m_jcp.type == SINGLE_THREAD) {
            add(r64_src_ptr, xmm_len);
        }
    } else if (m_jcp.type == N_THREAD) {
        uni_vmovdqu64(v_dst, ptr[r64_src_ptr]);
        vpshufb(v_dst, v_dst, v_shuf_mask);
    }
    if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        sub(r64_work_amount, xmm_len);
    }
}

template <>
void ComputeHash<avx512_core>::bulk_fold(const Vmm& v_dst) {
    if (m_jcp.type != SINGLE_THREAD && m_jcp.type != FIRST_THREAD && m_jcp.type != N_THREAD) {
        return;
    }
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, static_cast<uint32_t>(get_vlen() * 2lu - xmm_len));
    jl(l_end, T_NEAR);

    auto v_src_0 = getVmm();
    auto v_dst_0 = getVmm();
    auto v_dst_1 = getVmm();
    auto v_dst_2 = getVmm();
    auto& v_dst_3 = v_dst;
    auto v_k_loop = getVmm();
    auto v_aux_0 = getVmm();

    auto xmm_src_0 = Xbyak::Xmm(v_src_0.getIdx());
    auto xmm_src_1 = getXmm();
    auto xmm_dst_0 = Xbyak::Xmm(v_dst_0.getIdx());
    auto xmm_dst_1 = Xbyak::Xmm(v_dst_1.getIdx());
    auto xmm_dst_2 = Xbyak::Xmm(v_dst_2.getIdx());
    auto xmm_dst_3 = Xbyak::Xmm(v_dst_3.getIdx());
    auto xmm_k_loop = Xbyak::Xmm(v_k_loop.getIdx());
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());
    auto xmm_aux_0 = Xbyak::Xmm(v_aux_0.getIdx());

    RegistersPool::Reg<Xbyak::Reg64> r64_bulk_step;
    if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        r64_bulk_step = getReg64();
        mov(r64_bulk_step, ptr[r64_params + GET_OFF(threads_num)]);
        sal(r64_bulk_step, static_cast<int>(std::log2(get_vlen())));  // * vlen
    }

    if (m_jcp.type == SINGLE_THREAD) {
        uni_vbroadcasti64x2(v_k_loop, ptr[r64_k_ptr + K_8_9_OFF]);
    } else {
        uni_vbroadcasti64x2(v_k_loop, ptr[r64_k_ptr + K_16_17_OFF]);
    }

    uni_vmovdqu64(v_dst_0, v_dst);

    if (!is_vpclmulqdq) {
        vextracti64x2(xmm_dst_1, v_dst_0, 0x1);
        vextracti64x2(xmm_dst_2, v_dst_0, 0x2);
        vextracti64x2(xmm_dst_3, v_dst_0, 0x3);
    }

    if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        add(r64_src_ptr, r64_bulk_step);
        prefetcht2(ptr[r64_src_ptr + 16384]);
    } else {
        add(r64_src_ptr, static_cast<uint32_t>(get_vlen() - xmm_len));
        prefetcht2(ptr[r64_src_ptr + 4096]);
    }
    prefetcht1(ptr[r64_src_ptr + 1024]);
    prefetcht0(ptr[r64_src_ptr + 64]);

    sub(r64_work_amount, static_cast<uint32_t>(get_vlen() * 2lu - xmm_len));

    L(l_fold_loop);
    {
        uni_vmovdqu64(v_src_0, ptr[r64_src_ptr]);
        vpshufb(v_src_0, v_src_0, v_shuf_mask);

        if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
            add(r64_src_ptr, r64_bulk_step);
            prefetcht2(ptr[r64_src_ptr + 16384]);
        } else {
            add(r64_src_ptr, static_cast<uint32_t>(get_vlen()));
            prefetcht2(ptr[r64_src_ptr + 4096]);
        }
        prefetcht1(ptr[r64_src_ptr + 1024]);
        prefetcht0(ptr[r64_src_ptr + 64]);

        if (is_vpclmulqdq) {
            vpclmulqdq(v_aux_0, v_dst_0, v_k_loop, 0b00000000);
            vpclmulqdq(v_dst_0, v_dst_0, v_k_loop, 0b00010001);
            uni_vpxorq(v_aux_0, v_aux_0, v_src_0);
            uni_vpxorq(v_dst_0, v_dst_0, v_aux_0);
        } else {
            // 0
            vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            uni_vpxorq(xmm_dst_0, xmm_dst_0, xmm_aux_0);

            // 1
            vextracti64x2(xmm_src_1, v_src_0, 0x1);
            vpclmulqdq(xmm_aux_0, xmm_dst_1, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_1, xmm_dst_1, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);

            // 2
            vextracti64x2(xmm_src_1, v_src_0, 0x2);
            vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            uni_vpxorq(xmm_dst_2, xmm_dst_2, xmm_aux_0);

            // 3
            vextracti64x2(xmm_src_1, v_src_0, 0x3);
            vpclmulqdq(xmm_aux_0, xmm_dst_3, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_3, xmm_dst_3, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        }

        sub(r64_work_amount, static_cast<uint32_t>(get_vlen()));
        jge(l_fold_loop, T_NEAR);
    }
    add(r64_work_amount, static_cast<uint32_t>(get_vlen()));

    if (m_jcp.type == SINGLE_THREAD) {
        if (is_vpclmulqdq) {
            vextracti64x2(xmm_dst_1, v_dst_0, 0x1);
            vextracti64x2(xmm_dst_2, v_dst_0, 0x2);
            vextracti64x2(xmm_dst_3, v_dst_0, 0x3);
        }

        vpclmulqdq(xmm_aux_0, xmm_dst_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00000000);
        vpclmulqdq(xmm_dst_0, xmm_dst_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00010001);
        uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);

        vpclmulqdq(xmm_aux_0, xmm_dst_1, ptr[r64_k_ptr + K_4_5_OFF], 0b00000000);
        vpclmulqdq(xmm_dst_1, xmm_dst_1, ptr[r64_k_ptr + K_4_5_OFF], 0b00010001);
        uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_1);

        vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_2_3, 0b00000000);
        vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_2_3, 0b00010001);
        uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_2);
    } else {
        if (is_vpclmulqdq) {
            uni_vmovdqu64(ptr[r64_dst_ptr], v_dst_0);
        } else {
            uni_vmovdqu64(ptr[r64_dst_ptr + xmm_len * 0lu], xmm_dst_0);
            uni_vmovdqu64(ptr[r64_dst_ptr + xmm_len * 1lu], xmm_dst_1);
            uni_vmovdqu64(ptr[r64_dst_ptr + xmm_len * 2lu], xmm_dst_2);
            uni_vmovdqu64(ptr[r64_dst_ptr + xmm_len * 3lu], xmm_dst_3);
        }
    }

    L(l_end);
}

template <cpu_isa_t isa>
void ComputeHash<isa>::bulk_fold(const Vmm& v_dst) {
    if (m_jcp.type != SINGLE_THREAD && m_jcp.type != FIRST_THREAD && m_jcp.type != N_THREAD) {
        return;
    }
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, static_cast<uint32_t>(get_vlen() * 2lu - xmm_len));
    jl(l_end, T_NEAR);

    auto v_src_0 = getVmm();
    auto v_dst_0 = getVmm();
    auto& v_dst_1 = v_dst;
    auto v_aux_0 = getVmm();
    auto v_k_loop = getVmm();

    auto xmm_src_0 = Xbyak::Xmm(v_src_0.getIdx());
    auto xmm_src_1 = getXmm();
    auto xmm_dst_0 = Xbyak::Xmm(v_dst_0.getIdx());
    auto xmm_dst_1 = Xbyak::Xmm(v_dst_1.getIdx());
    auto xmm_k_loop = Xbyak::Xmm(v_k_loop.getIdx());
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());
    auto xmm_aux_0 = Xbyak::Xmm(v_aux_0.getIdx());

    RegistersPool::Reg<Xbyak::Reg64> r64_bulk_step;
    if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        r64_bulk_step = getReg64();
        mov(r64_bulk_step, ptr[r64_params + GET_OFF(threads_num)]);
        sal(r64_bulk_step, static_cast<int>(std::log2(get_vlen())));  // * vlen
    }

    if (m_jcp.type == SINGLE_THREAD) {
        uni_vbroadcasti64x2(v_k_loop, ptr[r64_k_ptr + K_4_5_OFF]);
    } else {
        uni_vbroadcasti64x2(v_k_loop, ptr[r64_k_ptr + K_8_9_OFF]);
    }

    uni_vmovdqu64(v_dst_0, v_dst);

    if (!is_vpclmulqdq) {
        vextracti128(xmm_dst_1, v_dst_0, 0x1);
    }

    if (m_jcp.type == SINGLE_THREAD) {
        add(r64_src_ptr, static_cast<uint32_t>(get_vlen() - xmm_len));
    } else {
        add(r64_src_ptr, r64_bulk_step);
    }
    prefetcht2(ptr[r64_src_ptr + 4096]);
    prefetcht1(ptr[r64_src_ptr + 1024]);
    prefetcht0(ptr[r64_src_ptr + 64]);

    sub(r64_work_amount, static_cast<uint32_t>(get_vlen() * 2lu - xmm_len));

    L(l_fold_loop);
    {
        uni_vmovdqu64(v_src_0, ptr[r64_src_ptr]);
        vpshufb(v_src_0, v_src_0, v_shuf_mask);

        if (m_jcp.type == SINGLE_THREAD) {
            add(r64_src_ptr, static_cast<uint32_t>(get_vlen()));
        } else {
            add(r64_src_ptr, r64_bulk_step);
        }
        prefetcht2(ptr[r64_src_ptr + 4096]);
        prefetcht1(ptr[r64_src_ptr + 1024]);
        prefetcht0(ptr[r64_src_ptr + 64]);

        if (is_vpclmulqdq) {
            vpclmulqdq(v_aux_0, v_dst_0, v_k_loop, 0b00000000);
            vpclmulqdq(v_dst_0, v_dst_0, v_k_loop, 0b00010001);
            uni_vpxorq(v_aux_0, v_aux_0, v_src_0);
            uni_vpxorq(v_dst_0, v_dst_0, v_aux_0);
        } else {
            // 0
            vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            uni_vpxorq(xmm_dst_0, xmm_dst_0, xmm_aux_0);
            // 1
            vextracti128(xmm_src_1, v_src_0, 0x1);
            vpclmulqdq(xmm_aux_0, xmm_dst_1, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_1, xmm_dst_1, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);
        }

        sub(r64_work_amount, static_cast<uint32_t>(get_vlen()));
        jge(l_fold_loop, T_NEAR);
    }
    add(r64_work_amount, static_cast<uint32_t>(get_vlen()));

    if (m_jcp.type == SINGLE_THREAD) {
        if (is_vpclmulqdq) {
            vextracti128(xmm_dst_1, v_dst_0, 0x1);
        }
        vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_2_3, 0b00000000);
        vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_2_3, 0b00010001);
        uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);
        uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_dst_0);
    } else {
        if (is_vpclmulqdq) {
            uni_vmovdqu64(ptr[r64_dst_ptr], v_dst_0);
        } else {
            uni_vmovdqu64(ptr[r64_dst_ptr + xmm_len * 0lu], xmm_dst_0);
            uni_vmovdqu64(ptr[r64_dst_ptr + xmm_len * 1lu], xmm_dst_1);
        }
    }

    L(l_end);
}

template <>
void ComputeHash<avx512_core>::join(const Vmm& v_dst) {
    if (m_jcp.type != FINAL_FOLD) {
        return;
    }

    mov(r64_aux, ptr[r64_params + GET_OFF(intermediate_ptr)]);
    prefetcht0(ptr[r64_aux + 1024]);

    auto xmm_src_0 = getXmm();
    auto xmm_src_last = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux_0 = getXmm();
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());

    uni_vmovdqu64(xmm_src_last, ptr[r64_aux + xmm_len * 7]);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_14_15_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_14_15_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_12_13_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_12_13_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 2lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_10_11_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_10_11_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 3lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_8_9_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_8_9_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 4lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 5lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_4_5_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_4_5_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 6lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, xmm_k_2_3, 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, xmm_k_2_3, 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);
}

template <cpu_isa_t isa>
void ComputeHash<isa>::join(const Vmm& v_dst) {
    if (m_jcp.type != FINAL_FOLD) {
        return;
    }

    mov(r64_aux, ptr[r64_params + GET_OFF(intermediate_ptr)]);
    prefetcht0(ptr[r64_aux + 1024]);

    auto xmm_src_0 = getXmm();
    auto xmm_src_last = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux_0 = getXmm();
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());

    uni_vmovdqu64(xmm_src_last, ptr[r64_aux + xmm_len * 3]);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 0lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 1lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_4_5_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_4_5_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 2lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, xmm_k_2_3, 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, xmm_k_2_3, 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);
}

template <cpu_isa_t isa>
void ComputeHash<isa>::fold_to_128(const Vmm& v_dst) {
    if (m_jcp.type != SINGLE_THREAD && m_jcp.type != FINAL_FOLD) {
        return;
    }
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, xmm_len);
    jl(l_end, T_NEAR);

    auto xmm_src = getXmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_aux = getXmm();

    L(l_fold_loop);
    {
        uni_vmovdqu64(xmm_src, ptr[r64_src_ptr]);
        vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

        vpclmulqdq(xmm_aux, xmm_dst, xmm_k_2_3, 0b00000000);
        vpclmulqdq(xmm_dst, xmm_dst, xmm_k_2_3, 0b00010001);
        uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);
        uni_vpxorq(xmm_dst, xmm_dst, xmm_src);

        add(r64_src_ptr, xmm_len);
        sub(r64_work_amount, xmm_len);
        cmp(r64_work_amount, xmm_len);
        jge(l_fold_loop, T_NEAR);
    }

    L(l_end);
}

template <cpu_isa_t isa>
void ComputeHash<isa>::fold_to_64(const Vmm& v_dst) {
    if (m_jcp.type != SINGLE_THREAD && m_jcp.type != FINAL_FOLD) {
        return;
    }
    Xbyak::Label l_fold_to_64;
    cmp(r64_work_amount, 0);
    jle(l_fold_to_64, T_NEAR);

    auto xmm_src = getXmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_aux = getXmm();
    auto xmm_aux_1 = getXmm();
    auto xmm_aux_2 = getXmm();

    partial_load(xmm_src, ptr[r64_src_ptr], r64_work_amount);
    vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

    vpclmulqdq(xmm_aux, xmm_dst, xmm_k_2_3, 0b00000000);
    vpclmulqdq(xmm_dst, xmm_dst, xmm_k_2_3, 0b00010001);
    uni_vpxorq(xmm_aux, xmm_aux, xmm_src);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    L(l_fold_to_64);

    mov(r64_aux, K_2);
    vpinsrq(xmm_aux, xmm_aux, r64_aux, 0x0);
    vpclmulqdq(xmm_aux, xmm_dst, xmm_aux, 0b00000001);
    vpslldq(xmm_dst, xmm_dst, 0x8);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    mov(r64_aux, P_1);
    vpinsrq(xmm_aux_2, xmm_aux_2, r64_aux, 0x0);
    vpclmulqdq(xmm_aux, xmm_dst, xmm_aux_2, 0b00000001);
    mov(r64_aux, 0x0);
    vpinsrq(xmm_aux_1, xmm_dst, r64_aux, 0x0);
    uni_vpxorq(xmm_aux, xmm_aux, xmm_aux_1);
    vpinsrq(xmm_aux_1, xmm_aux, r64_aux, 0x0);

    mov(r64_aux, P_2);
    vpinsrq(xmm_aux_2, xmm_aux_2, r64_aux, 0x1);
    vpclmulqdq(xmm_aux, xmm_aux, xmm_aux_2, 0b00010001);
    uni_vpxorq(xmm_aux, xmm_aux, xmm_aux_1);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    vpextrq(ptr[r64_dst_ptr], xmm_dst, 0x0);
}

}  // namespace jit
#endif  // OV_CORE_USE_XBYAK_JIT

size_t compute_hash(const void* src, size_t size) {
#ifdef OV_CORE_USE_XBYAK_JIT
    if (util::may_i_use_dynamic_code()) {
        if (Generator::mayiuse(avx2)) {
            uint64_t result = 0lu;

            // Parallel section
            constexpr uint64_t min_wa_per_thread = 131072lu;  // 2^17
            const uint64_t size_u64 = static_cast<uint64_t>(size);
            if (size_u64 >= min_wa_per_thread * 2lu) {
                static auto first_thr_kernel = Generator::mayiuse(avx512_core)
                                                   ? jit::ComputeHash<avx512_core>::create({jit::FIRST_THREAD})
                                                   : jit::ComputeHash<avx2>::create({jit::FIRST_THREAD});
                static auto n_thr_kernel = Generator::mayiuse(avx512_core)
                                               ? jit::ComputeHash<avx512_core>::create({jit::N_THREAD})
                                               : jit::ComputeHash<avx2>::create({jit::N_THREAD});
                static auto final_fold_kernel = Generator::mayiuse(avx512_core)
                                                    ? jit::ComputeHash<avx512_core>::create({jit::FINAL_FOLD})
                                                    : jit::ComputeHash<avx2>::create({jit::FINAL_FOLD});

                static const uint64_t max_thr_num = 2lu;
                uint64_t thr_num = std::min(size_u64 / min_wa_per_thread, max_thr_num);
                const uint64_t el_per_thread =
                    first_thr_kernel->get_vlen() * ((size_u64 / thr_num) / first_thr_kernel->get_vlen());
                std::vector<uint8_t> intermediate(thr_num * first_thr_kernel->get_vlen());

                parallel_nt_static(static_cast<int>(thr_num), [&](const int ithr, const int nthr) {
                    uint64_t start = el_per_thread * ithr;
                    if (start >= size_u64) {
                        return;
                    }
                    uint64_t work_amount = (el_per_thread + start > size_u64) ? size_u64 - start : el_per_thread;

                    jit::ComputeHashCallArgs args;

                    args.src_ptr = reinterpret_cast<const uint8_t*>(src) + first_thr_kernel->get_vlen() * ithr;
                    args.dst_ptr = &(intermediate[first_thr_kernel->get_vlen() * ithr]);
                    args.k_ptr = jit::K_PULL;
                    args.work_amount = work_amount;
                    args.size = size_u64;
                    args.threads_num = thr_num;

                    if (ithr == 0) {
                        (*first_thr_kernel)(&args);
                    } else {
                        (*n_thr_kernel)(&args);
                    }
                });

                jit::ComputeHashCallArgs args;
                args.work_amount = size_u64 - el_per_thread * thr_num;
                args.src_ptr = reinterpret_cast<const uint8_t*>(src) + size_u64 - args.work_amount;
                args.dst_ptr = &result;
                args.k_ptr = jit::K_PULL;
                args.size = size_u64;
                args.intermediate_ptr = intermediate.data();

                (*final_fold_kernel)(&args);
            } else {
                static auto single_thr_kernel = Generator::mayiuse(avx512_core)
                                                    ? jit::ComputeHash<avx512_core>::create({jit::SINGLE_THREAD})
                                                    : jit::ComputeHash<avx2>::create({jit::SINGLE_THREAD});

                jit::ComputeHashCallArgs args;
                args.src_ptr = src;
                args.dst_ptr = &result;
                args.k_ptr = jit::K_PULL;
                args.work_amount = size_u64;
                args.size = size_u64;

                (*single_thr_kernel)(&args);
            }

            return result;
        }
    }

#endif  // OV_CORE_USE_XBYAK_JIT

    constexpr auto cel_size = sizeof(size_t);
    size_t seed = size;
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

}  // namespace runtime
}  // namespace ov
