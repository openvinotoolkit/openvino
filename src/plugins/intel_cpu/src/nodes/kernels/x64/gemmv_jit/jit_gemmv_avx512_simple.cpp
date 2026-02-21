// Implementation of simple AVX-512 GEMM-v (u8 weights, fp32 input, no scales/bias)

#include "jit_gemmv_avx512_simple.hpp"

#include "jit_prebuilt_pool.hpp"
#include "openvino/core/except.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

jit_gemmv_avx512_simple_kernel::jit_gemmv_avx512_simple_kernel()
    : dnnl::impl::cpu::x64::jit_generator_t("jit_gemmv_avx512_simple_kernel",
            dnnl::impl::cpu::x64::cpu_isa_t::avx512_core) {
    auto st = create_kernel();
    if (st != dnnl::impl::status::success) {
        OPENVINO_THROW("Failed to build jit_gemmv_avx512_simple kernel");
    }
    fn_ = reinterpret_cast<fn_t>(jit_ker());
}

void jit_gemmv_avx512_simple_kernel::generate() {
    using namespace Xbyak;
    setDefaultJmpNEAR(true);
#if defined(OPENVINO_ARCH_X86_64)
    endbr64();
#endif
    // Prologue
    push(r12); push(r13); push(r14); push(r15);
    Reg64 reg_args = rdi;
    Reg64 reg_x = r8;
    Reg64 reg_w = r9;
    Reg64 reg_y = r13;
    Reg64 reg_k = r14;
    Reg64 reg_it = r15;

    Zmm zC = zmm0, zW = zmm1, zX = zmm2; // accum, weights, x-bcast
    Zmm zS = zmm15, zZP = zmm4, zTmp = zmm5, zComp = zmm6;
    Xmm xWb = xmm7;  // 16 bytes (legacy path)
    Xmm xWlo = xmm12; // new path: low 8 bytes
    Xmm xWhi = xmm13; // new path: high 8 bytes
    Ymm yIlo = ymm8, yIhi = ymm9;
    Ymm yFlo = ymm10, yFhi = ymm11;
    // no k-mask path (benchmark allocates Y with M_pad)

    // Load args
    mov(reg_x, ptr[reg_args + offsetof(gemmv_avx512_simple_call_args, x)]);
    mov(reg_w, ptr[reg_args + offsetof(gemmv_avx512_simple_call_args, wq)]);
    mov(reg_y, ptr[reg_args + offsetof(gemmv_avx512_simple_call_args, y)]);
    mov(eax, dword[reg_args + offsetof(gemmv_avx512_simple_call_args, K)]);
    mov(reg_k, rax);

    // No tail mask setup

    // Load pre-expanded scales vector svec[16]
    mov(rax, ptr[reg_args + offsetof(gemmv_avx512_simple_call_args, svec)]);
    vmovups(zS, ptr[rax]);
    // svec is always provided by host code; no fallback needed

    // C = 0
    vxorps(zC, zC, zC);

    // for k in [0..K)
    xor_(reg_it, reg_it);
    Label Lloop, Ldone;
    L(Lloop);
    cmp(reg_it, reg_k);
    jge(Ldone);

    // load 16 bytes of W as two 8-byte chunks, expand to fp32
    vmovq(xWlo, ptr[reg_w]);        // low  8 bytes
    vmovq(xWhi, ptr[reg_w + 8]);    // high 8 bytes
    add(reg_w, 16);
    // sign/zero by w_is_u8
    {
        Label L_is_i8, L_cvt_done;
        mov(eax, dword[reg_args + offsetof(gemmv_avx512_simple_call_args, w_is_u8)]);
        test(eax, eax);
        jz(L_is_i8);
        vpmovzxbd(yIlo, xWlo);
        vcvtdq2ps(yFlo, yIlo);
        vpmovzxbd(yIhi, xWhi);
        vcvtdq2ps(yFhi, yIhi);
        jmp(L_cvt_done);
        L(L_is_i8);
        vpmovsxbd(yIlo, xWlo);
        vcvtdq2ps(yFlo, yIlo);
        vpmovsxbd(yIhi, xWhi);
        vcvtdq2ps(yFhi, yIhi);
        L(L_cvt_done);
    }
    vxorps(zW, zW, zW);
    vinsertf32x8(zW, zW, yFlo, 0);
    vinsertf32x8(zW, zW, yFhi, 1);
    // pack-only debug: directly accumulate unpacked W and continue
    {
        Label L_after_pack_only;
        mov(eax, dword[reg_args + offsetof(gemmv_avx512_simple_call_args, pack_only)]);
        test(eax, eax);
        jz(L_after_pack_only);
        vaddps(zC, zC, zW);
        inc(reg_it);
        jmp(Lloop);
        L(L_after_pack_only);
    }
    // dequant: optionally w *= s (skip via debug flag)
    {
        Label L_after_mul;
        mov(eax, dword[reg_args + offsetof(gemmv_avx512_simple_call_args, skip_mul)]);
        test(eax, eax);
        jnz(L_after_mul);
        vmulps(zW, zW, zS);
        L(L_after_mul);
    }

    // x broadcast (or const 1.0f if debug x_const_one)
    {
        Label L_use_mem, L_x_done;
        mov(eax, dword[reg_args + offsetof(gemmv_avx512_simple_call_args, x_const_one)]);
        test(eax, eax);
        jz(L_use_mem);
        mov(eax, 0x3f800000); // 1.0f
        movd(xmm0, eax);
        vpbroadcastd(zX, xmm0);
        jmp(L_x_done);
        L(L_use_mem);
        vbroadcastss(zX, dword[reg_x + reg_it*4]);
        L(L_x_done);
    }
    {
        Label L_do_add;
        Label L_acc_done;
        mov(eax, dword[reg_args + offsetof(gemmv_avx512_simple_call_args, no_fma)]);
        test(eax, eax);
        jz(L_do_add);
        // no_fma==0 -> use FMA
        vfmadd231ps(zC, zW, zX);
        jmp(L_acc_done);
        L(L_do_add);
        // no_fma != 0 -> MUL + ADD
        vmulps(zTmp, zW, zX);
        vaddps(zC, zC, zTmp);
        L(L_acc_done);
    }

    inc(reg_it);
    jmp(Lloop);
    L(Ldone);

    // apply bias and zp compensation: C += bias - s*zp*sum_x
    // Add bias vector if provided: y += bvec
    {
        Label L_bias_skip;
        Label L_bias_done;
        mov(rax, ptr[reg_args + offsetof(gemmv_avx512_simple_call_args, bvec)]);
        test(rax, rax); jz(L_bias_skip);
        vmovups(zTmp, ptr[rax]);
        vaddps(zC, zC, zTmp);
        jmp(L_bias_done);
        L(L_bias_skip);
        L(L_bias_done);
    }
    // Subtract precomputed zcomp if provided: y -= zcomp
    {
        Label L_zcomp_skip;
        Label L_zcomp_done;
        mov(rax, ptr[reg_args + offsetof(gemmv_avx512_simple_call_args, zcomp)]);
        test(rax, rax); jz(L_zcomp_skip);
        vmovups(zTmp, ptr[rax]);
        vsubps(zC, zC, zTmp);
        jmp(L_zcomp_done);
        L(L_zcomp_skip);
        L(L_zcomp_done);
    }

    vmovups(ptr[reg_y], zC);

    pop(r15); pop(r14); pop(r13); pop(r12);
    ret();
}

JitGemmvAvx512Simple::JitGemmvAvx512Simple() {
    fn_ = jit_prebuilt_pool::get_typed<fn_t>(kernel_kind::avx512_simple);
    OPENVINO_ASSERT(fn_ != nullptr, "jit_gemmv_avx512_simple kernel pointer is null");
}

void JitGemmvAvx512Simple::operator()(const gemmv_ukr_params_t* p) const {
    const int M_blk = 16;
    const int full = p->M / M_blk;
    const int tail = p->M % M_blk;
    gemmv_avx512_simple_call_args a{};
    a.x = static_cast<const float*>(p->x);
    a.K = p->K;
    a.gran = static_cast<int>(p->gran);
    if (const char* ft = std::getenv("GEMMV_FORCE_PER_TENSOR")) {
        if (std::atoi(ft) != 0) a.gran = static_cast<int>(quant_granularity_t::per_tensor);
    }
    a.w_is_u8 = (p->w_type == w_dtype_t::u8) ? 1 : 0;
    // mode: from env GEMMV_JIT_SIMPLE_MODE or default 3
    int mode = 3;
    if (const char* m = std::getenv("GEMMV_JIT_SIMPLE_MODE")) {
        int v = std::atoi(m);
        if (v >= 0 && v <= 3) mode = v;
    }
    a.mode = mode;
    // sum_x for zp
    float sumx = 0.f;
    if (p->zps) {
        const float* xf = static_cast<const float*>(p->x);
        for (int k = 0; k < p->K; ++k) sumx += xf[k];
    }
    a.sum_x = sumx;
    if (const char* dbg = std::getenv("GEMMV_DEBUG")) {
        if (std::atoi(dbg) != 0) {
            fprintf(stderr, "[jit_simple] x=%p K=%d wq=%p ldw=%d y=%p M=%d gran=%d w_is_u8=%d mode=%d\n",
                a.x, a.K, p->wq, p->ld_w_bytes, p->y, p->M, a.gran, a.w_is_u8, a.mode);
        }
    }
    // debug flag to skip mul inside jit
    a.skip_mul = 0;
    if (const char* sk = std::getenv("GEMMV_JIT_SIMPLE_SKIP_MUL")) {
        if (std::atoi(sk) != 0) a.skip_mul = 1;
    }
    a.x_const_one = 0;
    if (const char* xc = std::getenv("GEMMV_JIT_SIMPLE_XCONST")) {
        if (std::atoi(xc) != 0) a.x_const_one = 1;
    }
    a.no_fma = 0;
    if (const char* nf = std::getenv("GEMMV_JIT_SIMPLE_NOFMA")) {
        if (std::atoi(nf) != 0) a.no_fma = 1;
    }
    a.pack_only = 0;
    if (const char* po = std::getenv("GEMMV_JIT_SIMPLE_PACKONLY")) {
        if (std::atoi(po) != 0) a.pack_only = 1;
    }
    float* y = static_cast<float*>(p->y);
    for (int bi = 0; bi < full; ++bi) {
        a.wq = p->wq + bi * p->ld_w_bytes;
        a.y = y + bi * M_blk;
        // Per-channel pointers must advance with block
        alignas(64) float svec[16];
        alignas(64) float bvec[16];
        alignas(64) float zcvec[16];
        const bool per_ch = (p->gran == quant_granularity_t::per_channel);
        const float* sc_base = p->scales ? (per_ch ? p->scales + bi * M_blk : p->scales) : nullptr;
        const float* b_base  = p->bias   ? (per_ch ? p->bias   + bi * M_blk : p->bias)   : nullptr;
        const int32_t* zp_base = p->zps  ? (per_ch ? p->zps    + bi * M_blk : p->zps)    : nullptr;
        // mode handling
        for (int m = 0; m < M_blk; ++m) {
            float s = 1.0f;
            if (a.mode >= 1 && sc_base) s = per_ch ? sc_base[m] : sc_base[0];
            svec[m] = s;
            float bb = 0.0f;
            if (a.mode >= 2 && b_base) bb = per_ch ? b_base[m] : b_base[0];
            bvec[m] = bb;
            float zc = 0.0f;
            if (a.mode >= 3 && zp_base) {
                float zp = (float)(per_ch ? zp_base[m] : zp_base[0]);
                zc = s * zp * a.sum_x;
            }
            zcvec[m] = zc;
        }
        a.svec = svec;
        a.bvec = (a.mode >= 2 && b_base) ? bvec : nullptr;
        a.zcomp = (a.mode >= 3 && zp_base) ? zcvec : nullptr;
        if (const char* dbg = std::getenv("GEMMV_DEBUG")) {
            if (std::atoi(dbg) > 1) {
                fprintf(stderr, "[jit_simple] bi=%d svec=%p bvec=%p zcomp=%p\n", bi, (const void*)a.svec, (const void*)a.bvec, (const void*)a.zcomp);
            }
        }
        a.M_tail = 0; // full
        fn_(&a);
    }
    if (tail) {
        const int bi = full;
        a.wq = p->wq + bi * p->ld_w_bytes;
        a.y = y + bi * M_blk;
        alignas(64) float svec2[16];
        alignas(64) float bvec2[16];
        alignas(64) float zcvec2[16];
        const bool per_ch2 = (p->gran == quant_granularity_t::per_channel);
        const float* sc_base2 = p->scales ? (per_ch2 ? p->scales + bi * M_blk : p->scales) : nullptr;
        const float* b_base2  = p->bias   ? (per_ch2 ? p->bias   + bi * M_blk : p->bias)   : nullptr;
        const int32_t* zp_base2 = p->zps  ? (per_ch2 ? p->zps    + bi * M_blk : p->zps)    : nullptr;
        for (int m = 0; m < M_blk; ++m) {
            float s = 1.0f;
            if (a.mode >= 1 && sc_base2) s = per_ch2 ? sc_base2[m] : sc_base2[0];
            svec2[m] = s;
            float bb = 0.0f;
            if (a.mode >= 2 && b_base2) bb = per_ch2 ? b_base2[m] : b_base2[0];
            bvec2[m] = bb;
            float zc = 0.0f;
            if (a.mode >= 3 && zp_base2) {
                float zp = (float)(per_ch2 ? zp_base2[m] : zp_base2[0]);
                zc = s * zp * a.sum_x;
            }
            zcvec2[m] = zc;
        }
        a.svec = svec2;
        a.bvec = (a.mode >= 2 && b_base2) ? bvec2 : nullptr;
        a.zcomp = (a.mode >= 3 && zp_base2) ? zcvec2 : nullptr;
        a.M_tail = tail;
        fn_(&a);
    }
}

} // namespace ov::intel_cpu::x64::gemmv_jit
