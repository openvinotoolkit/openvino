// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstring>
#include "act_sparse_fc_kernel.hpp"
#include "jit_kernel_base.hpp"

#include "openvino/core/parallel.hpp"

//#include "/home/tingqian/aboutSHW/include/linux_perf.hpp"
//#include "/home/openvino-ci-58/tingqian/aboutSHW/include/linux_perf.hpp"

#define PROFILE(x) LinuxPerf::Profile(x)
#define PROFILE(x) 1

#include "simd.hpp"

// https://github.com/intel-sandbox/dynSparseFC/blob/main/dyn_sparse_fc.cpp

#ifndef ASSERT
#    define ASSERT(cond)                                                     \
        if (!(cond)) {                                                       \
            std::stringstream ss;                                            \
            ss << __FILE__ << ":" << __LINE__ << " " << #cond << " failed!"; \
            throw std::runtime_error(ss.str());                              \
        }
#endif

#ifdef _WIN32
#define abi_param_regs_num 4
#else
#define abi_param_regs_num 6
#endif
// first few regs contains input arguments passed in through stack
constexpr Xbyak::Operand::Code abi_x86_64_regs[] = {
#ifdef _WIN32
        Xbyak::Operand::RCX, Xbyak::Operand::RDX, Xbyak::Operand::R8,  Xbyak::Operand::R9, // args passed in register
        Xbyak::Operand::RDI, Xbyak::Operand::RSI,

        Xbyak::Operand::RBX, Xbyak::Operand::RBP, Xbyak::Operand::R10, Xbyak::Operand::R11, Xbyak::Operand::R12,
        Xbyak::Operand::R13, Xbyak::Operand::R14, Xbyak::Operand::R15
#else
        Xbyak::Operand::RDI, Xbyak::Operand::RSI, Xbyak::Operand::RDX, Xbyak::Operand::RCX, Xbyak::Operand::R8, Xbyak::Operand::R9, // args passed in register

        Xbyak::Operand::RBX, Xbyak::Operand::RBP, Xbyak::Operand::R10, Xbyak::Operand::R11, Xbyak::Operand::R12,
        Xbyak::Operand::R13, Xbyak::Operand::R14, Xbyak::Operand::R15
#endif
};

class JitKernel : public ov::intel_cpu::kernel::JitKernelBase {
public:
  DECLARE_CPU_JIT_AUX_FUNCTIONS(JitKernel);
  bool use_avx512;

#if defined(HAVE_AVX512F)
  JitKernel(const char* name) : JitKernelBase(name, dnnl::impl::cpu::x64::cpu_isa_t::avx512_core) {
    use_avx512 = true;
    mov(rax, rsp);
    JitKernelBase::preamble();
  }
#else
  JitKernel(const char* name) : JitKernelBase(name, dnnl::impl::cpu::x64::cpu_isa_t::avx2) {
    use_avx512 = false;
    mov(rax, rsp);
    JitKernelBase::preamble();
  }
#endif

  void generate() override {};

  // add an int64_t return value
  template <typename... kernel_args_t>
  int64_t operator()(kernel_args_t... args) const {
    using jit_kernel_func_t = int64_t (*)(const kernel_args_t... args);
    auto *fptr = (jit_kernel_func_t)jit_ker();
    return (*fptr)(std::forward<kernel_args_t>(args)...);
  }

  void finalize(Xbyak::Reg64 return_value = {}) {
    if (!return_value.isNone())
        mov(rax, return_value);
    JitKernelBase::postamble();
    JitKernelBase::create_kernel();
  }

  Xbyak::Reg64 get_sreg(int i, bool is_arg = false) {
    if (i < abi_param_regs_num)
        return Xbyak::Reg64(abi_x86_64_regs[i]);
    if (i >= sizeof(abi_x86_64_regs)/sizeof(abi_x86_64_regs[0]))
        throw std::runtime_error(std::string("try to allocate invalid scalar register #") + std::to_string(i));

    auto r = Xbyak::Reg64(abi_x86_64_regs[i]);
    if (is_arg)
        mov(r, ptr[rax + (i - abi_param_regs_num + 1)*8]);// load from stack
    return r;
  }

  Xbyak::Xmm Vmm(int id) {
    if (use_avx512) {
        if (id >= 32)
            throw std::runtime_error(std::string("try to use invalid zmm register: #") + std::to_string(id));
        return Xbyak::Zmm(id);
    } else {
        if (id >= 16)
            throw std::runtime_error(std::string("try to use invalid ymm register: #") + std::to_string(id));
        return Xbyak::Ymm(id);
    }
  }
  void simd_setzero_ps(Xbyak::Xmm vmm) {
    if (use_avx512) {
        vpxord(vmm, vmm, vmm);
    } else {
        vpxor(vmm, vmm, vmm);
    }
  }
  void simd_loadu_ps(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
    vmovups(vmm, addr);
  }
  // load packed half into packed single
  void simd_loadu_phps(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
    if (use_avx512) {
        auto vreg_256 = Xbyak::Ymm(vmm.getIdx());
        vmovdqu(vreg_256, addr);
        vcvtph2ps(vmm, vreg_256);
    } else {
        auto vreg_128 = Xbyak::Xmm(vmm.getIdx());
        movdqu(vreg_128, addr);
        vcvtph2ps(vmm, vreg_128);
    }
  }
  void simd_load_epu8_epi32(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
    vpmovzxbd(vmm, addr);
  }
  void simd_load_epi8_epi32(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
    vpmovsxbd(vmm, addr);
  }
  void simd_storeu_ps(const Xbyak::Address& addr, Xbyak::Xmm vmm) {
    vmovups(addr, vmm);
  }
  void simd_fmadd_ps(Xbyak::Xmm c, Xbyak::Xmm a, const Xbyak::Operand& b) {
    vfmadd231ps(c, a, b);
  }
  void simd_sub_ps(Xbyak::Xmm c, Xbyak::Xmm a, Xbyak::Xmm b) {
    vsubps(c, a, b);
  }
  void simd_mul_ps(Xbyak::Xmm c, Xbyak::Xmm a, Xbyak::Xmm b) {
    vmulps(c, a, b);
  }
  void simd_broadcast_ss(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
    vbroadcastss(vmm, addr);
  }
  void simd_cvtepi32_ps(Xbyak::Xmm vmm_dst, Xbyak::Xmm vmm_src) {
    vcvtdq2ps(vmm_dst, vmm_src);
  }

  // for_loop() performs following:
  //    for(int idx=0; idx + step <= cnt; idx+=step) {
  //       loop_body();
  //    }
  template<typename Fn, typename STEP>
  void for_loop(Xbyak::Reg64 idx, Xbyak::Reg64 cnt, STEP step, const Fn& loop_body) {
    Xbyak::Label loop, exit;
    mov(idx, 0);

    L(loop);
    add(idx, step);
    cmp(idx, cnt);
    jg(exit, T_NEAR);
    sub(idx, step);

    loop_body();
    add(idx, step);

    jmp(loop, T_NEAR);
    L(exit);
    // at exit, idx is pointing to tail
    sub(idx, step);
  }

  template<typename DT>
  size_t vmm_width() {
    return Vmm(0).getBit()/(sizeof(DT) * 8);
  }
};

/*
    for (int k = 0; k < bK; k++, dst += 2 * SIMDW, srcW += W_stride) {
        auto wf0 = simd_loadu_ps(srcW + SIMDW * 0);
        auto wf1 = simd_loadu_ps(srcW + SIMDW * 1);
        // prefetch right
        simd_prefetch(srcW + 64, _MM_HINT_T0);
        simd_storeu_ps(dst + SIMDW * 0, wf0);
        simd_storeu_ps(dst + SIMDW * 1, wf1);
    }
*/
static std::shared_ptr<JitKernel> jit_compile_gemmRegBlk(int rows, int cols, int prefetch_B_adv = 0) {
    auto jit = std::make_shared<JitKernel>(__func__);
    auto simd_width_bytes = jit->vmm_width<uint8_t>();

    auto is_preload_b = (rows >= cols);
    auto vmmC = [&](int row, int col) {
        return jit->Vmm(row * cols + col);
    };
    auto vmmB = [&](int col) {
        if (is_preload_b)
            return jit->Vmm(rows * cols + col);
        else
            return jit->Vmm(rows * cols);
    };
    auto vmmA = [&](int row) {
        if (is_preload_b)
            return jit->Vmm(rows * cols + cols);
        else
            return jit->Vmm(rows * cols + 1 + row);
    };

    // load all arguments into register
    auto A_ptr = jit->get_sreg(0, true);
    auto A_stride = jit->get_sreg(1, true);
    auto B_ptr = jit->get_sreg(2, true);
    auto B_stride = jit->get_sreg(3, true);
    auto dst_ptr = jit->get_sreg(4, true);
    auto dst_stride = jit->get_sreg(5, true);
    auto K = jit->get_sreg(6, true);
    auto accumulate = jit->get_sreg(7, true);
    auto stemp = jit->get_sreg(8);

    jit->lea(A_stride, jit->ptr[A_stride*4]);
    jit->lea(B_stride, jit->ptr[B_stride*4]);
    jit->lea(dst_stride, jit->ptr[dst_stride*4]);
    // initilaize C
    {
        Xbyak::Label skip_load;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                auto ymm = vmmC(r, c);
                jit->vxorps(ymm, ymm, ymm);
            }
        jit->and_(accumulate, 1);
        jit->jz(skip_load);
        {
            // load subC[m_rows, m_cols]
            jit->mov(stemp, dst_ptr);
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    jit->simd_loadu_ps(vmmC(r, c), jit->ptr[stemp + c * simd_width_bytes]);
                }
                jit->add(stemp, dst_stride);
            }
        }
        jit->L(skip_load);
    }

    // loop over K
    //            B:    1 x cols regs
    // A : 1 regs C: rows x cols regs
    {
        Xbyak::Label loop_over_k;
        auto A_ptr3 = accumulate; // accumulate can be re-used

        auto loadA = [&](int r) {
            switch (r) {
            case 0:
                jit->simd_broadcast_ss(vmmA(r), jit->ptr[A_ptr]);
                break;
            case 1:
                jit->simd_broadcast_ss(vmmA(r), jit->ptr[A_ptr + A_stride]);
                break;
            case 2:
                jit->simd_broadcast_ss(vmmA(r), jit->ptr[A_ptr + 2 * A_stride]);
                break;
            case 3:
                jit->simd_broadcast_ss(vmmA(r), jit->ptr[A_ptr3]);
                break;
            case 4:
                jit->simd_broadcast_ss(vmmA(r), jit->ptr[A_ptr3 + A_stride]);
                break;
            case 5:
                jit->simd_broadcast_ss(vmmA(r), jit->ptr[A_ptr3 + 2 * A_stride]);
                break;
            default:
                throw std::runtime_error("number of reg-blocking rows is not supported");
            }
        };

        if (rows > 3) {
            jit->lea(A_ptr3, jit->ptr[A_ptr + 2 * A_stride]);
            jit->lea(A_ptr3, jit->ptr[A_ptr3 + A_stride]);
        }

        jit->align(64, false);
        jit->L(loop_over_k);
        if (is_preload_b) {
            // preload B regs
            for (int c = 0; c < cols; c++)
                jit->simd_loadu_ps(vmmB(c), jit->ptr[B_ptr + c * simd_width_bytes]);

            if (prefetch_B_adv > 0)
                jit->prefetcht0(jit->ptr[B_ptr + prefetch_B_adv]);

            jit->lea(B_ptr, jit->ptr[B_ptr + B_stride]);
            for (int r = 0; r < rows; r++) {
                loadA(r);
                for (int c = 0; c < cols; c++)
                    jit->simd_fmadd_ps(vmmC(r, c), vmmA(r), vmmB(c));
            }

            jit->lea(A_ptr, jit->ptr[A_ptr + 4]);
            if (rows > 3)
                jit->lea(A_ptr3, jit->ptr[A_ptr3 + 4]);
        } else {
            // preload A regs
            for (int r = 0; r < rows; r++)
                loadA(r);

            for (int c = 0; c < cols; c++) {
                jit->simd_loadu_ps(vmmB(c), jit->ptr[B_ptr + c * simd_width_bytes]);
                for (int r = 0; r < rows; r++)
                    jit->simd_fmadd_ps(vmmC(r, c), vmmA(r), vmmB(c));
            }

            jit->lea(B_ptr, jit->ptr[B_ptr + B_stride]);
            jit->lea(A_ptr, jit->ptr[A_ptr + 4]);
            if (rows > 3)
                jit->lea(A_ptr3, jit->ptr[A_ptr3 + 4]);
        }
        jit->dec(K);
        jit->jnz(loop_over_k, jit->T_NEAR);
    }

    // save C
    {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                jit->simd_storeu_ps(jit->ptr[dst_ptr + c * simd_width_bytes], vmmC(r, c));
            }
            jit->add(dst_ptr, dst_stride);
        }
    }
    jit->finalize();
    return jit;
}

static void gemm6x2_Mx2(const float * pA, int64_t A_stride,
                        const float * pB, int64_t B_stride,
                        const float * pC, int64_t C_stride,
                        int M, int64_t bK, int64_t is_accumulate_C) {
    static std::shared_ptr<JitKernel> gemm6x2[6] = {
        jit_compile_gemmRegBlk(6, 2),
        jit_compile_gemmRegBlk(1, 2),
        jit_compile_gemmRegBlk(2, 2),
        jit_compile_gemmRegBlk(3, 2),
        jit_compile_gemmRegBlk(4, 2),
        jit_compile_gemmRegBlk(5, 2),
    };
    int m;
    for (m = 0; m + 6 <= M; m += 6, pA += 6 * A_stride, pC += 6 * C_stride) {
        (*gemm6x2[0])(pA, A_stride, pB, B_stride, pC, C_stride, bK, is_accumulate_C);
    }
    if (m < M)
        (*gemm6x2[M-m])(pA, A_stride, pB, B_stride, pC, C_stride, bK, is_accumulate_C);
}

static std::shared_ptr<JitKernel> jit_compile_accumulate_wf16() {
    auto jit = std::make_shared<JitKernel>(__func__);
    auto simd_width = jit->vmm_width<float>();
    // load all arguments into register
    auto dst = jit->get_sreg(0, true);
    auto OC = jit->get_sreg(1, true);
    auto gate_ids = jit->get_sreg(2, true);
    auto gate_cnt = jit->get_sreg(3, true);
    auto pw0 = jit->get_sreg(4, true);
    auto dense_x = jit->get_sreg(5, true);
    auto IC = jit->get_sreg(6, true);

    auto g = jit->get_sreg(7);
    auto i = jit->get_sreg(8);
    auto p_w0 = jit->get_sreg(9);
    auto p_w1 = jit->get_sreg(10);
    auto p_w2 = jit->get_sreg(11);
    auto p_w3 = jit->get_sreg(12);

    jit->mov(p_w0, 0);
    jit->mov(p_w1, 0);
    jit->mov(p_w2, 0);
    jit->mov(p_w3, 0);
    jit->for_loop(g, gate_cnt, 4, [&](){
        jit->mov(p_w0.cvt32(), jit->dword[gate_ids + g*4 + 0*4]);
        jit->mov(p_w1.cvt32(), jit->dword[gate_ids + g*4 + 1*4]);
        jit->mov(p_w2.cvt32(), jit->dword[gate_ids + g*4 + 2*4]);
        jit->mov(p_w3.cvt32(), jit->dword[gate_ids + g*4 + 3*4]);
        jit->imul(p_w0, OC);
        jit->imul(p_w1, OC);
        jit->imul(p_w2, OC);
        jit->imul(p_w3, OC);
        jit->lea(p_w0, jit->ptr[pw0 + p_w0*2]);
        jit->lea(p_w1, jit->ptr[pw0 + p_w1*2]);
        jit->lea(p_w2, jit->ptr[pw0 + p_w2*2]);
        jit->lea(p_w3, jit->ptr[pw0 + p_w3*2]);

        auto vscale0 = jit->Vmm(0);
        auto vscale1 = jit->Vmm(1);
        auto vscale2 = jit->Vmm(2);
        auto vscale3 = jit->Vmm(3);
        jit->simd_broadcast_ss(vscale0, jit->ptr[dense_x + g*4 + 0*4]);
        jit->simd_broadcast_ss(vscale1, jit->ptr[dense_x + g*4 + 1*4]);
        jit->simd_broadcast_ss(vscale2, jit->ptr[dense_x + g*4 + 2*4]);
        jit->simd_broadcast_ss(vscale3, jit->ptr[dense_x + g*4 + 3*4]);
        jit->for_loop(i, OC, simd_width, [&](){
            auto vdst = jit->Vmm(4);
            auto vw0 = jit->Vmm(5);
            auto vw1 = jit->Vmm(6);
            auto vw2 = jit->Vmm(7);
            auto vw3 = jit->Vmm(8);
            jit->simd_loadu_ps(vdst, jit->ptr[dst + i*4]);
            jit->simd_loadu_phps(vw0, jit->ptr[p_w0 + i*2]);
            jit->simd_loadu_phps(vw1, jit->ptr[p_w1 + i*2]);
            jit->simd_loadu_phps(vw2, jit->ptr[p_w2 + i*2]);
            jit->simd_loadu_phps(vw3, jit->ptr[p_w3 + i*2]);
            jit->simd_fmadd_ps(vdst, vw0, vscale0);
            jit->simd_fmadd_ps(vdst, vw1, vscale1);
            jit->simd_fmadd_ps(vdst, vw2, vscale2);
            jit->simd_fmadd_ps(vdst, vw3, vscale3);
            jit->simd_storeu_ps(jit->ptr[dst + i*4], vdst);
        });
    });
    jit->finalize(i);
    return jit;
}

// static inline void reduce_outputs(float* dst0, float* src0, int num_copies, int N, int64_t OC) {
static std::shared_ptr<JitKernel> jit_compile_reduce_outputs() {
    auto jit = std::make_shared<JitKernel>(__func__);
    auto simd_width = jit->vmm_width<float>();
    // load all arguments into register
    auto dst0 = jit->get_sreg(0, true); // float*
    auto src0 = jit->get_sreg(1, true); // float*
    auto num_copies = jit->get_sreg(2, true); // int
    auto N = jit->get_sreg(3, true); // int
    auto OC = jit->get_sreg(4, true); // int

    auto n = jit->get_sreg(5);
    auto i = jit->get_sreg(6);
    auto k = jit->get_sreg(7);
    auto src_stride = jit->get_sreg(8);

    jit->mov(src_stride, N);
    jit->imul(src_stride, OC);

    auto ptemp = jit->get_sreg(8);
    jit->for_loop(n, N, 1, [&](){
        jit->for_loop(i, OC, simd_width, [&](){
            jit->lea(ptemp, jit->ptr[src0 + i*sizeof(float)]);
            auto vsum = jit->Vmm(0);
            auto vw = jit->Vmm(1);
            jit->simd_setzero_ps(vsum);
            jit->for_loop(k, num_copies, 1, [&](){
                jit->simd_loadu_ps(vw, jit->ptr[ptemp]);
                jit->vaddps(vsum, vsum, vw);
                jit->lea(ptemp, jit->ptr[ptemp + src_stride*sizeof(float)]);
            });
            jit->simd_storeu_ps(jit->ptr[dst0 + i*sizeof(float)], vsum);
        });
        jit->add(src0, OC);
        jit->add(dst0, OC);
    });

    jit->finalize();
    return jit;
}

/*
dst0 : [N, OC]
src0 : [num_copies, N, OC]
*/
static inline void reduce_outputs(float* dst0, float* src0, int num_copies, int N, int64_t OC) {
    static auto jit_reduce = jit_compile_reduce_outputs();
    int64_t simd_width = jit_reduce->vmm_width<float>();
    if (OC % simd_width) {
        throw std::runtime_error(std::string("OC is not multiple of ") + std::to_string(simd_width));
    }
    ov::parallel_nt(0, [&](const int ithr, const int nthr) {
        int64_t oc0, oc1;
        ov::splitter(OC/simd_width, nthr, ithr, oc0, oc1);
        oc0 *= simd_width;
        oc1 *= simd_width;
        if (oc1 > OC) oc1 = OC;

        auto* dst = dst0;
        auto* src = src0;

        (*jit_reduce)(dst0 + oc0, src0 + oc0, num_copies, N, oc1 - oc0);
    });
}

static std::shared_ptr<JitKernel> jit_compile_repack_2xsimdw(
            bool is_f16 = true,
            bool is_int8_peroc = false,
            bool with_zero_point = false) {
    auto jit = std::make_shared<JitKernel>(__func__);
    auto simd_width = jit->vmm_width<float>();
    // load all arguments into register
    auto src = jit->get_sreg(0, true); // pointer to ov::float16/u8/i8
    auto src_stride = jit->get_sreg(1, true);
    auto dst = jit->get_sreg(2, true); // float*
    auto bK = jit->get_sreg(3, true);
    auto scales = jit->get_sreg(4, is_int8_peroc); // scales
    auto zero_point = jit->get_sreg(5, with_zero_point); // zero-point

    auto k = jit->get_sreg(6);

    auto wf0 = jit->Vmm(0);
    auto wf1 = jit->Vmm(1);

    auto vzp0 = jit->Vmm(2);
    auto vzp1 = jit->Vmm(3);

    auto vscale0 = jit->Vmm(4);
    auto vscale1 = jit->Vmm(5);
    if (is_int8_peroc) {
        jit->simd_loadu_ps(vscale0, jit->ptr[scales + 0*simd_width*sizeof(float)]);
        jit->simd_loadu_ps(vscale1, jit->ptr[scales + 1*simd_width*sizeof(float)]);
        if (with_zero_point) {
            jit->simd_loadu_ps(vzp0, jit->ptr[zero_point + 0*simd_width*sizeof(float)]);
            jit->simd_loadu_ps(vzp1, jit->ptr[zero_point + 1*simd_width*sizeof(float)]);
        }
    }
    jit->for_loop(k, bK, 1, [&](){
        if (is_f16) {
            jit->simd_loadu_phps(wf0, jit->ptr[src + simd_width*0*sizeof(ov::float16)]);
            jit->simd_loadu_phps(wf1, jit->ptr[src + simd_width*1*sizeof(ov::float16)]);
        } else if (is_int8_peroc) {
            if (with_zero_point) {
                jit->simd_load_epu8_epi32(wf0, jit->ptr[src + simd_width*0*sizeof(uint8_t)]);
                jit->simd_load_epu8_epi32(wf1, jit->ptr[src + simd_width*1*sizeof(uint8_t)]);
            } else {
                jit->simd_load_epi8_epi32(wf0, jit->ptr[src + simd_width*0*sizeof(uint8_t)]);
                jit->simd_load_epi8_epi32(wf1, jit->ptr[src + simd_width*1*sizeof(uint8_t)]);
            }

            jit->simd_cvtepi32_ps(wf0, wf0);
            jit->simd_cvtepi32_ps(wf1, wf1);
            if (with_zero_point) {
                jit->simd_sub_ps(wf0, wf0, vzp0);
                jit->simd_sub_ps(wf1, wf1, vzp1);
            }
            jit->simd_mul_ps(wf0, wf0, vscale0);
            jit->simd_mul_ps(wf1, wf1, vscale1);
        }

        jit->prefetcht0(jit->ptr[src + 64]);
        jit->simd_storeu_ps(jit->ptr[dst + simd_width*0*sizeof(float)], wf0);
        jit->simd_storeu_ps(jit->ptr[dst + simd_width*1*sizeof(float)], wf1);
        jit->lea(dst, jit->ptr[dst + simd_width*2*sizeof(float)]);
        if (is_f16) {
            jit->lea(src, jit->ptr[src + src_stride*sizeof(ov::float16)]);
        } else if (is_int8_peroc) {
            jit->lea(src, jit->ptr[src + src_stride*sizeof(uint8_t)]);
        }
    });

    jit->finalize();
    return jit;
}

template <class T>
static T* scratch_alloc(size_t cnt) {
    thread_local uint8_t scratch[1024 * 1024 * 2] __attribute__((aligned(4096)));
    // assert(cnt * sizeof(T) < sizeof(scratch));
    // DEBUG_LOG(reinterpret_cast<void*>(scratch));
    return reinterpret_cast<T*>(scratch);
}

static void MM_ComputeBounded_reuseA_f16(const float* A,
                                  float* C,
                                  const ov::float16* W,
                                  int M,
                                  int IC,
                                  int OC,
                                  int n0,
                                  int n1) {
    static auto repack_2xsimdw = jit_compile_repack_2xsimdw();
    constexpr int BK = 54;
    float* scratch = scratch_alloc<float>(BK * (SIMDW*2) + OC);

    int K = IC;
    int64_t A_stride = IC;
    int64_t C_stride = OC;
    int64_t W_stride = OC;

    float* repacked_B = scratch;

    for (int k = 0; k < K; k += BK, A += BK, W += BK * W_stride) {
        int64_t bK = std::min(K - k, BK);
        int64_t is_accumulate_C = (k > 0);

        for (int n = n0; n + 2 * SIMDW <= n1; n += 2 * SIMDW) {
            // prepack [BK, 16] into scratch
            (*repack_2xsimdw)(W + n, W_stride, repacked_B, bK);
            gemm6x2_Mx2(A, A_stride, repacked_B, 2 * SIMDW, C + n, C_stride, M, bK, is_accumulate_C);
        }
    }
}

static std::shared_ptr<JitKernel> get_decompress_zp_u8() {
    auto jit = std::make_shared<JitKernel>(__func__);
    auto simd_width = jit->vmm_width<float>();

    auto zp_input_u8 = jit->get_sreg(0, true);
    auto zp_output_f32 = jit->get_sreg(1, true);
    auto cnt = jit->get_sreg(2, true);
    auto n = jit->get_sreg(3);

    auto vzpi32 = jit->Vmm(0);
    jit->for_loop(n, cnt, simd_width, [&]() {
        jit->simd_load_epu8_epi32(vzpi32, jit->ptr[zp_input_u8 + n*1]);
        jit->simd_cvtepi32_ps(vzpi32, vzpi32);
        jit->simd_storeu_ps(jit->ptr[zp_output_f32 + n*4], vzpi32);
    });
    // tails are converted using C instead.
    jit->finalize(n);
    return jit;
}

static void MM_ComputeBounded_reuseA_i8(
            const float * A,
            float * C,
            const uint8_t* W,
            const uint8_t* zp,
            const float* scales,
            int M, int IC, int OC,
            int64_t n0, int64_t n1) {
    static auto decompress_zp_u8 = get_decompress_zp_u8();
    static auto repack_2xsimdw_i8_zp = jit_compile_repack_2xsimdw(false, true, true);
    static auto repack_2xsimdw_i8_nozp = jit_compile_repack_2xsimdw(false, true, false);

    auto repack_2xsimdw_i8 = zp ? repack_2xsimdw_i8_zp : repack_2xsimdw_i8_nozp;
    constexpr int BK = 54;
    float* scratch = scratch_alloc<float>(BK * (SIMDW*2) + OC);

    int K = IC;
    auto A_stride = IC;
    auto C_stride = OC;
    auto W_stride = OC;

    float* repacked_B = scratch;
    float* zero_points = scratch + BK * (SIMDW*2);

    // deocompress zero-point into scratch
    if (zp) {
        int n = n0 + (*decompress_zp_u8)(zp + n0, zero_points, n1-n0);
        for (; n < n1; n ++) {
            zero_points[n - n0] = zp[n];
        }
    }

    for (int k = 0; k < K; k += BK, A += BK, W += BK * W_stride) {
        int bK = std::min(K - k, BK);
        auto is_accumulate_C = (k > 0);

        for (int n = n0; n + 2 * SIMDW <= n1; n += 2 * SIMDW) {
            // prepack [BK, 16] into scratch
            (*repack_2xsimdw_i8)(W + n, W_stride, repacked_B, bK, scales + n, zero_points + n - n0);
            gemm6x2_Mx2(A, A_stride, repacked_B, 2 * SIMDW, C + n, C_stride, M, bK, is_accumulate_C);
        }
    }
}

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

// x : [M, IC]
// W : [IC, OC]
//template <typename WType>
void dynPruneLinear_f16(const float* input,
                       float threshold,
                       float zero_point,
                       const ov::float16* W,
                       float* output,
                       int M,
                       int IC,
                       int OC) {
    if (M > 1) {
        parallel_nt(0, [&](const int ithr, const int nthr) {
            int n0, n1;
            splitter(OC / (2 * SIMDW), nthr, ithr, n0, n1);
            n0 *= 2 * SIMDW;
            n1 *= 2 * SIMDW;
            MM_ComputeBounded_reuseA_f16(input, output, W, M, IC, OC, n0, n1);
        });
        return;
    }
    static auto jit_accumulate_wf16 = jit_compile_accumulate_wf16();

    auto prof = PROFILE("gate_ids");
    static std::vector<int> gate_ids;
    static std::vector<float> gate_val;
    int gate_cnt = 0;
    gate_ids.resize(IC);
    gate_val.resize(IC);
    for (int channel = 0; channel < IC; channel++) {
        auto* src = input + channel;
        for (int m = 0; m < M; m++, src += IC) {
            auto& value = src[m];
            if (std::abs(value - zero_point) >= threshold) {
                gate_ids[gate_cnt] = channel;
                gate_val[gate_cnt] = value;
                gate_cnt++;
                break;
            }
        }
    }
    // pad to 4
    auto last_channel = gate_ids[gate_cnt - 1];
    while (gate_cnt & 3) {
        gate_ids[gate_cnt] = last_channel;
        gate_val[gate_cnt] = 0.0f;
        gate_cnt++;
    }

    // this mm kernel is the most time-consuming one
    auto nthr_max = parallel_get_max_threads();
    static std::vector<float> output_temp;
    output_temp.resize(nthr_max * M * OC);

    if (OC % SIMDW) {
        throw std::runtime_error(std::string("OC is not multiple of ") + std::to_string(SIMDW));
    }

    parallel_nt(0, [&](const int ithr, const int nthr) {
        int g0, g1;
        splitter(gate_cnt/4, nthr, ithr, g0, g1);
        g0 *= 4;
        g1 *= 4;
        auto* pdst = &output_temp[ithr * M * OC];
        memset(pdst, 0, M * OC * sizeof(output_temp[0]));
        (*jit_accumulate_wf16)(pdst, (OC), &gate_ids[g0], (g1 - g0), W, &gate_val[g0], (int64_t)(IC));
    });

    prof = PROFILE("reduce");
    reduce_outputs(output, output_temp.data(), nthr_max, M, OC);
}

template<bool with_zp>
void accumulate_w8_peroc(float* base_dst, int64_t OC,
                        int* ic_ids, int ic_cnt,
                        const uint8_t* Wu8,
                        const uint8_t* zp,
                        const float* scales,
                        float* dense_x, int64_t IC) {
    // decompress zero-point
    thread_local std::vector<float> zpbuff;
    zpbuff.resize(OC);
    auto* dst_zp = zpbuff.data();

    if (with_zp) {
        int oc = 0;
        for (; oc + SIMDW <= OC; oc += SIMDW) {
            auto zpu32 = simd_load_epu8_epi32(static_cast<void const*>(zp + oc));
            auto zpf32 = simd_cvtepi32_ps(zpu32);
            simd_storeu_ps(dst_zp + oc, zpf32);
        }
        for (; oc < OC; oc ++) {
            dst_zp[oc] = zp[oc];
        }
    }

    // vector x weights
    for (int g = 0; g < ic_cnt; g+=4) {
        auto ic0 = ic_ids[g];
        auto ic1 = ic_ids[g+1];
        auto ic2 = ic_ids[g+2];
        auto ic3 = ic_ids[g+3];

        const auto* p_w0 = Wu8 + ic0 * OC;
        const auto* p_w1 = Wu8 + ic1 * OC;
        const auto* p_w2 = Wu8 + ic2 * OC;
        const auto* p_w3 = Wu8 + ic3 * OC;

        int oc = 0;

        auto vx0 = simd_broadcast_ss(dense_x + g + 0);
        auto vx1 = simd_broadcast_ss(dense_x + g + 1);
        auto vx2 = simd_broadcast_ss(dense_x + g + 2);
        auto vx3 = simd_broadcast_ss(dense_x + g + 3);
        for (; oc + SIMDW <= OC; oc += SIMDW) {
            if (with_zp) {
                auto vscales = simd_loadu_ps(scales + oc);
                auto vzp = simd_loadu_ps(dst_zp + oc);
                auto vdst = simd_loadu_ps(base_dst + oc);

                auto wdw0 = simd_load_epu8_epi32(static_cast<void const*>(p_w0 + oc));
                auto wdw1 = simd_load_epu8_epi32(static_cast<void const*>(p_w1 + oc));
                auto wdw2 = simd_load_epu8_epi32(static_cast<void const*>(p_w2 + oc));
                auto wdw3 = simd_load_epu8_epi32(static_cast<void const*>(p_w3 + oc));

                auto vsum = simd_setzero_ps();

                vsum = simd_fmadd_ps(simd_sub_ps(simd_cvtepi32_ps(wdw0), vzp), vx0, vsum);
                vsum = simd_fmadd_ps(simd_sub_ps(simd_cvtepi32_ps(wdw1), vzp), vx1, vsum);
                vsum = simd_fmadd_ps(simd_sub_ps(simd_cvtepi32_ps(wdw2), vzp), vx2, vsum);
                vsum = simd_fmadd_ps(simd_sub_ps(simd_cvtepi32_ps(wdw3), vzp), vx3, vsum);

                vdst = simd_fmadd_ps(vsum, vscales, vdst);
                simd_storeu_ps(base_dst + oc, vdst);
            } else {
                auto vscales = simd_loadu_ps(scales + oc);
                auto vdst = simd_loadu_ps(base_dst + oc);

                auto wdw0 = simd_load_epi8_epi32(static_cast<void const*>(p_w0 + oc));
                auto wdw1 = simd_load_epi8_epi32(static_cast<void const*>(p_w1 + oc));
                auto wdw2 = simd_load_epi8_epi32(static_cast<void const*>(p_w2 + oc));
                auto wdw3 = simd_load_epi8_epi32(static_cast<void const*>(p_w3 + oc));

                auto vsum = simd_setzero_ps();

                vsum = simd_fmadd_ps(simd_cvtepi32_ps(wdw0), vx0, vsum);
                vsum = simd_fmadd_ps(simd_cvtepi32_ps(wdw1), vx1, vsum);
                vsum = simd_fmadd_ps(simd_cvtepi32_ps(wdw2), vx2, vsum);
                vsum = simd_fmadd_ps(simd_cvtepi32_ps(wdw3), vx3, vsum);

                vdst = simd_fmadd_ps(vsum, vscales, vdst);
                simd_storeu_ps(base_dst + oc, vdst);
            }
        }

        if (oc < OC) {
            auto x0 = dense_x[g + 0];
            auto x1 = dense_x[g + 1];
            auto x2 = dense_x[g + 2];
            auto x3 = dense_x[g + 3];
            for (; oc < OC; oc ++) {
                auto weight0 = p_w0[oc];
                auto weight1 = p_w1[oc];
                auto weight2 = p_w2[oc];
                auto weight3 = p_w3[oc];
                if (with_zp) {
                    weight0 -= dst_zp[oc];
                    weight1 -= dst_zp[oc];
                    weight2 -= dst_zp[oc];
                    weight3 -= dst_zp[oc];
                }
                weight0 *= scales[oc];
                weight1 *= scales[oc];
                weight2 *= scales[oc];
                weight3 *= scales[oc];
                base_dst[oc] += x0 * weight0 + x1 * weight1 + x2 * weight2 + x3 * weight3;
            }
        }
    }
}

template<bool with_zp>
void accumulate_w4(float* base_dst, int OC,
                   int* ic_ids, int ic_cnt,
                   const uint8_t* W,
                   const uint8_t* zp,
                   const float* scales,
                   float* dense_x, int IC, int IC_group_size) {
    // decompress zero-point
    thread_local std::vector<float> zpbuff;
    zpbuff.resize(OC);
    int last_gid = -1;
    // vector x weights
    for (int g = 0; g < ic_cnt; g+=4) {
        auto ic0 = ic_ids[g];
        auto ic1 = ic_ids[g+1];
        auto ic2 = ic_ids[g+2];
        auto ic3 = ic_ids[g+3];
        auto gid = ic0 / IC_group_size;
        auto* p_scales = scales + gid*OC;

        // entering a new group, decompress zero-points
        if (last_gid != gid) {
            if (with_zp) {
                auto* dst_zp = zpbuff.data();
                auto* src_zp = zp + gid * (OC/2);
                int oc = 0;
                auto vmask_u4 = simd_set1_epi32(0xF);
                for (; oc + SIMDW*2 <= OC; oc += SIMDW*2, src_zp += SIMDW) {
                    auto vzp16xu4_i32 = simd_load_epu8_epi32(static_cast<void const*>(src_zp));
                    auto vzp16xu4_i32_low = simd_and_si(vzp16xu4_i32, vmask_u4);
                    auto vzp16xu4_i32_high = simd_srli_epi32(vzp16xu4_i32, 4);
                    auto vzpf32_low = simd_cvtepi32_ps(vzp16xu4_i32_low);
                    auto vzpf32_high = simd_cvtepi32_ps(vzp16xu4_i32_high);
                    simd_storeu_ps(dst_zp + oc, vzpf32_low);
                    simd_storeu_ps(dst_zp + oc + SIMDW, vzpf32_high);
                }
                for (; oc < OC; oc +=2, src_zp++) {
                    dst_zp[oc] = src_zp[0] & 0xF;
                    dst_zp[oc + 1] = src_zp[0] >> 4;
                }
            }
            last_gid = gid;
        }

        const auto* p_w0 = W + ic0 * OC/2;
        const auto* p_w1 = W + ic1 * OC/2;
        const auto* p_w2 = W + ic2 * OC/2;
        const auto* p_w3 = W + ic3 * OC/2;
        auto* dst_zp = zpbuff.data();

        int oc = 0;

        auto vmask_u4 = simd_set1_epi32(0xF);
        auto vx0 = simd_broadcast_ss(dense_x + g + 0);
        auto vx1 = simd_broadcast_ss(dense_x + g + 1);
        auto vx2 = simd_broadcast_ss(dense_x + g + 2);
        auto vx3 = simd_broadcast_ss(dense_x + g + 3);
        if (with_zp) {
            for (; oc + SIMDW*2 <= OC; oc += SIMDW*2) {
                auto vzp0 = simd_loadu_ps(dst_zp + oc);
                auto vzp1 = simd_loadu_ps(dst_zp + oc + SIMDW);

                auto vdst0 = simd_loadu_ps(base_dst + oc);
                auto vdst1 = simd_loadu_ps(base_dst + oc + SIMDW);

                auto wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w0)); p_w0 += SIMDW;
                auto vsum0 = simd_setzero_ps();
                auto vsum1 = simd_setzero_ps();

                auto wdw0 = simd_cvtepi32_ps(simd_and_si(wdw_i32, vmask_u4));
                auto wdw1 = simd_cvtepi32_ps(simd_srli_epi32(wdw_i32, 4));
                vsum0 = simd_fmadd_ps(simd_sub_ps(wdw0, vzp0), vx0, vsum0);
                vsum1 = simd_fmadd_ps(simd_sub_ps(wdw1, vzp1), vx0, vsum1);

                wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w1)); p_w1 += SIMDW;
                wdw0 = simd_cvtepi32_ps(simd_and_si(wdw_i32, vmask_u4));
                wdw1 = simd_cvtepi32_ps(simd_srli_epi32(wdw_i32, 4));
                vsum0 = simd_fmadd_ps(simd_sub_ps(wdw0, vzp0), vx1, vsum0);
                vsum1 = simd_fmadd_ps(simd_sub_ps(wdw1, vzp1), vx1, vsum1);

                wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w2)); p_w2 += SIMDW;
                wdw0 = simd_cvtepi32_ps(simd_and_si(wdw_i32, vmask_u4));
                wdw1 = simd_cvtepi32_ps(simd_srli_epi32(wdw_i32, 4));
                vsum0 = simd_fmadd_ps(simd_sub_ps(wdw0, vzp0), vx2, vsum0);
                vsum1 = simd_fmadd_ps(simd_sub_ps(wdw1, vzp1), vx2, vsum1);

                wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w3)); p_w3 += SIMDW;
                wdw0 = simd_cvtepi32_ps(simd_and_si(wdw_i32, vmask_u4));
                wdw1 = simd_cvtepi32_ps(simd_srli_epi32(wdw_i32, 4));
                vsum0 = simd_fmadd_ps(simd_sub_ps(wdw0, vzp0), vx3, vsum0);
                vsum1 = simd_fmadd_ps(simd_sub_ps(wdw1, vzp1), vx3, vsum1);

                auto vscales0 = simd_loadu_ps(p_scales + oc);
                auto vscales1 = simd_loadu_ps(p_scales + oc + SIMDW);

                vdst0 = simd_fmadd_ps(vsum0, vscales0, vdst0);
                vdst1 = simd_fmadd_ps(vsum1, vscales1, vdst1);
                simd_storeu_ps(base_dst + oc, vdst0);
                simd_storeu_ps(base_dst + oc + SIMDW, vdst1);
            }
        } else {
            for (; oc + SIMDW*2 <= OC; oc += SIMDW*2) {
                auto vdst0 = simd_loadu_ps(base_dst + oc);
                auto vdst1 = simd_loadu_ps(base_dst + oc + SIMDW);

                auto wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w0)); p_w0 += SIMDW;
                auto vsum0 = simd_setzero_ps();
                auto vsum1 = simd_setzero_ps();

                auto wdw0 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-4), 32-4));
                auto wdw1 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-8), 32-4));
                vsum0 = simd_fmadd_ps(wdw0, vx0, vsum0);
                vsum1 = simd_fmadd_ps(wdw1, vx0, vsum1);

                wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w1)); p_w1 += SIMDW;

                wdw0 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-4), 32-4));
                wdw1 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-8), 32-4));
                vsum0 = simd_fmadd_ps(wdw0, vx1, vsum0);
                vsum1 = simd_fmadd_ps(wdw1, vx1, vsum1);

                wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w2)); p_w2 += SIMDW;
                wdw0 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-4), 32-4));
                wdw1 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-8), 32-4));
                vsum0 = simd_fmadd_ps(wdw0, vx2, vsum0);
                vsum1 = simd_fmadd_ps(wdw1, vx2, vsum1);

                wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w3)); p_w3 += SIMDW;
                wdw0 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-4), 32-4));
                wdw1 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-8), 32-4));
                vsum0 = simd_fmadd_ps(wdw0, vx3, vsum0);
                vsum1 = simd_fmadd_ps(wdw1, vx3, vsum1);

                auto vscales0 = simd_loadu_ps(p_scales + oc);
                auto vscales1 = simd_loadu_ps(p_scales + oc + SIMDW);

                vdst0 = simd_fmadd_ps(vsum0, vscales0, vdst0);
                vdst1 = simd_fmadd_ps(vsum1, vscales1, vdst1);
                simd_storeu_ps(base_dst + oc, vdst0);
                simd_storeu_ps(base_dst + oc + SIMDW, vdst1);
            }
        }

        if (oc < OC) {
            auto x0 = dense_x[g + 0];
            auto x1 = dense_x[g + 1];
            auto x2 = dense_x[g + 2];
            auto x3 = dense_x[g + 3];
            for (; oc < OC; oc += SIMDW*2, p_w0 += SIMDW, p_w1 += SIMDW, p_w2 += SIMDW, p_w3 += SIMDW) {
                for (int i = 0; i < SIMDW; i++) {
                    auto scale = p_scales[oc + i];
                    float weight0;
                    float weight1;
                    float weight2;
                    float weight3;
                    if (with_zp) {
                        auto zero_point = dst_zp[oc + i];
                        weight0 = (p_w0[i] & 0xF) - zero_point;
                        weight1 = (p_w1[i] & 0xF) - zero_point;
                        weight2 = (p_w2[i] & 0xF) - zero_point;
                        weight3 = (p_w3[i] & 0xF) - zero_point;
                    } else {
                        weight0 = (reinterpret_cast<const int8_t*>(p_w0)[i] << 4) >> 4;
                        weight1 = (reinterpret_cast<const int8_t*>(p_w1)[i] << 4) >> 4;
                        weight2 = (reinterpret_cast<const int8_t*>(p_w2)[i] << 4) >> 4;
                        weight3 = (reinterpret_cast<const int8_t*>(p_w3)[i] << 4) >> 4;
                    }
                    base_dst[oc + i] += (x0 * weight0 + x1 * weight1 + x2 * weight2 + x3 * weight3) * scale;
                }
                for (int i = 0; i < SIMDW; i++) {
                    auto scale = p_scales[oc + i + SIMDW];
                    float weight0;
                    float weight1;
                    float weight2;
                    float weight3;
                    if (with_zp) {
                        auto zero_point = dst_zp[oc + i + SIMDW];
                        weight0 = (p_w0[i] >> 4) - zero_point;
                        weight1 = (p_w1[i] >> 4) - zero_point;
                        weight2 = (p_w2[i] >> 4) - zero_point;
                        weight3 = (p_w3[i] >> 4) - zero_point;
                    } else {
                        weight0 = (reinterpret_cast<const int8_t*>(p_w0)[i] >> 4);
                        weight1 = (reinterpret_cast<const int8_t*>(p_w1)[i] >> 4);
                        weight2 = (reinterpret_cast<const int8_t*>(p_w2)[i] >> 4);
                        weight3 = (reinterpret_cast<const int8_t*>(p_w3)[i] >> 4);
                    }
                    base_dst[oc + i + SIMDW] += (x0 * weight0 + x1 * weight1 + x2 * weight2 + x3 * weight3) * scale;
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<bool with_zp>
void repack_weight_for_4x3(const uint8_t* W, int strideW, const float* scales, const float* zp, int K, int N, float* repacked_B_nx3, float* repacked_B_nx1) {
    //assert((N % 8) == 0);
#if 1
    for (int k = 0; k < K; k++) {
        int n0 = 0;
        auto* src = W + k*strideW;
        auto* dst = repacked_B_nx3 + k*SIMDW*3;
        auto dst_stride = K*SIMDW*3;
        for (n0 = 0; n0 + SIMDW*3 <= N; n0 += SIMDW*3, dst += dst_stride) {
            SIMD_F32 wf0;
            SIMD_F32 wf1;
            SIMD_F32 wf2;
            if (with_zp) {
                auto wi0 = simd_load_epu8_epi32(static_cast<void const*>(src + n0 + SIMDW * 0));
                auto wi1 = simd_load_epu8_epi32(static_cast<void const*>(src + n0 + SIMDW * 1));
                auto wi2 = simd_load_epu8_epi32(static_cast<void const*>(src + n0 + SIMDW * 2));
                auto zp0 = simd_loadu_ps(zp + n0 + SIMDW * 0);
                auto zp1 = simd_loadu_ps(zp + n0 + SIMDW * 1);
                auto zp2 = simd_loadu_ps(zp + n0 + SIMDW * 2);
                wf0 = simd_sub_ps(simd_cvtepi32_ps(wi0), (zp0));
                wf1 = simd_sub_ps(simd_cvtepi32_ps(wi1), (zp1));
                wf2 = simd_sub_ps(simd_cvtepi32_ps(wi2), (zp2));
            } else {
                auto wi0 = simd_load_epi8_epi32(static_cast<void const*>(src + n0 + SIMDW * 0));
                auto wi1 = simd_load_epi8_epi32(static_cast<void const*>(src + n0 + SIMDW * 1));
                auto wi2 = simd_load_epi8_epi32(static_cast<void const*>(src + n0 + SIMDW * 2));
                wf0 = simd_cvtepi32_ps(wi0);
                wf1 = simd_cvtepi32_ps(wi1);
                wf2 = simd_cvtepi32_ps(wi2);
            }
            wf0 = simd_mul_ps(wf0, simd_loadu_ps(scales + n0 + SIMDW*0));
            wf1 = simd_mul_ps(wf1, simd_loadu_ps(scales + n0 + SIMDW*1));
            wf2 = simd_mul_ps(wf2, simd_loadu_ps(scales + n0 + SIMDW*2));
            simd_storeu_ps(dst + SIMDW*0, wf0);
            simd_storeu_ps(dst + SIMDW*1, wf1);
            simd_storeu_ps(dst + SIMDW*2, wf2);
        }

        dst = repacked_B_nx1 + k*SIMDW;
        dst_stride = K*SIMDW;
        for (; n0 < N; n0 += SIMDW, dst += dst_stride) {
            for (int n = n0; n < n0+SIMDW; n++) {
                SIMD_F32 wf0;
                if (with_zp) {
                    auto wi0 = simd_load_epu8_epi32(static_cast<void const*>(src + n0 + SIMDW * 0));
                    auto zp0 = simd_loadu_ps(zp + n0 + SIMDW * 0);
                    wf0 = simd_sub_ps(simd_cvtepi32_ps(wi0), (zp0));
                } else {
                    auto wi0 = simd_load_epi8_epi32(static_cast<void const*>(src + n0 + SIMDW * 0));
                    wf0 = simd_cvtepi32_ps(wi0);
                }
                wf0 = simd_mul_ps(wf0, simd_loadu_ps(scales + n0 + SIMDW*0));
                simd_storeu_ps(dst + SIMDW*0, wf0);
            }
        }
    }
#else
    for (int k = 0; k < K; k++) {
        int n0 = 0;
        auto* src = W + k*strideW;
        auto* dst = repacked_B_nx3 + k*SIMDW*3;
        auto dst_stride = K*SIMDW*3;
        for (n0 = 0; n0 + SIMDW*3 <= N; n0 += SIMDW*3, dst += dst_stride) {
            for (int n = n0; n < n0+SIMDW*3; n++) {
                dst[n-n0] = (src[n] - zp[n]) * scales[n];
                //printf("%d,%d,%d  %d, %f, %f, =>  %f\n", k, n0, n, src[n], zp[n], scales[n], dst[n-n0]);
            }
        }
        dst = repacked_B_nx1 + k*SIMDW;
        dst_stride = K*SIMDW;
        for (; n0 < N; n0 += SIMDW, dst += dst_stride) {
            for (int n = n0; n < n0+SIMDW; n++) {
                dst[n-n0] = (src[n] - zp[n]) * scales[n];
            }
        }
    }
#endif
}

void MM_ComputeBounded_reuseB_i8(const float * A,
                                 float * C,
                                 const uint8_t* W,
                                 const uint8_t* zp,
                                 const float* scales,
                                 int M, int IC, int OC,
                                 int n0, int n1) {
    static std::shared_ptr<JitKernel> gemm4x3[6] = {
        jit_compile_gemmRegBlk(4, 3),
        jit_compile_gemmRegBlk(1, 3),
        jit_compile_gemmRegBlk(2, 3),
        jit_compile_gemmRegBlk(3, 3),
    };
    static std::shared_ptr<JitKernel> gemm4x1[6] = {
        jit_compile_gemmRegBlk(4, 1),
        jit_compile_gemmRegBlk(1, 1),
        jit_compile_gemmRegBlk(2, 1),
        jit_compile_gemmRegBlk(3, 1),
    };

    constexpr int BK = 512;
    constexpr int BN = 512;
    auto bN_SIMDWx3 = BN / (SIMDW*3) * (SIMDW*3);
    auto bN_SIMDWx1 = BN - bN_SIMDWx3;
    float* scratch = scratch_alloc<float>(BN * BK + BN);
    float* repacked_B_n24 = scratch;
    float* repacked_B_n8 = repacked_B_n24 + bN_SIMDWx3 * BK;
    float* zero_points = repacked_B_n8 + SIMDW*3 * BK;

    const int64_t A_stride = IC;
    const int64_t B_stride = OC;
    const int64_t C_stride = OC;

    for (int cur_n = n0; cur_n < n1; cur_n += BN) {
        int bN = std::min(n1 - cur_n, BN);
        const auto* pW = W + cur_n;

        // decompress zero-point
        if (zp) {
            for (int n = 0; n < bN; n += SIMDW) {
                auto zp0 = simd_load_epu8_epi32(static_cast<void const*>(zp + cur_n + n));
                auto zpf32 = simd_cvtepi32_ps(zp0);
                simd_storeu_ps(zero_points + n, zpf32);
            }
        }

        for (int k0 = 0; k0 < IC; k0 += BK, pW += BK * B_stride) {
            int64_t bK = std::min(IC - k0, BK);
            if (zp) {
                repack_weight_for_4x3<true>(pW, B_stride,
                                    scales + cur_n,
                                    zero_points,
                                    bK, bN,
                                    repacked_B_n24,
                                    repacked_B_n8);
            } else {
                repack_weight_for_4x3<false>(pW, B_stride,
                                    scales + cur_n,
                                    zero_points,
                                    bK, bN,
                                    repacked_B_n24,
                                    repacked_B_n8);
            }

            bool is_accumulate_C = (k0 > 0);
            auto* pC = C + cur_n;
            int m;
            // re-use repacked B sub-matrix in L2 cache as long as we can.
            const auto* pA = A + k0;
            for (m = 0; m + 4 <= M; m += 4, pA += 4 * A_stride, pC += 4 * C_stride) {
                auto* pB = repacked_B_n24;
                int n = 0;
                for (; n + SIMDW * 3 <= bN; n += SIMDW * 3, pB += SIMDW * 3 * bK)
                    (*gemm4x3[0])(pA, A_stride, pB, SIMDW*3, pC + n, C_stride, bK, is_accumulate_C);
                pB = repacked_B_n8;
                for (; n < bN; n += SIMDW, pB += SIMDW * bK)
                    (*gemm4x1[0])(pA, A_stride, pB, SIMDW, pC + n, C_stride, bK, is_accumulate_C);
            }
            // M tails
            if (m < M) {
                auto* pB = repacked_B_n24;
                int n = 0;
                for (; n + SIMDW * 3 <= bN; n += SIMDW * 3, pB += SIMDW * 3 * bK)
                    (*gemm4x3[M - m])(pA, A_stride, pB, SIMDW*3, pC + n, C_stride, bK, is_accumulate_C);
                pB = repacked_B_n8;
                for (; n < bN; n += SIMDW, pB += SIMDW * bK)
                    (*gemm4x1[M - m])(pA, A_stride, pB, SIMDW, pC + n, C_stride, bK, is_accumulate_C);
            }
        }
    }
}

void MM_ComputeBounded_reuseA_i4(
            const float * A,
            float * C,
            const uint8_t* W,
            const uint8_t* zp,
            const float* scales,
            int M, int IC, int OC,
            int n0, int n1, int icgs) {
    int BK = icgs;
    float* scratch = scratch_alloc<float>(BK * (SIMDW*2) + OC);

    int K = IC;
    auto A_stride = IC;
    auto C_stride = OC;
    auto W_stride = (OC/2);

    float* repacked_B = scratch;
    float* zero_points = scratch + BK*(SIMDW*2);
    auto Z_stride = zp ? W_stride : 0;

    for (int k = 0; k < K; k += BK, A += BK, W += BK * W_stride, zp += Z_stride, scales += OC) {
        int bK = std::min(K - k, BK);
        auto is_accumulate_C = (k > 0);

        // deocompress zero-point into scratch buffer
        if (zp) {
            const auto* pzp = zp + n0/2;
            int n = 0;
            auto vmask_u4 = simd_set1_epi32(0xF);
            for (n = n0; n + SIMDW*2 <= n1; n += SIMDW*2, pzp += SIMDW) {
                auto vzp16xu4_i32 = simd_load_epu8_epi32(static_cast<void const*>(pzp));
                // 8 x low-4bits  : 0,1,2,3,4,5,6,7
                // 8 x high-4bits : 8,9,a,b,c,d,e,f

                auto vzp16xu4_i32_low = simd_and_si(vzp16xu4_i32, vmask_u4);
                auto vzp16xu4_i32_high = simd_srli_epi32(vzp16xu4_i32, 4);

                auto vzpf32_low = simd_cvtepi32_ps(vzp16xu4_i32_low);
                auto vzpf32_high = simd_cvtepi32_ps(vzp16xu4_i32_high);
                simd_storeu_ps(zero_points + n - n0, vzpf32_low);
                simd_storeu_ps(zero_points + n - n0 + SIMDW, vzpf32_high);
            }
            for (; n < n1; n += 2, pzp++) {
                zero_points[n - n0] = (*pzp) & 0xF;
                zero_points[n - n0 + 1] = (*pzp) >> 4;
            }
        }

        for (int n = n0; n + 2 * SIMDW <= n1; n += 2 * SIMDW) {
            // prepack subB [BK, 16] into scratch
            // because BK is fully contained within IC-group (BK == n*IC_group_size), it can share same zp & scales
            if (zp) {
                auto* dst = repacked_B;
                auto vzp0 = simd_loadu_ps(zero_points + (n - n0));
                auto vzp1 = simd_loadu_ps(zero_points + (n - n0) + SIMDW);
                auto vscale0 = simd_loadu_ps(scales + n);
                auto vscale1 = simd_loadu_ps(scales + n + SIMDW);
                const auto* srcW = W + n/2;
                auto vmask_u4 = simd_set1_epi32(0xF);
                for (int k = 0; k < bK; k++, dst += 2 * SIMDW, srcW += W_stride) {
                    // 16 x i4
                    auto wdw = simd_load_epu8_epi32(static_cast<void const*>(srcW + SIMDW * 0));

                    auto wdw_low = simd_and_si(wdw, vmask_u4);
                    auto wdw_high = simd_srli_epi32(wdw, 4);

                    auto wf0 = simd_sub_ps(simd_cvtepi32_ps(wdw_low), vzp0);
                    auto wf1 = simd_sub_ps(simd_cvtepi32_ps(wdw_high), vzp1);
                    wf0 = simd_mul_ps(wf0, vscale0);
                    wf1 = simd_mul_ps(wf1, vscale1);

                    // prefetch right
                    simd_prefetch(srcW + 64, _MM_HINT_T0);

                    simd_storeu_ps(dst + SIMDW * 0, wf0);
                    simd_storeu_ps(dst + SIMDW * 1, wf1);
                }
            } else {
                auto* dst = repacked_B;
                auto vscale0 = simd_loadu_ps(scales + n);
                auto vscale1 = simd_loadu_ps(scales + n + SIMDW);
                const auto* srcW = W + n/2;
                for (int k = 0; k < bK; k++, dst += 2 * SIMDW, srcW += W_stride) {
                    // 16 x i4
                    auto wdw = simd_load_epu8_epi32(static_cast<void const*>(srcW + SIMDW * 0));

                    auto wdw_low = simd_srai_epi32(simd_slli_epi32(wdw, 32-4), 32-4);
                    auto wdw_high = simd_srai_epi32(simd_slli_epi32(wdw, 32-8), 32-4);

                    auto wf0 = simd_cvtepi32_ps(wdw_low);
                    auto wf1 = simd_cvtepi32_ps(wdw_high);
                    wf0 = simd_mul_ps(wf0, vscale0);
                    wf1 = simd_mul_ps(wf1, vscale1);

                    // prefetch right
                    simd_prefetch(srcW + 64, _MM_HINT_T0);

                    simd_storeu_ps(dst + SIMDW * 0, wf0);
                    simd_storeu_ps(dst + SIMDW * 1, wf1);
                }
            }

            gemm6x2_Mx2(A, A_stride, repacked_B, 2 * SIMDW, C + n, C_stride, M, bK, is_accumulate_C);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void dynPruneLinear_i8(const float* input,      // [M, IC]
                        float threshold,
                        float zero_point,
                        const uint8_t* W,       // [IC, OC]
                        const uint8_t* zp,      // [OC]
                        const float* scales,    // [OC]
                        float* output,          // [M, OC]
                        int M, int IC, int OC) {
    if (M > 1) {
        if (M < 32) {
            parallel_nt(0, [&](const int ithr, const int nthr) {
                int n0, n1;
                splitter(OC/(2*SIMDW), nthr, ithr, n0, n1);
                n0 *= 2*SIMDW;
                n1 *= 2*SIMDW;
                MM_ComputeBounded_reuseA_i8(
                    input, output,
                    W, zp, scales, M, IC, OC, n0, n1);
            });
        } else {
            parallel_nt(0, [&](const int ithr, const int nthr) {
                int n0, n1;
                splitter(OC/(SIMDW), nthr, ithr, n0, n1);
                n0 *= SIMDW;
                n1 *= SIMDW;
                MM_ComputeBounded_reuseB_i8(
                    input, output,
                    W, zp, scales, M, IC, OC, n0, n1);
            });
        }
        return;
    }

    auto prof = PROFILE("gate_ids");
    static std::vector<int> gate_ids;
    static std::vector<float> gate_val;
    int gate_cnt = 0;
    gate_ids.resize(IC);
    gate_val.resize(IC);
    for (int channel = 0; channel < IC; channel++) {
        auto* src = input + channel;
        for (int m = 0; m < M; m++, src += IC) {
            auto& value = src[m];
            if (std::abs(value - zero_point) >= threshold) {
                gate_ids[gate_cnt] = channel;
                gate_val[gate_cnt] = value;
                gate_cnt++;
                break;
            }
        }
    }

    // pad to 4
    auto last_channel = gate_ids[gate_cnt - 1];
    while (gate_cnt & 3) {
        gate_ids[gate_cnt] = last_channel;
        gate_val[gate_cnt] = 0.0f;
        gate_cnt++;
    }

    // std::cout << M << "," << IC << "," << OC << "," << threshold << "," << zero_point << std::endl;
    prof = PROFILE("mm");

    // this mm kernel is the most time-consuming one
    auto nthr_max = parallel_get_max_threads();
    static std::vector<float> output_temp;
    output_temp.resize(nthr_max * M * OC);

    parallel_nt(0, [&](const int ithr, const int nthr) {
        int g0, g1;
        splitter(gate_cnt/4, nthr, ithr, g0, g1);
        g0 *= 4;
        g1 *= 4;
        auto* pdst = &output_temp[ithr * M * OC];
        memset(pdst, 0, M * OC * sizeof(output_temp[0]));
        if (zp)
            accumulate_w8_peroc<true>(pdst, OC, &gate_ids[g0], g1 - g0, W, zp, scales, &gate_val[g0], IC);
        else
            accumulate_w8_peroc<false>(pdst, OC, &gate_ids[g0], g1 - g0, W, zp, scales, &gate_val[g0], IC);
    });

    prof = PROFILE("reduce");
    reduce_outputs(output, output_temp.data(), nthr_max, M, OC);
    return;
}

void dynPruneLinear_i4(const float* input,      // [M, IC]
                        float threshold,
                        float zero_point,
                        const uint8_t* W,       // [IC, OC]
                        const uint8_t* zp,      // [OC]
                        const float* scales,    // [OC]
                        float* output,          // [M, OC]
                        int M, int IC, int OC,
                        int IC_group_size) {
    if ((OC % (2*SIMDW)) > 0) {
        throw std::runtime_error("OC is not multiple of 16");
    }

    if (M > 1) {
        // a reference impl
        parallel_nt(0, [&](const int ithr, const int nthr) {
            int n0, n1;
            splitter(OC/(2*SIMDW), nthr, ithr, n0, n1);
            n0 *= 2*SIMDW;
            n1 *= 2*SIMDW;
            MM_ComputeBounded_reuseA_i4(
                input, output,
                W, zp, scales, M, IC, OC, n0, n1, IC_group_size);
        });
        return;
    }

    auto prof = PROFILE("gate_ids");
    static std::vector<int> gate_ids;
    static std::vector<float> gate_val;
    int gate_cnt = 0;
    gate_ids.resize(IC);
    gate_val.resize(IC);
    for (int c0 = 0; c0 < IC; c0 += IC_group_size) {
        for (int c1 = 0; c1 < IC_group_size; c1++) {
            auto channel = c0 + c1;
            auto& value = input[channel];
            if (std::abs(value - zero_point) >= threshold) {
                gate_ids[gate_cnt] = channel;
                gate_val[gate_cnt] = value;
                gate_cnt++;
            }
        }
        if (gate_cnt & 3) {
            // padding : ensuer 4 rows are from same group
            auto n_pad = 4 - (gate_cnt & 3);
            auto ic_pad = gate_ids[gate_cnt-1];
            for (int i = 0; i < n_pad; i++) {
                gate_ids[gate_cnt] = ic_pad;
                gate_val[gate_cnt] = 0.0f;
                gate_cnt++;
            }
        }
    }

    // std::cout << M << "," << IC << "," << OC << "," << threshold << "," << zero_point << std::endl;
    prof = PROFILE("mm");

    // this mm kernel is the most time-consuming one
    auto nthr_max = parallel_get_max_threads();
    static std::vector<float> output_temp;
    output_temp.resize(nthr_max * M * OC);

    parallel_nt(0, [&](const int ithr, const int nthr) {
        int g0, g1;
        splitter(gate_cnt/4, nthr, ithr, g0, g1);
        g0 *= 4;
        g1 *= 4;
        auto* pdst = &output_temp[ithr * M * OC];
        memset(pdst, 0, M * OC * sizeof(output_temp[0]));
        if (zp)
            accumulate_w4<true>(pdst, OC, &gate_ids[g0], g1 - g0, W, zp, scales, &gate_val[g0], IC, IC_group_size);
        else
            accumulate_w4<false>(pdst, OC, &gate_ids[g0], g1 - g0, W, zp, scales, &gate_val[g0], IC, IC_group_size);
    });

    prof = PROFILE("reduce");
    reduce_outputs(output, output_temp.data(), nthr_max, M, OC);
    return;
}

// [OC, IC/2, 2] => [IC, OC/2, 2]
// each row is further reordered in unit of 16 x i4 in [0,8,1,9,2,a,3,b,4,c,5,d,6,e,7,f] order
void dynPruneLinear_repack_i4(uint8_t * src, uint8_t * dst, int IC, int OC) {
    auto src_stride = IC / 2;

    int ic = 0;
    uint8_t scratch0[64];
    uint8_t scratch1[64];
    for (; ic + 2*SIMDW*4 <= IC; ic += 2*SIMDW*4) {
        // 64-ic
        auto* pdst = dst + ic * (OC / 2);
        auto vmask_low_u4 = simd_set1_epi8(0xF);
        auto vmask_high_u4 = simd_set1_epi8(0xF0);
        for (int oc = 0; oc < OC; oc += (SIMDW*2), pdst += SIMDW) {
            // 64-ic x 16-oc
            auto* psrc_oc0 = src + (ic / 2) + (oc + 0)*src_stride;
            auto* psrc_oc1 = src + (ic / 2) + (oc + SIMDW)*src_stride;
            for (int k = 0; k < SIMDW; k++, psrc_oc0 += src_stride, psrc_oc1 += src_stride) {
                auto b0 = simd_loadu_i32(psrc_oc0); // oc+0: ic0~64
                auto b1 = simd_loadu_i32(psrc_oc1); // oc+8: ic0~64
                auto b0_ic0 = simd_and_si(b0, vmask_low_u4);
                auto b0_ic1 = simd_and_si(simd_srli_epi16(b0, 4), vmask_low_u4);

                auto b1_ic0 = simd_and_si(simd_slli_epi16(b1, 4), vmask_high_u4);
                auto b1_ic1 = simd_and_si(b1, vmask_high_u4);

                auto bdst_ic0 = simd_or_si(b1_ic0, b0_ic0);    // even channels
                auto bdst_ic1 = simd_or_si(b1_ic1, b0_ic1);    // odd channels

                simd_storeu_si(scratch0, bdst_ic0);
                simd_storeu_si(scratch1, bdst_ic1);

                auto* pdst_temp0 = pdst + k;
                auto* pdst_temp1 = pdst + k + (OC / 2);
                for (int i = 0; i < SIMDW * 4; i++, pdst_temp0 += OC, pdst_temp1 += OC) {
                    *pdst_temp0 = scratch0[i];
                    *pdst_temp1 = scratch1[i];
                }
            }
        }
    }

    // [OC, IC/2, 2] => [IC, OC/2, 2]
    // tails
    for (; ic < IC; ic += 2) {
        auto* pdst_a = dst + ic * (OC / 2);
        auto* pdst_b = pdst_a + (OC / 2);
        for (int oc = 0; oc < OC; oc += SIMDW*2, pdst_a += SIMDW, pdst_b += SIMDW) {
            // interleave
            auto* psrc_oc0 = src + (ic / 2) + (oc + 0)*src_stride;
            auto* psrc_oc1 = src + (ic / 2) + (oc + SIMDW)*src_stride;
            for (int k = 0; k < SIMDW; k++, psrc_oc0 += src_stride, psrc_oc1 += src_stride) {
                auto data0 = *psrc_oc0;  // [ic1, ic0] packed in same u8
                auto u40a = (data0 & 0xF);
                auto u40b = (data0 >> 4);
                auto data1 = *psrc_oc1;
                auto u41a = (data1 & 0xF);
                auto u41b = (data1 >> 4);
                pdst_a[k] = (u41a << 4) | u40a;
                pdst_b[k] = (u41b << 4) | u40b;
            }
        }
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov