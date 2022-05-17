/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cassert>

#include "common/math_utils.hpp"
#include "common/utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/gemm/f32/jit_sse41_gemv_n_f32_kern.hpp"

#ifdef _WIN32
static const bool is_windows = true;
#else
static const bool is_windows = false;
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

// Convert between vector register lengths.
static inline Xmm make_xmm(const Xmm &v) {
    return Xmm(v.getIdx());
}

static inline Ymm make_ymm(const Xmm &v) {
    return Ymm(v.getIdx());
}

static inline int log2_of_pow2(int n) {
    assert(n > 0);
    int e = 0;
    while (n >>= 1)
        e++;

    return e;
}

// Load vector register data for x, y or A.
void jit_sse41_gemv_n_f32_kern::v_load(
        const Xmm &dst, const Address &src, int nelems) {
    if (nelems >= v_nelems_) {
        uni_vmovups(dst, src);
        return;
    }

    switch (nelems) {
        case 1: uni_vmovss(make_xmm(dst), src); break;
        case 2: uni_vmovlps(make_xmm(dst), src); break;
        case 4: uni_vmovups(make_xmm(dst), src); break;
        case 8: uni_vmovups(make_ymm(dst), src); break;
        default:
            if (nelems > 8)
                uni_vmovups(dst | k1 | T_z, src);
            else if (nelems > 4)
                uni_vmovups(make_ymm(dst) | k1 | T_z, src);
            else
                uni_vmovups(make_xmm(dst) | k1 | T_z, src);
            break;
    }
}

// Store vector register data for x, y or A.
void jit_sse41_gemv_n_f32_kern::v_store(
        const Address &dst, const Xmm &src, int nelems) {
    if (nelems >= v_nelems_) {
        uni_vmovups(dst, src);
        return;
    }

    switch (nelems) {
        case 1: uni_vmovss(dst, make_xmm(src)); break;
        case 2: uni_vmovsd(dst, make_xmm(src)); break;
        case 4: uni_vmovups(dst, make_xmm(src)); break;
        case 8: uni_vmovups(dst, make_ymm(src)); break;
        default:
            if (nelems > 8)
                uni_vmovups(dst, src | k1);
            else if (nelems > 4)
                uni_vmovups(dst, make_ymm(src) | k1);
            else
                uni_vmovups(dst, make_xmm(src) | k1);
            break;
    }
}

// Perform Hadamard product of 2 vectors and accumulate.
// Use FMA instruction, otherwise emulate.
void jit_sse41_gemv_n_f32_kern::dot_product(
        const Xmm &dst, const Xmm &src1, const Xmm &src2) {
    if (has_avx2_)
        vfmadd231ps(dst, src1, src2);
    else if (has_avx_) {
        vmulps(scratch_, src1, src2);
        vaddps(dst, dst, scratch_);
    } else {
        mulps(src2, src1);
        addps(dst, src2);
    }
}

void jit_sse41_gemv_n_f32_kern::kernel_loop(
        int unroll_m, int unroll_n, bool fetch, bool last) {
    int um_vecs = utils::div_up(unroll_m, v_nelems_);

    for (int i = 0; i < unroll_n; i++) {
        int mult = i % 4;
        auto A = i < 4 ? A1_ : A2_;
        for (int j = 0; j < um_vecs; j++) {
            decltype(LDA_ * mult) lda_mult = mult == 3 ? LDA3_ : LDA_ * mult;
            // Load A.
            if (fetch && (j * v_nelems_ * size_ % 64 == 0))
                prefetch_a(ptr[A + lda_mult
                        + size_
                                * (prefetch_size_a_ + j * v_nelems_
                                        - offset_a_)]);
            v_load(a_, ptr[A + lda_mult + size_ * (j * v_nelems_ - offset_a_)],
                    unroll_m);

            // Load y if needed.
            if (i == 0) {
                if (fetch && (j * v_nelems_ * size_ % 64 == 0))
                    prefetch_y(ptr[Y1_
                            + size_
                                    * (prefetch_size_y_ + j * v_nelems_
                                            - offset_y_)]);
                v_load(y_[j], ptr[Y1_ + size_ * (j * v_nelems_ - offset_y_)],
                        unroll_m);
            }

            dot_product(y_[j], x_[i], a_);
        }
    }

    // Store y.
    for (int j = 0; j < um_vecs; j++) {
        auto y = y_[j];
        v_store(ptr[Y1_ + size_ * (j * v_nelems_ - offset_y_)], y, unroll_m);
        uni_vxorps(y, y, y);
    }

    if (!last) {
        add(A1_, unroll_m * size_);
        if (unroll_n > 4) add(A2_, unroll_m * size_);
        add(Y1_, unroll_m * size_);
    }
}

// Inner loop for A non-transposed.
void jit_sse41_gemv_n_f32_kern::innerloop(int unroll_m, int unroll_n) {
    mov(Y1_, Y_);

    // Load x and scale by alpha.
    prefetch_x(ptr[X_ + size_ * (prefetch_size_x_ - offset_x_)]);
    for (int i = 0; i < unroll_n; i++) {
        auto x = x_[i];
        uni_vbroadcastss(x, ptr[X_ + size_ * (0 - offset_x_)]);
        uni_vmulps(x, x, alpha_);
        add(X_, INCX_);
    }

    int um_vecs = utils::div_up(unroll_m, v_nelems_);
    for (int i = 0; i < um_vecs; i++) {
        auto y = y_[i];
        uni_vxorps(y, y, y);
    }

    const int num_rem_labels = log2_of_pow2(unroll_m);
    std::vector<Label> label_m_remainder(num_rem_labels);
    mov(I_, M_);
    sar(I_, num_rem_labels);
    jle(label_m_remainder[0], T_NEAR);

    Label label_m_loop;
    L_aligned(label_m_loop);
    {
        kernel_loop(unroll_m, unroll_n, true, false);

        dec(I_);
        jg(label_m_loop, T_NEAR);
    }

    Label label_m_loop_end;
    int label_idx = 0;
    const bool use_mask = has_avx512_;
    int min_unroll_m = use_mask ? v_nelems_ : 1;
    for (int um = (unroll_m >> 1); um >= min_unroll_m; um >>= 1) {
        L_aligned(label_m_remainder[label_idx++]);
        mov(I_, M_);
        test(I_, um);
        if (um > 1)
            jle(label_m_remainder[label_idx], T_NEAR);
        else
            jle(label_m_loop_end, T_NEAR);

        kernel_loop(um, unroll_n, false, um == 1);
    }

    if (use_mask) {
        L_aligned(label_m_remainder[label_idx]);
        mov(I_, M_);
        and_(I_, v_nelems_ - 1);
        jle(label_m_loop_end, T_NEAR);

        // Prepare mask.
        mov(rbx, rcx);
        mov(rcx, I_);
        mov(rax, -1);
        shl(rax, cl);
        kmovq(k1, rax);
        knotq(k1, k1);
        mov(rcx, rbx);

        kernel_loop(v_nelems_ - 1, unroll_n, false, true);
    }
    L_aligned(label_m_loop_end);
}

void jit_sse41_gemv_n_f32_kern::outerloop(int unroll_x, int unroll_y,
        Label *&cur_outerloop_label, Label *&outerloop_end_label) {
    bool is_tail = unroll_y < unroll_n_;

    if (is_tail) {
        L_aligned(*cur_outerloop_label);
        cur_outerloop_label++;
    }
    cmp(N_, unroll_y);
    jl(*cur_outerloop_label, T_NEAR); // Jump to next outerloop label.

    Label label_n_loop;
    L_aligned(label_n_loop);
    {
        // Note: This restrict the max n-unroll to be 1, 2, 4, or 8.
        // Different approach is need for larger unrolls.  One could use extra
        // general purpose registers to track further columns of A.
        mov(A1_, A_);
        if (unroll_y > 4) lea(A2_, ptr[A1_ + LDA_ * 4]);
        if (!is_tail) lea(A_, ptr[A_ + LDA_ * unroll_y]);

        innerloop(unroll_x, unroll_y);

        if (!is_tail) {
            sub(N_, unroll_y);
            cmp(N_, unroll_y);
            jge(label_n_loop, T_NEAR);
        } else if (unroll_y > 1) {
            jmp(*outerloop_end_label, T_NEAR);
        }
    }
}

void jit_sse41_gemv_n_f32_kern::generate() {
    // Prologue
    preamble();

    if (is_windows) {
        mov(LDA_, arg_lda_);
        mov(X_, arg_x_);
    }

    mov(INCX_, arg_incx_);
    mov(Y_, arg_y_);
    // incy is assumed 1 for non-transpose A.

    uni_vbroadcastss(alpha_, qword[ALPHA_]);

    mov(M_, qword[M_]);
    mov(N_, qword[N_]);
    mov(LDA_, qword[LDA_]);
    mov(INCX_, qword[INCX_]);

    sub(A_, -offset_a_ * size_);
    sub(X_, -offset_x_ * size_);
    sub(Y_, -offset_y_ * size_);

    lea(LDA_, ptr[LDA_ * size_]);
    lea(INCX_, ptr[INCX_ * size_]);
    lea(LDA3_, ptr[LDA_ + LDA_ * 2]);

    std::vector<Label> outerloop_labels(max_unroll_n_);
    Label *cur_outerloop_label = &outerloop_labels[0];
    Label *outerloop_end_label = &outerloop_labels[unroll_n_ - 1];

    // Main n loop + n remainders.
    for (int un = unroll_n_; un > 0; un--)
        outerloop(unroll_m_, un, cur_outerloop_label, outerloop_end_label);

    L_aligned(*outerloop_end_label);

    // Epilogue.
    postamble();
}

// Function signature: gemv(*m, *n, *alpha, *a, *lda, *x, *incx, *y, *incy)
jit_sse41_gemv_n_f32_kern::jit_sse41_gemv_n_f32_kern(void)
    : jit_generator(nullptr, 100000)
    , arg_lda_(0)
    , arg_x_(0)
    , arg_incx_(0)
    , arg_y_(0)
    , arg_incy_(0) {

    has_avx512_ = mayiuse(avx512_core);
    has_avx2_ = mayiuse(avx2);
    has_avx_ = mayiuse(avx);
    has_sse41_ = mayiuse(sse41);

    int unroll_m = 0;
    int unroll_n = 0;
    int v_type = -1;
    if (has_avx512_) {
        unroll_m = 16 * (512 / 32);
        unroll_n = 8;
        v_type = 2;
        fetch_ = true;
    } else if (has_avx_) {
        unroll_m = 4 * (256 / 32);
        unroll_n = 8;
        v_type = 1;
        fetch_ = false;
    } else {
        unroll_m = 4 * (128 / 32);
        unroll_n = 8;
        v_type = 0;
        fetch_ = false;
    }

    // Assign integer registers
    M_ = abi_param1; // is_windows ? rcx : rdx
    N_ = abi_param2; // is_windows ? rdx : rsi
    ALPHA_ = abi_param3; // is_windows ? r8 : rdx
    A_ = abi_param4; // is_windows ? r9 : rcx
    LDA_ = is_windows ? r10 : r8;
    X_ = is_windows ? r11 : r9;
    INCX_ = is_windows ? r12 : r10;
    Y_ = is_windows ? rdi : r11;
    // incy is assumed to be 1 for A non-transposed.

    I_ = rax;
    A1_ = r13;
    A2_ = is_windows ? r8 : rdx;
    Y1_ = r14;
    LDA3_ = r15;

    // Set vector register type.
    switch (v_type) {
        default:
        case 2: kind_ = Operand::ZMM; break;
        case 1: kind_ = Operand::YMM; break;
        case 0: kind_ = Operand::XMM; break;
    }
    v_nelems_ = 16 / size_ << v_type;
    int um_vecs = utils::div_up(unroll_m, v_nelems_);

    // Assumptions on unroll_m and unroll_n.
    assert(math::is_pow2(unroll_m));
    assert(math::is_pow2(unroll_n));
    assert(unroll_n <= 8);
    assert(unroll_m >= v_nelems_);
    assert(um_vecs <= max_um_vecs_);

    unroll_m_ = unroll_m;
    unroll_n_ = unroll_n;

    // Assign vector registers
    int rn = 0;
    a_ = Xmm(kind_, rn++);
    for (int i = 0; i < um_vecs; i++)
        y_[i] = Xmm(kind_, rn++);

    for (int i = 0; i < unroll_n_; i++)
        x_[i] = Xmm(kind_, rn++);

    alpha_ = Xmm(kind_, rn++);
    scratch_ = Xmm(kind_, rn++);

    assert(IMPLICATION(has_avx512_, rn <= 32));
    assert(IMPLICATION(!has_avx512_, rn <= 16));

    // Assign stack variables.
    auto args_offset = get_size_of_abi_save_regs() + 8 + (is_windows ? 48 : 0);

    arg_lda_ = ptr[rsp + (args_offset - 16)];
    arg_x_ = ptr[rsp + (args_offset - 8)];
    arg_incx_ = ptr[rsp + (args_offset + 0)]; // Assumed 1 for A transpose.
    arg_y_ = ptr[rsp + (args_offset + 8)];
    arg_incy_ = ptr[rsp + (args_offset + 16)]; // Assumed 1 for A non-transpose.
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
