/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/gemm/amx/jit_avx512_core_amx_copy_kern.hpp"

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
static inline Zmm make_zmm(const Xmm &v) {
    return Zmm(v.getIdx());
}

void jit_avx512_core_amx_copy_kern::transpose(int s, const Ymm &dst1,
        const Ymm &dst2, const Ymm &src1, const Ymm &src2) {
    switch (s) {
        case 32:
            vshufi32x4(dst1, src1, src2, 0x44);
            vshufi32x4(dst2, src1, src2, 0xee);
            break;

        case 16:
            vshufi32x4(dst1, src1, src2, 0x88);
            vshufi32x4(dst2, src1, src2, 0xdd);
            vshufi32x4(dst1, dst1, dst1, 0xd8);
            vshufi32x4(dst2, dst2, dst2, 0xd8);
            break;

        case 8:
            vunpcklpd(dst1, src1, src2);
            vunpckhpd(dst2, src1, src2);
            break;

        case 4:
            vunpcklps(dst2, src1, src2);
            vunpckhps(src1, src1, src2);
            vunpcklpd(dst1, dst2, src1);
            vunpckhpd(dst2, dst2, src1);
            break;

        case 2:
            vpunpcklwd(dst2, src1, src2);
            vpunpckhwd(src1, src1, src2);
            vshufps(dst1, dst2, src1, 0x88);
            vshufps(dst2, dst2, src1, 0xdd);
            break;

        case 1:
            vpunpcklbw(dst1, src1, src2);
            vpunpckhbw(dst2, src1, src2);
            vpshuflw(dst1, dst1, 0xd8);
            vpshufhw(dst1, dst1, 0xd8);
            vpshuflw(dst2, dst2, 0xd8);
            vpshufhw(dst2, dst2, 0xd8);
            vpshufd(src1, dst1, 0xd8);
            vpshufd(src2, dst2, 0xd8);
            vpunpcklqdq(dst1, src1, src2);
            vpunpckhqdq(dst2, src1, src2);
            break;
    }
}

void jit_avx512_core_amx_copy_kern::amxtrans8(const Ymm &dst1, const Ymm &dst2,
        const Ymm &src1, const Ymm &src2, const Ymm &src3, const Ymm &src4) {
    vpunpcklbw(dst1, src1, src2);
    vpunpckhbw(dst2, src1, src2);
    vpunpcklbw(src1, src3, src4);
    vpunpckhbw(src2, src3, src4);
    vpunpcklwd(src3, dst1, src1);
    vpunpckhwd(src4, dst1, src1);
    vpunpcklwd(dst1, dst2, src2);
    vpunpckhwd(dst2, dst2, src2);
    vshufi32x4(src1, src3, src4, 0x00);
    vshufi32x4(src2, src3, src4, 0x03);
    vshufi32x4(src3, dst1, dst2, 0x00);
    vshufi32x4(src4, dst1, dst2, 0x03);
}

void jit_avx512_core_amx_copy_kern::amxtrans16(
        const Ymm &dst1, const Ymm &dst2, const Ymm &src1, const Ymm &src2) {
    vpunpcklwd(dst1, src1, src2);
    vpunpckhwd(dst2, src1, src2);
    vshufi32x4(src1, dst1, dst2, 0x88);
    vshufi32x4(src2, dst1, dst2, 0xdd);
    vshufi32x4(src1, src1, src1, 0xd8);
    vshufi32x4(src2, src2, src2, 0xd8);
}

void jit_avx512_core_amx_copy_kern::load(
        const Xmm &dst, const Address &src, bool corner) {
    if (!corner && isize_ == 1)
        vmovdqu8(dst, src);
    else if (!corner && isize_ == 2)
        vmovdqu16(dst, src);
    else if (corner && isize_ == 1)
        vmovdqu8(dst | k1 | T_z, src);
    else
        vmovdqu16(dst | k1 | T_z, src);
}

void jit_avx512_core_amx_copy_kern::store(const Address &dst, const Xmm &src) {
    if (size_ == 1)
        vmovdqu8(dst, src);
    else
        vmovdqu16(dst, src);
}

void jit_avx512_core_amx_copy_kern::kernel_AN(
        int unroll_x, int unroll_y, int step, Reg64 A, Reg64 B, bool corner) {
    // Transpose data.
    int u[] = {32, 16, 8, 4};
    for (int k = 0; k < nstages_; k++) {
        for (int j = 0; j < 8; j++)
            transpose(u[k], vecs_[k][0][j], vecs_[k][1][j], vecs_[k][0][j + 1],
                    vecs_[k][1][j + 1]);
    }

    // Store data.
    int k = 0;
    if (unroll_y >= 16)
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 2; i++) {
                Ymm v = vecs_[nstages_ - 1][i][j];
                store(ptr[B + ((k + step * 16) * lsize_ - offset_b_) * size_],
                        v);
                k++;
            }
    else
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 2; i++) {
                Ymm v = vecs_[nstages_ - 1][i][j];
                vmovdqu32(ptr[B
                                  + (k * unroll_y * (4 / size_)
                                            + step * 16 * lsize_ - offset_b_)
                                          * size_]
                                | k2,
                        v);
                k++;
            }
}

void jit_avx512_core_amx_copy_kern::kernel_BN(
        int unroll_x, int unroll_y, int step, Reg64 A, Reg64 B, bool corner) {
    // Store data.
    for (int i = 0; i < 16; i++)
        if (unroll_y >= i + 1)
            store(ptr[B + ((i + step * 16) * lsize_ - offset_b_) * size_],
                    src_[i]);
}

void jit_avx512_core_amx_copy_kern::kernel_AT(
        int unroll_x, int unroll_y, int step, Reg64 A, Reg64 B, bool corner) {
    Ymm v[16];

    // Transpose data.
    if (isize_ == 1) {
        for (int i = 0; i < 16; i++)
            v[i] = make_zmm(src_[i]);

        for (int i = 0; i < 16; i += 4)
            amxtrans8(tmp1_, tmp2_, src_[i], src_[i + 1], src_[i + 2],
                    src_[i + 3]);

        for (int j = 0; j < 2; j++)
            for (int i = j; i < 16; i += 4)
                vshufi32x4(v[i], v[i], v[i + 2], 0x44);

    } else {
        v[0] = tmp1_;
        v[1] = tmp2_;
        for (int i = 2; i < 16; i++)
            v[i] = src_[i - 2];

        for (int i = 0; i < 16; i += 2)
            amxtrans16(v[0], v[1], src_[i], src_[i + 1]);

        for (int i = 0; i < 16; i += 2)
            transpose(32, v[i], v[i + 1], src_[i], src_[i + 1]);
    }

    // Store data.
    if (!corner) {
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 16 / isize_; i += 4 / (isize_ * isize_))
                store(ptr[B
                              + ((i * isize_ / 2 + 32 / isize_ * j + step * 8)
                                                * lsize_
                                        - offset_b_)
                                      * size_],
                        v[isize_ * i + j]);

    } else {
        Label remainder_1_end, remainder_2_end;

        cmp(I_, 16);
        jl(remainder_1_end, T_NEAR);

        lea(T_, ptr[I_ - 16]);
        imul(T_, T_, step * lsize_ * isize_ / 2);
        kshiftrq(k3, k2, 16);

        for (int i = 0; i < 16; i += 4 / isize_) {
            auto b_off0 = ((i / 2 + step * 8) * lsize_ - offset_b_) * size_;
            auto b_off1 = (32 / isize_ * lsize_ - offset_b_) * size_;

            vmovdqu32(ptr[B + b_off0], v[i]);
            vmovdqu32(ptr[B + T_ + b_off1] | k3, v[i + 1]);

            if (i < 16 - 4 / isize_)
                lea(T_,
                        ptr[T_ + I_ * (4 / isize_ * size_)
                                - 2 / isize_ * lsize_ * size_]);
        }
        jmp(remainder_2_end, T_NEAR);
        L(remainder_1_end);

        lea(T_, ptr[I_]);
        imul(T_, T_, step * lsize_ * isize_ / 2);

        for (int i = 0; i < 16; i += 4 / isize_) {
            vmovdqu32(ptr[B + T_ - offset_b_ * size_] | k2, v[i]);

            if (i < 16 - 4 / isize_)
                lea(T_, ptr[T_ + I_ * (4 / isize_ * size_)]);
        }
        L(remainder_2_end);
    }
}

void jit_avx512_core_amx_copy_kern::kernel_BT(
        int unroll_x, int unroll_y, int step, Reg64 A, Reg64 B, bool corner) {
    // Transpose data.
    int u[] = {16, 8, 4, 2, 1};
    int *p_u = isize_ == 1 ? &u[1] : &u[0];
    for (int k = 0; k < nstages_; k++)
        for (int j = 0; j < 8; j++)
            transpose(p_u[k], vecs_[k][0][j], vecs_[k][1][j],
                    vecs_[k][0][j + 1], vecs_[k][1][j + 1]);

    // Store data.
    Label store_end;

    int k = 0;
    for (int o = 0; o < 2; o++)
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 2; i++) {
                if (corner) {
                    cmp(I_, k + 1);
                    jl(store_end, T_NEAR);
                }

                Ymm v = vecs_[nstages_ - 1][i][j];
                if (isize_ == 1) {
                    if (o > 0) vshufi32x4(v, v, v, 0x03);
                    store(ptr[B
                                  + (2 * k * lsize_ + step * 16 - offset_b_)
                                          * size_],
                            make_xmm(v));
                } else {
                    if (o > 0) vshuff32x4(v, v, v, 0xee);
                    store(ptr[B + (k * lsize_ + step * 16 - offset_b_) * size_],
                            make_ymm(v));
                }
                k++;
            }
    L(store_end);
}

void jit_avx512_core_amx_copy_kern::kernel(
        int unroll_x, int unroll_y, int step, Reg64 A, Reg64 B, bool corner) {

    // Load matrix.
    for (int i = 0; i < 16; i++) {
        Reg64 a = A2_;
        if (i < 4) a = A;

        int rem = i % 4;
        decltype(LDA_ * rem) lda_mult = (rem == 3) ? LDA3_ : LDA_ * rem;

        if (unroll_y >= i + 1) {
            load(src_[i], ptr[a + lda_mult + (0 - offset_a_) * isize_], corner);
            if (rem == 3) lea(A2_, ptr[a + LDA_ * 4]);
        } else {
            vxorps(src_[i], src_[i], src_[i]);
        }
    }

    if (!is_trans_ && is_a_)
        kernel_AN(unroll_x, unroll_y, step, A, B, corner);
    else if (!is_trans_ && !is_a_)
        kernel_BN(unroll_x, unroll_y, step, A, B, corner);
    else if (is_trans_ && is_a_)
        kernel_AT(unroll_x, unroll_y, step, A, B, corner);
    else
        kernel_BT(unroll_x, unroll_y, step, A, B, corner);
}

void jit_avx512_core_amx_copy_kern::copy_m(int unroll_m, int unroll_n) {
    if (is_trans_) {
        mov(B1_, B_);
        add(B_, unroll_m * unroll_n * size_);
    }

    Label kernel_loop, kernel_tail, kernel_tail_end;

    mov(I_, M_);
    sar(I_, lscale_);
    jle(kernel_tail, T_NEAR);

    Reg64 a[] = {A1_, A2_, A2_, A2_};
    Reg64 b = !is_trans_ ? B_ : B1_;
    int uy = !is_trans_ ? unroll_n : unroll_k_;
    int n_kernel_calls = uy >= 64 ? 4 : 2;

    L_aligned(kernel_loop);
    {
        for (int i = 0; i < n_kernel_calls; i++)
            kernel(unroll_n, unroll_n - i * 16, i, a[i], b, false);

        add(A1_, lsize_ * isize_);

        if (!is_trans_)
            add(B_, unroll_n * lsize_ * size_);
        else
            add(B1_, STRIDE_);
        dec(I_);
        jg(kernel_loop, T_NEAR);
    }

    L_aligned(kernel_tail);
    mov(I_, M_);
    and_(I_, lsize_ - 1);
    je(kernel_tail_end, T_NEAR);

    if (is_trans_) mov(B1_, BB_);

    for (int i = 0; i < n_kernel_calls; i++)
        kernel(unroll_n, unroll_n - i * 16, i, a[i], b, true);

    if (!is_trans_) {
        add(B_, unroll_n * lsize_ * size_);
    } else {
        imul(I_, I_, unroll_n * size_);
        add(BB_, I_);
    }

    L_aligned(kernel_tail_end);
}

void jit_avx512_core_amx_copy_kern::copy_ns(int unroll_n, Label &epilogue) {
    if (unroll_n > 0) {
        copy_ns(unroll_n - 1, epilogue);

        Label copy_m_end;
        cmp(N_, unroll_n);
        jg(copy_m_end, T_NEAR);

        copy_m(is_trans_ ? unroll_m_ : unroll_k_, unroll_n);
        jmp(epilogue, T_NEAR);

        L_aligned(copy_m_end);
    }
}

void jit_avx512_core_amx_copy_kern::copy_n(int unroll_n, Label &epilogue) {

    Label copy_m_loop, copy_m_end;

    cmp(N_, unroll_n);
    jl(copy_m_end, T_NEAR);

    L_aligned(copy_m_loop);
    {
        mov(A1_, A_);
        mov(I_, LDA_);
        imul(I_, I_, unroll_n);
        add(A_, I_);

        copy_m(is_trans_ ? unroll_m_ : unroll_k_, unroll_n);

        sub(N_, unroll_n);
        cmp(N_, unroll_n);
        jge(copy_m_loop, T_NEAR);
    }
    L_aligned(copy_m_end);

    mov(A1_, A_);
    cmp(N_, 0);
    jle(epilogue, T_NEAR);

    copy_ns(unroll_n - 1, epilogue);
}

void jit_avx512_core_amx_copy_kern::generate() {
    // Prologue
    preamble();
    sub(rsp, stack_alloc_size_);

    mov(M_, qword[M_]);
    mov(N_, qword[N_]);
    mov(LDA_, qword[LDA_]);

    if (is_windows) mov(B_, arg_b_);

    sub(A_, -offset_a_ * isize_);
    sub(B_, -offset_b_ * size_);

    sal(LDA_, isize_ - 1);
    lea(LDA3_, ptr[LDA_ + LDA_ * 2]);

    // Generate masks.
    mov(rbx, rcx);
    mov(rcx, M_);
    and_(rcx, lsize_ - 1);
    mov(rax, -1);
    shl(rax, cl);

    if (isize_ == 1) {
        kmovq(k1, rax);
        knotq(k1, k1);
    } else {
        kmovd(k1, eax);
        knotd(k1, k1);
    }

    if (is_a_ && !is_trans_) {
        mov(rcx, N_);
        and_(rcx, 15);
        mov(rax, -1);
        shl(rax, cl);
        kmovw(k2, eax);
        knotw(k2, k2);
    } else if (is_a_ && is_trans_) {
        mov(rcx, M_);
        and_(rcx, 31);
        mov(rax, -1);
        shl(rax, cl);
        kmovq(k2, rax);
        knotq(k2, k2);
    }
    mov(rcx, rbx);

    if (is_trans_) {
        mov(STRIDE_, N_);
        add(STRIDE_, unroll_k_ - 1);
        and_(STRIDE_, ~(unroll_k_ - 1));

        mov(BB_, M_);
        and_(BB_, ~(unroll_m_ - 1));
        imul(BB_, STRIDE_);
        lea(BB_, ptr[B_ + BB_ * size_]);
        imul(STRIDE_, STRIDE_, unroll_m_ * size_);
    }

    Label epilogue;

    copy_n(is_trans_ ? unroll_k_ : unroll_m_, epilogue);

    L(epilogue);

    // Epilogue.
    add(rsp, stack_alloc_size_);
    postamble();
}

jit_avx512_core_amx_copy_kern::jit_avx512_core_amx_copy_kern(
        bool is_a, bool is_trans, int isize)
    : jit_generator(nullptr, 800000), arg_b_(0) {

    is_a_ = is_a;
    is_trans_ = is_trans;
    isize_ = isize;
    size_ = isize;

    assert(utils::one_of(isize_, 2, 1));
    assert(isize_ == size_);

    unroll_k_ = 64 / isize_;

    if (!is_trans_ && size_ == 1) {
        lscale_ = 6;
        lsize_ = 64;
    } else if (!is_trans && size_ == 2) {
        lscale_ = 5;
        lsize_ = 32;
    } else if (is_trans && size_ == 1) {
        lscale_ = 5;
        lsize_ = 32;
    } else {
        lscale_ = 5;
        lsize_ = 32;
    }

    // Assign integer registers
    if (!is_trans_) {
        M_ = is_windows ? rcx : rdi;
        N_ = is_windows ? rdx : rsi;
    } else {
        M_ = is_windows ? rdx : rsi;
        N_ = is_windows ? rcx : rdi;
    }
    A_ = is_windows ? r8 : rdx;
    LDA_ = is_windows ? r9 : rcx;
    B_ = is_windows ? rdi : r9;

    I_ = rax;
    A1_ = is_windows ? rsi : r8;
    A2_ = r10;
    LDA3_ = r11;
    B1_ = r12;
    BB_ = r13;
    STRIDE_ = r14;
    T_ = r15;

    // Assign vector registers
    bool is_int8_trans = isize_ == 1 && is_trans_;
    tmp1_ = is_int8_trans ? ymm16 : zmm16;
    tmp2_ = is_int8_trans ? ymm17 : zmm17;

    for (int i = 0; i < 16; i++)
        src_[i] = is_int8_trans ? Ymm(i) : Zmm(i);

    for (int k = 0; k < nstages_; k++)
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 8 + 1; j++) {
                int idx = idx_[k][i][j];
                vecs_[k][i][j] = is_int8_trans ? Ymm(idx) : Zmm(idx);
            }

    // Assign stack variables
    stack_alloc_size_ = 64;
    auto args_offset = stack_alloc_size_ + get_size_of_abi_save_regs() + 8
            + (is_windows ? 48 : 0);

    arg_b_ = ptr[rsp + (args_offset - 8)];
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
