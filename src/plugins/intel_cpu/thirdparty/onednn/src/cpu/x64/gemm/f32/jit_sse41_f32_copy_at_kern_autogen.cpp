/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include "cpu/x64/gemm/f32/common_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_sse41_f32_copy_at_kern::jit_sse41_f32_copy_at_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_sse41_f32_copy_at_kern::generate() {

#ifndef _WIN32
#define M rdi
#define N rsi
#define A rdx
#define LDA rcx
#define ALPHA r8
#define B r9

#define I rax
#define A1 r10
#define A2 r8
#define LDA3 r11

#else
#define M rcx
#define N rdx
#define A r8
#define LDA r9
#define ALPHA rsi
#define B rdi
#define I rax
#define A1 r10
#define A2 rsi
#define LDA3 r11

#define ARG_ALPHA 40 + stacksize + rsp
#define ARG_B 48 + stacksize + rsp

#endif

    inLocalLabel();
    {
        std::vector<Xbyak::Label> labels(65);
        preamble();
#ifdef _WIN32
        auto stacksize = get_size_of_abi_save_regs();
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(M, qword[M]);
        mov(N, qword[N]);
        mov(LDA, qword[LDA]);
        sub(A, 0x0);
        sub(B, -128);
        shl(LDA, 0x2);
        lea(LDA3, ptr[LDA + LDA * 2]);
        movss(xmm6, dword[ALPHA]);
        pshufd(xmm6, xmm6, 0x0);
        pcmpeqb(xmm3, xmm3);
        psrld(xmm3, 0x17);
        pslld(xmm3, 0x19);
        psrld(xmm3, 0x2);
        pcmpeqb(xmm4, xmm4);
        pslld(xmm4, 0x1f);
        ucomiss(xmm6, xmm3);
        jne(labels[19], T_NEAR);
        cmp(N, 0x8);
        jl(labels[3], T_NEAR);
        align(4);

        L(labels[22]);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[0], T_NEAR);
        align(4);

        L(labels[28]);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movups(xmm2, xword[A1 + LDA * 2]);
        movups(xmm3, xword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x60], xmm1);
        movups(xword[B - 0x40], xmm2);
        movups(xword[B - 0x20], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        movups(xmm0, xword[A2]);
        movups(xmm1, xword[A2 + LDA * 1]);
        movups(xmm2, xword[A2 + LDA * 2]);
        movups(xmm3, xword[A2 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        movups(xword[B - 0x70], xmm0);
        movups(xword[B - 0x50], xmm1);
        movups(xword[B - 0x30], xmm2);
        movups(xword[B - 0x10], xmm3);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -16);
        sub(B, -128);
        dec(I);
        jg(labels[28], T_NEAR);
        align(4);

        L(labels[0]);
        test(M, 0x2);
        jle(labels[1], T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        movhps(xmm0, qword[A1 + LDA * 2]);
        movhps(xmm1, qword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x60], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        movsd(xmm0, qword[A2]);
        movsd(xmm1, qword[A2 + LDA * 1]);
        movhps(xmm0, qword[A2 + LDA * 2]);
        movhps(xmm1, qword[A2 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        movups(xword[B - 0x70], xmm0);
        movups(xword[B - 0x50], xmm1);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -8);
        sub(B, -64);
        align(4);

        L(labels[1]);
        test(M, 0x1);
        jle(labels[2], T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A1 + LDA * 2]);
        movss(xmm3, dword[A1 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        movups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        movss(xmm0, dword[A2]);
        movss(xmm1, dword[A2 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A2 + LDA * 2]);
        movss(xmm3, dword[A2 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        movups(xword[B - 0x70], xmm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -32);
        align(4);

        L(labels[2]);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(labels[22], T_NEAR);
        align(4);

        L(labels[3]);
        cmp(N, 0x4);
        jl(labels[8], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[5], T_NEAR);
        align(4);

        L(labels[4]);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movups(xmm2, xword[A1 + LDA * 2]);
        movups(xmm3, xword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x70], xmm1);
        movups(xword[B - 0x60], xmm2);
        movups(xword[B - 0x50], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -16);
        sub(B, -64);
        dec(I);
        jg(labels[4], T_NEAR);
        align(4);

        L(labels[5]);
        test(M, 0x2);
        jle(labels[6], T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        movhps(xmm0, qword[A1 + LDA * 2]);
        movhps(xmm1, qword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x70], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -8);
        sub(B, -32);
        align(4);

        L(labels[6]);
        test(M, 0x1);
        jle(labels[7], T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A1 + LDA * 2]);
        movss(xmm3, dword[A1 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        movups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -4);
        sub(B, -16);
        align(4);

        L(labels[7]);
        sub(N, 0x4);
        align(4);

        L(labels[8]);
        cmp(N, 0x2);
        jl(labels[13], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[10], T_NEAR);
        align(4);

        L(labels[9]);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm4);
        movlps(qword[B - 0x80], xmm0);
        movhps(qword[B - 0x78], xmm0);
        movlps(qword[B - 0x70], xmm1);
        movhps(qword[B - 0x68], xmm1);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -16);
        sub(B, -32);
        dec(I);
        jg(labels[9], T_NEAR);
        align(4);

        L(labels[10]);
        test(M, 0x2);
        jle(labels[11], T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movlps(qword[B - 0x80], xmm0);
        movhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(labels[11]);
        test(M, 0x1);
        jle(labels[12], T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(labels[12]);
        sub(N, 0x2);
        align(4);

        L(labels[13]);
        cmp(N, 0x1);
        jl(labels[18], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[15], T_NEAR);
        align(4);

        L(labels[14]);
        movups(xmm0, xword[A1]);
        pshufd(xmm1, xmm0, 0x55);
        pshufd(xmm2, xmm0, 0xaa);
        pshufd(xmm3, xmm0, 0xff);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        movss(dword[B - 0x78], xmm2);
        movss(dword[B - 0x74], xmm3);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -16);
        sub(B, -16);
        dec(I);
        jg(labels[14], T_NEAR);
        align(4);

        L(labels[15]);
        test(M, 0x2);
        jle(labels[16], T_NEAR);
        movsd(xmm0, qword[A1]);
        pshufd(xmm1, xmm0, 0x55);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(labels[16]);
        test(M, 0x1);
        jle(labels[17], T_NEAR);
        movss(xmm0, dword[A1]);
        movss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(labels[17]);
        sub(N, 0x1);
        align(4);

        L(labels[18]);
        jmp(labels[64], T_NEAR);
        align(4);

        L(labels[19]);
        xorps(xmm3, xmm4);
        ucomiss(xmm6, xmm3);
        jne(labels[43], T_NEAR);
        movaps(xmm6, xmm4);
        cmp(N, 0x8);
        jl(labels[26], T_NEAR);
        align(4);

        L(labels[20]);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[23], T_NEAR);
        align(4);

        L(labels[21]);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movups(xmm2, xword[A1 + LDA * 2]);
        movups(xmm3, xword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        xorps(xmm2, xmm6);
        xorps(xmm3, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x60], xmm1);
        movups(xword[B - 0x40], xmm2);
        movups(xword[B - 0x20], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        movups(xmm0, xword[A2]);
        movups(xmm1, xword[A2 + LDA * 1]);
        movups(xmm2, xword[A2 + LDA * 2]);
        movups(xmm3, xword[A2 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        xorps(xmm2, xmm6);
        xorps(xmm3, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xword[B - 0x50], xmm1);
        movups(xword[B - 0x30], xmm2);
        movups(xword[B - 0x10], xmm3);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -16);
        sub(B, -128);
        dec(I);
        jg(labels[21], T_NEAR);
        align(4);

        L(labels[23]);
        test(M, 0x2);
        jle(labels[24], T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        movhps(xmm0, qword[A1 + LDA * 2]);
        movhps(xmm1, qword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x60], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        movsd(xmm0, qword[A2]);
        movsd(xmm1, qword[A2 + LDA * 1]);
        movhps(xmm0, qword[A2 + LDA * 2]);
        movhps(xmm1, qword[A2 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xword[B - 0x50], xmm1);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -8);
        sub(B, -64);
        align(4);

        L(labels[24]);
        test(M, 0x1);
        jle(labels[25], T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A1 + LDA * 2]);
        movss(xmm3, dword[A1 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        movss(xmm0, dword[A2]);
        movss(xmm1, dword[A2 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A2 + LDA * 2]);
        movss(xmm3, dword[A2 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -32);
        align(4);

        L(labels[25]);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(labels[20], T_NEAR);
        align(4);

        L(labels[26]);
        cmp(N, 0x4);
        jl(labels[32], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[29], T_NEAR);
        align(4);

        L(labels[27]);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movups(xmm2, xword[A1 + LDA * 2]);
        movups(xmm3, xword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        xorps(xmm2, xmm6);
        xorps(xmm3, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x70], xmm1);
        movups(xword[B - 0x60], xmm2);
        movups(xword[B - 0x50], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -16);
        sub(B, -64);
        dec(I);
        jg(labels[27], T_NEAR);
        align(4);

        L(labels[29]);
        test(M, 0x2);
        jle(labels[30], T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        movhps(xmm0, qword[A1 + LDA * 2]);
        movhps(xmm1, qword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x70], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -8);
        sub(B, -32);
        align(4);

        L(labels[30]);
        test(M, 0x1);
        jle(labels[31], T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A1 + LDA * 2]);
        movss(xmm3, dword[A1 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -4);
        sub(B, -16);
        align(4);

        L(labels[31]);
        sub(N, 0x4);
        align(4);

        L(labels[32]);
        cmp(N, 0x2);
        jl(labels[37], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[34], T_NEAR);
        align(4);

        L(labels[33]);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm4);
        xorps(xmm0, xmm6);
        xorps(xmm1, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movhps(qword[B - 0x78], xmm0);
        movlps(qword[B - 0x70], xmm1);
        movhps(qword[B - 0x68], xmm1);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -16);
        sub(B, -32);
        dec(I);
        jg(labels[33], T_NEAR);
        align(4);

        L(labels[34]);
        test(M, 0x2);
        jle(labels[35], T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(labels[35]);
        test(M, 0x1);
        jle(labels[36], T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(labels[36]);
        sub(N, 0x2);
        align(4);

        L(labels[37]);
        cmp(N, 0x1);
        jl(labels[42], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[39], T_NEAR);
        align(4);

        L(labels[38]);
        movups(xmm0, xword[A1]);
        xorps(xmm0, xmm6);
        pshufd(xmm1, xmm0, 0x55);
        pshufd(xmm2, xmm0, 0xaa);
        pshufd(xmm3, xmm0, 0xff);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        movss(dword[B - 0x78], xmm2);
        movss(dword[B - 0x74], xmm3);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -16);
        sub(B, -16);
        dec(I);
        jg(labels[38], T_NEAR);
        align(4);

        L(labels[39]);
        test(M, 0x2);
        jle(labels[40], T_NEAR);
        movsd(xmm0, qword[A1]);
        xorps(xmm0, xmm6);
        pshufd(xmm1, xmm0, 0x55);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(labels[40]);
        test(M, 0x1);
        jle(labels[41], T_NEAR);
        movss(xmm0, dword[A1]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(labels[41]);
        sub(N, 0x1);
        align(4);

        L(labels[42]);
        jmp(labels[64], T_NEAR);
        align(4);

        L(labels[43]);
        cmp(N, 0x8);
        jl(labels[49], T_NEAR);
        align(4);

        L(labels[44]);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[46], T_NEAR);
        align(4);

        L(labels[45]);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movups(xmm2, xword[A1 + LDA * 2]);
        movups(xmm3, xword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        mulps(xmm2, xmm6);
        mulps(xmm3, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x60], xmm1);
        movups(xword[B - 0x40], xmm2);
        movups(xword[B - 0x20], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        movups(xmm0, xword[A2]);
        movups(xmm1, xword[A2 + LDA * 1]);
        movups(xmm2, xword[A2 + LDA * 2]);
        movups(xmm3, xword[A2 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        mulps(xmm2, xmm6);
        mulps(xmm3, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xword[B - 0x50], xmm1);
        movups(xword[B - 0x30], xmm2);
        movups(xword[B - 0x10], xmm3);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -16);
        sub(B, -128);
        dec(I);
        jg(labels[45], T_NEAR);
        align(4);

        L(labels[46]);
        test(M, 0x2);
        jle(labels[47], T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        movhps(xmm0, qword[A1 + LDA * 2]);
        movhps(xmm1, qword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x60], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        movsd(xmm0, qword[A2]);
        movsd(xmm1, qword[A2 + LDA * 1]);
        movhps(xmm0, qword[A2 + LDA * 2]);
        movhps(xmm1, qword[A2 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xword[B - 0x50], xmm1);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -8);
        sub(B, -64);
        align(4);

        L(labels[47]);
        test(M, 0x1);
        jle(labels[48], T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A1 + LDA * 2]);
        movss(xmm3, dword[A1 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        movss(xmm0, dword[A2]);
        movss(xmm1, dword[A2 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A2 + LDA * 2]);
        movss(xmm3, dword[A2 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -32);
        align(4);

        L(labels[48]);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(labels[44], T_NEAR);
        align(4);

        L(labels[49]);
        cmp(N, 0x4);
        jl(labels[54], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[51], T_NEAR);
        align(4);

        L(labels[50]);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movups(xmm2, xword[A1 + LDA * 2]);
        movups(xmm3, xword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm5, xmm2);
        unpcklps(xmm2, xmm3);
        unpckhps(xmm5, xmm3);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm2);
        unpckhpd(xmm1, xmm2);
        movaps(xmm2, xmm4);
        movaps(xmm3, xmm4);
        unpcklpd(xmm2, xmm5);
        unpckhpd(xmm3, xmm5);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        mulps(xmm2, xmm6);
        mulps(xmm3, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x70], xmm1);
        movups(xword[B - 0x60], xmm2);
        movups(xword[B - 0x50], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -16);
        sub(B, -64);
        dec(I);
        jg(labels[50], T_NEAR);
        align(4);

        L(labels[51]);
        test(M, 0x2);
        jle(labels[52], T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        movhps(xmm0, qword[A1 + LDA * 2]);
        movhps(xmm1, qword[A1 + LDA3 * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm0);
        unpcklpd(xmm0, xmm4);
        unpckhpd(xmm1, xmm4);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xword[B - 0x70], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -8);
        sub(B, -32);
        align(4);

        L(labels[52]);
        test(M, 0x1);
        jle(labels[53], T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        movss(xmm2, dword[A1 + LDA * 2]);
        movss(xmm3, dword[A1 + LDA3 * 1]);
        unpcklps(xmm2, xmm3);
        unpcklpd(xmm0, xmm2);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -4);
        sub(B, -16);
        align(4);

        L(labels[53]);
        sub(N, 0x4);
        align(4);

        L(labels[54]);
        cmp(N, 0x2);
        jl(labels[59], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[56], T_NEAR);
        align(4);

        L(labels[55]);
        movups(xmm0, xword[A1]);
        movups(xmm1, xword[A1 + LDA * 1]);
        movaps(xmm4, xmm0);
        unpcklps(xmm0, xmm1);
        unpckhps(xmm4, xmm1);
        movaps(xmm1, xmm4);
        mulps(xmm0, xmm6);
        mulps(xmm1, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movhps(qword[B - 0x78], xmm0);
        movlps(qword[B - 0x70], xmm1);
        movhps(qword[B - 0x68], xmm1);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -16);
        sub(B, -32);
        dec(I);
        jg(labels[55], T_NEAR);
        align(4);

        L(labels[56]);
        test(M, 0x2);
        jle(labels[57], T_NEAR);
        movsd(xmm0, qword[A1]);
        movsd(xmm1, qword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(labels[57]);
        test(M, 0x1);
        jle(labels[58], T_NEAR);
        movss(xmm0, dword[A1]);
        movss(xmm1, dword[A1 + LDA * 1]);
        unpcklps(xmm0, xmm1);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(labels[58]);
        sub(N, 0x2);
        align(4);

        L(labels[59]);
        cmp(N, 0x1);
        jl(labels[64], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[61], T_NEAR);
        align(4);

        L(labels[60]);
        movups(xmm0, xword[A1]);
        mulps(xmm0, xmm6);
        pshufd(xmm1, xmm0, 0x55);
        pshufd(xmm2, xmm0, 0xaa);
        pshufd(xmm3, xmm0, 0xff);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        movss(dword[B - 0x78], xmm2);
        movss(dword[B - 0x74], xmm3);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -16);
        sub(B, -16);
        dec(I);
        jg(labels[60], T_NEAR);
        align(4);

        L(labels[61]);
        test(M, 0x2);
        jle(labels[62], T_NEAR);
        movsd(xmm0, qword[A1]);
        mulps(xmm0, xmm6);
        pshufd(xmm1, xmm0, 0x55);
        movss(dword[B - 0x80], xmm0);
        movss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(labels[62]);
        test(M, 0x1);
        jle(labels[63], T_NEAR);
        movss(xmm0, dword[A1]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(labels[63]);
        sub(N, 0x1);
        align(4);

        L(labels[64]);

        postamble();
    }
    outLocalLabel();

#undef M
#undef N
#undef A
#undef LDA
#undef ALPHA
#undef B
#undef I
#undef A1
#undef A2
#undef LDA3
#ifdef _WIN32
#undef ARG_ALPHA
#undef ARG_B
#endif
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
