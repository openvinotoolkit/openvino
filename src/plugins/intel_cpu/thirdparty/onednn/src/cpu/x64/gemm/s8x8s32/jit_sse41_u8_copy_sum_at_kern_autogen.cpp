/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "cpu/x64/gemm/s8x8s32/common_u8.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_sse41_u8_copy_sum_at_kern::jit_sse41_u8_copy_sum_at_kern()
    : jit_generator(nullptr, U8_COPY_KERNEL_CODE_SIZE) {}

void jit_sse41_u8_copy_sum_at_kern::generate() {

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

#define ARG_BIAS (24 + stacksize + rsp)

#else

#define M rcx
#define N rdx
#define A r8
#define LDA r9
#define ALPHA rax
#define B rdi

#define I rax
#define A1 rsi
#define A2 r10
#define LDA3 r11

#define ARG_ALPHA 40 + stacksize + rsp
#define ARG_B 48 + stacksize + rsp
#define ARG_BIAS 72 + stacksize + rsp

#endif

    inLocalLabel();
    {
        std::vector<Xbyak::Label> labels(40);

        preamble();
        auto stacksize = get_size_of_abi_save_regs();
#ifdef _WIN32
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(N, qword[N]);
        mov(M, qword[M]);
        mov(LDA, qword[LDA]);
        sub(A, -128);
        sub(B, -128);
        lea(LDA3, ptr[LDA + LDA * 2]);
        cmp(N, 0x10);
        jl(labels[29], T_NEAR);
        align(4);

        L(labels[22]);
        mov(A1, A);
        mov(I, LDA);
        shl(I, 0x4);
        add(A, I);
        pxor(xmm8, xmm8);
        pxor(xmm9, xmm9);
        pxor(xmm10, xmm10);
        pxor(xmm11, xmm11);
        mov(I, M);
        sar(I, 0x4);
        jle(labels[23], T_NEAR);
        align(4);

        L(labels[24]);
        movdqu(xmm0, xword[A1 - 0x80]);
        movdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A1 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B - 0x40], xmm1);
        pmovsxbw(xmm5, xmm4);
        movhlps(xmm6, xmm4);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B], xmm4);
        pmovsxbw(xmm5, xmm3);
        movhlps(xmm6, xmm3);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B + 0x40], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x70], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x30], xmm1);
        pmovsxbw(xmm5, xmm4);
        movhlps(xmm6, xmm4);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B + 0x10], xmm4);
        pmovsxbw(xmm5, xmm3);
        movhlps(xmm6, xmm3);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B + 0x50], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        movdqu(xword[B - 0x60], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        movdqu(xword[B - 0x20], xmm1);
        pmovsxbw(xmm5, xmm4);
        movhlps(xmm6, xmm4);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        movdqu(xword[B + 0x20], xmm4);
        pmovsxbw(xmm5, xmm3);
        movhlps(xmm6, xmm3);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        movdqu(xword[B + 0x60], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm11, xmm5);
        movdqu(xword[B - 0x50], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm11, xmm5);
        movdqu(xword[B - 0x10], xmm1);
        pmovsxbw(xmm5, xmm4);
        movhlps(xmm6, xmm4);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm11, xmm5);
        movdqu(xword[B + 0x30], xmm4);
        pmovsxbw(xmm5, xmm3);
        movhlps(xmm6, xmm3);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm11, xmm5);
        movdqu(xword[B + 0x70], xmm3);
        sub(A1, -16);
        sub(B, -256);
        dec(I);
        jg(labels[24], T_NEAR);
        align(4);

        L(labels[23]);
        test(M, 0x8);
        jle(labels[25], T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        movq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A1 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B - 0x40], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x70], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x30], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        movdqu(xword[B - 0x60], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        movdqu(xword[B - 0x20], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm11, xmm5);
        movdqu(xword[B - 0x50], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm11, xmm5);
        movdqu(xword[B - 0x10], xmm1);
        sub(A1, -8);
        sub(B, -128);
        align(4);

        L(labels[25]);
        test(M, 0x4);
        jle(labels[26], T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A1 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x70], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        movdqu(xword[B - 0x60], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm11, xmm5);
        movdqu(xword[B - 0x50], xmm0);
        sub(A1, -4);
        sub(B, -64);
        align(4);

        L(labels[26]);
        test(M, 0x2);
        jle(labels[27], T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A1 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        pinsrw(xmm0, eax, 0x3);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrw(xmm0, eax, 0x7);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm6, xmm6);
        pmovsxwd(xmm6, xmm6);
        paddd(xmm9, xmm6);
        movdqu(xword[B - 0x80], xmm0);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrw(xmm0, eax, 0x3);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x7);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm6, xmm6);
        pmovsxwd(xmm6, xmm6);
        paddd(xmm11, xmm6);
        movdqu(xword[B - 0x70], xmm0);
        sub(A1, -2);
        sub(B, -32);
        align(4);

        L(labels[27]);
        test(M, 0x1);
        jle(labels[28], T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A1 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A1 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A1 + LDA * 4]);
        pinsrb(xmm0, eax, 0x3);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x4);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x5);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x6);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrb(xmm0, eax, 0x7);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x8);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x9);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0xa);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        lea(A2, ptr[A2 + LDA * 4]);
        pinsrb(xmm0, eax, 0xb);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0xc);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0xd);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0xe);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        pinsrb(xmm0, eax, 0xf);
        pmovsxbd(xmm5, xmm0);
        paddd(xmm8, xmm5);
        pshufd(xmm6, xmm0, 0x55);
        pmovsxbd(xmm6, xmm6);
        paddd(xmm9, xmm6);
        pshufd(xmm5, xmm0, 0xaa);
        pmovsxbd(xmm5, xmm5);
        paddd(xmm10, xmm5);
        pshufd(xmm6, xmm0, 0xff);
        pmovsxbd(xmm6, xmm6);
        paddd(xmm11, xmm6);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[28]);
        mov(A1, qword[ARG_BIAS]);
        movdqu(xword[A1], xmm8);
        movdqu(xword[A1 + 0x10], xmm9);
        movdqu(xword[A1 + 0x20], xmm10);
        movdqu(xword[A1 + 0x30], xmm11);
        add(qword[ARG_BIAS], 0x40);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(labels[22], T_NEAR);
        align(4);

        L(labels[29]);
        cmp(N, 0x8);
        jl(labels[37], T_NEAR);
        align(4);

        L(labels[30]);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        lea(I, ptr[A1 + LDA * 8]);
        mov(A, I);
        pxor(xmm8, xmm8);
        pxor(xmm9, xmm9);
        mov(I, M);
        sar(I, 0x4);
        jle(labels[32], T_NEAR);
        align(4);

        L(labels[31]);
        movdqu(xmm0, xword[A1 - 0x80]);
        movdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A1 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -16);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B - 0x60], xmm1);
        pmovsxbw(xmm5, xmm4);
        movhlps(xmm6, xmm4);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B - 0x40], xmm4);
        pmovsxbw(xmm5, xmm3);
        movhlps(xmm6, xmm3);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B - 0x20], xmm3);
        movdqu(xmm0, xword[A2 - 0x80]);
        movdqu(xmm1, xword[A2 + LDA * 1 - 0x80]);
        movdqu(xmm2, xword[A2 + LDA * 2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA3 * 1 - 0x80]);
        sub(A2, -16);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x70], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x50], xmm1);
        pmovsxbw(xmm5, xmm4);
        movhlps(xmm6, xmm4);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x30], xmm4);
        pmovsxbw(xmm5, xmm3);
        movhlps(xmm6, xmm3);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x10], xmm3);
        sub(B, -128);
        dec(I);
        jg(labels[31], T_NEAR);
        align(4);

        L(labels[32]);
        test(M, 0x8);
        jle(labels[33], T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        movq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A1 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -8);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B - 0x60], xmm1);
        movq(xmm0, qword[A2 - 0x80]);
        movq(xmm1, qword[A2 + LDA * 1 - 0x80]);
        movq(xmm2, qword[A2 + LDA * 2 - 0x80]);
        movq(xmm3, qword[A2 + LDA3 * 1 - 0x80]);
        sub(A2, -8);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x70], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x50], xmm1);
        sub(B, -64);
        align(4);

        L(labels[33]);
        test(M, 0x4);
        jle(labels[34], T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A1 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -4);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        movd(xmm0, dword[A2 - 0x80]);
        movd(xmm1, dword[A2 + LDA * 1 - 0x80]);
        movd(xmm2, dword[A2 + LDA * 2 - 0x80]);
        movd(xmm3, dword[A2 + LDA3 * 1 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm9, xmm5);
        movdqu(xword[B - 0x70], xmm0);
        sub(B, -32);
        align(4);

        L(labels[34]);
        test(M, 0x2);
        jle(labels[35], T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A1 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A1 + LDA3 * 1 - 0x80]);
        sub(A1, -2);
        pinsrw(xmm0, eax, 0x3);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x4);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        pinsrw(xmm0, eax, 0x5);
        mov(ax, word[A2 + LDA * 2 - 0x80]);
        pinsrw(xmm0, eax, 0x6);
        mov(ax, word[A2 + LDA3 * 1 - 0x80]);
        sub(A2, -2);
        pinsrw(xmm0, eax, 0x7);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm8, xmm5);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm6, xmm6);
        pmovsxwd(xmm6, xmm6);
        paddd(xmm9, xmm6);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[35]);
        test(M, 0x1);
        jle(labels[36], T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A1 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A1 + LDA3 * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x3);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x4);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x5);
        mov(al, byte[A2 + LDA * 2 - 0x80]);
        pinsrb(xmm0, eax, 0x6);
        mov(al, byte[A2 + LDA3 * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x7);
        pmovsxbd(xmm5, xmm0);
        pshufd(xmm6, xmm0, 0x55);
        pmovsxbd(xmm6, xmm6);
        paddd(xmm8, xmm5);
        paddd(xmm9, xmm6);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[36]);
        mov(A1, qword[ARG_BIAS]);
        movdqu(xword[A1], xmm8);
        movdqu(xword[A1 + 0x10], xmm9);
        add(qword[ARG_BIAS], 0x20);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(labels[30], T_NEAR);
        align(4);

        L(labels[37]);
        cmp(N, 0x4);
        jl(labels[5], T_NEAR);
        align(4);

        L(labels[38]);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 2]);
        lea(I, ptr[A1 + LDA * 4]);
        mov(A, I);
        pxor(xmm7, xmm7);
        mov(I, M);
        sar(I, 0x4);
        jle(labels[0], T_NEAR);
        align(4);

        L(labels[39]);
        movdqu(xmm0, xword[A1 - 0x80]);
        movdqu(xmm1, xword[A1 + LDA * 1 - 0x80]);
        sub(A1, -16);
        movdqu(xmm2, xword[A2 - 0x80]);
        movdqu(xmm3, xword[A2 + LDA * 1 - 0x80]);
        sub(A2, -16);
        movdqa(xmm4, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm4, xmm1);
        movdqa(xmm5, xmm2);
        punpckldq(xmm2, xmm3);
        punpckhdq(xmm5, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        movdqa(xmm3, xmm4);
        punpcklqdq(xmm4, xmm5);
        punpckhqdq(xmm3, xmm5);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x70], xmm1);
        pmovsxbw(xmm5, xmm4);
        movhlps(xmm6, xmm4);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x60], xmm4);
        pmovsxbw(xmm5, xmm3);
        movhlps(xmm6, xmm3);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x50], xmm3);
        sub(B, -64);
        dec(I);
        jg(labels[39], T_NEAR);
        align(4);

        L(labels[0]);
        test(M, 0x8);
        jle(labels[1], T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        movq(xmm1, qword[A1 + LDA * 1 - 0x80]);
        sub(A1, -8);
        movq(xmm2, qword[A2 - 0x80]);
        movq(xmm3, qword[A2 + LDA * 1 - 0x80]);
        sub(A2, -8);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        movdqa(xmm1, xmm0);
        punpcklqdq(xmm0, xmm2);
        punpckhqdq(xmm1, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        pmovsxbw(xmm5, xmm1);
        movhlps(xmm6, xmm1);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x70], xmm1);
        sub(B, -32);
        align(4);

        L(labels[1]);
        test(M, 0x4);
        jle(labels[2], T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        movd(xmm1, dword[A1 + LDA * 1 - 0x80]);
        sub(A1, -4);
        movd(xmm2, dword[A2 - 0x80]);
        movd(xmm3, dword[A2 + LDA * 1 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        punpckldq(xmm2, xmm3);
        punpcklqdq(xmm0, xmm2);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[2]);
        test(M, 0x2);
        jle(labels[3], T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A1 + LDA * 1 - 0x80]);
        sub(A1, -2);
        pinsrw(xmm0, eax, 0x1);
        mov(ax, word[A2 - 0x80]);
        pinsrw(xmm0, eax, 0x2);
        mov(ax, word[A2 + LDA * 1 - 0x80]);
        sub(A2, -2);
        pinsrw(xmm0, eax, 0x3);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[3]);
        test(M, 0x1);
        jle(labels[4], T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(al, byte[A1 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x2);
        mov(al, byte[A2 + LDA * 1 - 0x80]);
        pinsrb(xmm0, eax, 0x3);
        pmovsxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[4]);
        mov(A1, qword[ARG_BIAS]);
        movdqu(xword[A1], xmm7);
        add(qword[ARG_BIAS], 0x10);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(labels[38], T_NEAR);
        align(4);

        L(labels[5]);
        cmp(N, 0x2);
        jl(labels[13], T_NEAR);
        align(4);

        L(labels[6]);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 1]);
        lea(I, ptr[A1 + LDA * 2]);
        mov(A, I);
        pxor(xmm7, xmm7);
        mov(I, M);
        sar(I, 0x4);
        jle(labels[8], T_NEAR);
        align(4);

        L(labels[7]);
        movdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        movdqu(xmm1, xword[A2 - 0x80]);
        sub(A2, -16);
        movdqa(xmm2, xmm0);
        punpckldq(xmm0, xmm1);
        punpckhdq(xmm2, xmm1);
        pshufd(xmm6, xmm0, 0xd8);
        pmovsxbw(xmm5, xmm6);
        movhlps(xmm6, xmm6);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        pshufd(xmm6, xmm2, 0xd8);
        pmovsxbw(xmm5, xmm6);
        movhlps(xmm6, xmm6);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x70], xmm2);
        sub(B, -32);
        dec(I);
        jg(labels[7], T_NEAR);
        align(4);

        L(labels[8]);
        test(M, 0x8);
        jle(labels[9], T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        movq(xmm1, qword[A2 - 0x80]);
        sub(A2, -8);
        punpckldq(xmm0, xmm1);
        pshufd(xmm6, xmm0, 0xd8);
        pmovsxbw(xmm5, xmm6);
        movhlps(xmm6, xmm6);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[9]);
        test(M, 0x4);
        jle(labels[10], T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        movd(xmm1, dword[A2 - 0x80]);
        sub(A2, -4);
        punpckldq(xmm0, xmm1);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[10]);
        test(M, 0x2);
        jle(labels[11], T_NEAR);
        mov(ax, word[A1 - 0x80]);
        sub(A1, -2);
        pinsrw(xmm0, eax, 0x0);
        mov(ax, word[A2 - 0x80]);
        sub(A2, -2);
        pinsrw(xmm0, eax, 0x1);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[11]);
        test(M, 0x1);
        jle(labels[12], T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        mov(byte[B - 0x80], al);
        mov(al, byte[A2 - 0x80]);
        pinsrb(xmm0, eax, 0x1);
        mov(byte[B - 0x7f], al);
        sub(B, -2);
        pmovsxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        align(4);

        L(labels[12]);
        mov(A1, qword[ARG_BIAS]);
        movq(qword[A1], xmm7);
        add(qword[ARG_BIAS], 0x8);
        sub(N, 0x2);
        cmp(N, 0x2);
        jge(labels[6], T_NEAR);
        align(4);

        L(labels[13]);
        cmp(N, 0x1);
        jl(labels[21], T_NEAR);
        align(4);

        L(labels[14]);
        mov(A1, A);
        add(A, LDA);
        pxor(xmm7, xmm7);
        mov(I, M);
        sar(I, 0x4);
        jle(labels[16], T_NEAR);
        align(4);

        L(labels[15]);
        movdqu(xmm0, xword[A1 - 0x80]);
        sub(A1, -16);
        pmovsxbw(xmm5, xmm0);
        movhlps(xmm6, xmm0);
        pmovsxbw(xmm6, xmm6);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movdqu(xword[B - 0x80], xmm0);
        sub(B, -16);
        dec(I);
        jg(labels[15], T_NEAR);
        align(4);

        L(labels[16]);
        test(M, 0x8);
        jle(labels[17], T_NEAR);
        movq(xmm0, qword[A1 - 0x80]);
        sub(A1, -8);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm6);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movq(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[17]);
        test(M, 0x4);
        jle(labels[18], T_NEAR);
        movd(xmm0, dword[A1 - 0x80]);
        sub(A1, -4);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        movd(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[18]);
        test(M, 0x2);
        jle(labels[19], T_NEAR);
        mov(ax, word[A1 - 0x80]);
        pinsrw(xmm0, eax, 0x0);
        pmovsxbw(xmm5, xmm0);
        phaddw(xmm5, xmm5);
        pmovsxwd(xmm5, xmm5);
        paddd(xmm7, xmm5);
        mov(word[B - 0x80], ax);
        sub(A1, -2);
        sub(B, -2);
        align(4);

        L(labels[19]);
        test(M, 0x1);
        jle(labels[20], T_NEAR);
        mov(al, byte[A1 - 0x80]);
        pinsrb(xmm0, eax, 0x0);
        pmovsxbd(xmm5, xmm0);
        paddd(xmm7, xmm5);
        mov(byte[B - 0x80], al);
        sub(B, -1);
        align(4);

        L(labels[20]);
        mov(A1, qword[ARG_BIAS]);
        movd(dword[A1], xmm7);
        add(qword[ARG_BIAS], 0x4);
        sub(N, 0x1);
        cmp(N, 0x1);
        jge(labels[14], T_NEAR);
        align(4);

        L(labels[21]);
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
#undef ARG_BIAS
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
