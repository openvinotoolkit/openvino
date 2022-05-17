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

jit_sse41_f32_copy_an_kern::jit_sse41_f32_copy_an_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_sse41_f32_copy_an_kern::generate() {

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
        std::vector<Xbyak::Label> labels(77);
        preamble();
#ifdef _WIN32
        auto stacksize = get_size_of_abi_save_regs();
        mov(ALPHA, ptr[ARG_ALPHA]);
        mov(B, ptr[ARG_B]);
#endif

        mov(M, qword[M]);
        mov(N, qword[N]);
        mov(LDA, qword[LDA]);
        sub(A, -128);
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
        jne(labels[30], T_NEAR);
        cmp(N, 0x8);
        jl(labels[10], T_NEAR);
        align(4);

        L(labels[29]);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[6], T_NEAR);
        align(4);

        L(labels[34]);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x70]);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x70]);
        movups(xword[B - 0x10], xmm0);
        movups(xmm0, xword[A2 - 0x80]);
        movups(xword[B], xmm0);
        movups(xmm0, xword[A2 - 0x70]);
        movups(xword[B + 0x10], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        movups(xword[B + 0x20], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x70]);
        movups(xword[B + 0x30], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        movups(xword[B + 0x40], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x70]);
        movups(xword[B + 0x50], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        movups(xword[B + 0x60], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x70]);
        movups(xword[B + 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -256);
        dec(I);
        jg(labels[34], T_NEAR);
        align(4);

        L(labels[6]);
        test(M, 0x4);
        jle(labels[7], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x70]);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x70]);
        movups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -128);
        align(4);

        L(labels[7]);
        test(M, 0x2);
        jle(labels[8], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        movups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(labels[8]);
        test(M, 0x1);
        jle(labels[9], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        movups(xword[B - 0x70], xmm0);
        sub(B, -32);
        align(4);

        L(labels[9]);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(labels[29], T_NEAR);
        align(4);

        L(labels[10]);
        cmp(N, 0x4);
        jl(labels[16], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[12], T_NEAR);
        align(4);

        L(labels[11]);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A2 - 0x80]);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        movups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -128);
        dec(I);
        jg(labels[11], T_NEAR);
        align(4);

        L(labels[12]);
        test(M, 0x4);
        jle(labels[13], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        movups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -64);
        align(4);

        L(labels[13]);
        test(M, 0x2);
        jle(labels[14], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(labels[14]);
        test(M, 0x1);
        jle(labels[15], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[15]);
        sub(N, 0x4);
        align(4);

        L(labels[16]);
        cmp(N, 0x2);
        jl(labels[22], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[18], T_NEAR);
        align(4);

        L(labels[17]);
        movsd(xmm0, qword[A1 - 0x80]);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        movlps(qword[B - 0x78], xmm0);
        movsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        movlps(qword[B - 0x70], xmm0);
        movsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        movlps(qword[B - 0x68], xmm0);
        movsd(xmm0, qword[A2 - 0x80]);
        movlps(qword[B - 0x60], xmm0);
        movsd(xmm0, qword[A2 + LDA * 1 - 0x80]);
        movlps(qword[B - 0x58], xmm0);
        movsd(xmm0, qword[A2 + LDA * 2 - 0x80]);
        movlps(qword[B - 0x50], xmm0);
        movsd(xmm0, qword[A2 + LDA3 * 1 - 0x80]);
        movlps(qword[B - 0x48], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -64);
        dec(I);
        jg(labels[17], T_NEAR);
        align(4);

        L(labels[18]);
        test(M, 0x4);
        jle(labels[19], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        movlps(qword[B - 0x78], xmm0);
        movsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        movlps(qword[B - 0x70], xmm0);
        movsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        movlps(qword[B - 0x68], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -32);
        align(4);

        L(labels[19]);
        test(M, 0x2);
        jle(labels[20], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        movlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(labels[20]);
        test(M, 0x1);
        jle(labels[21], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        movlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[21]);
        sub(N, 0x2);
        align(4);

        L(labels[22]);
        cmp(N, 0x1);
        jl(labels[28], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[24], T_NEAR);
        align(4);

        L(labels[23]);
        movss(xmm0, dword[A1 - 0x80]);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        movss(dword[B - 0x7c], xmm0);
        movss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        movss(dword[B - 0x78], xmm0);
        movss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        movss(dword[B - 0x74], xmm0);
        movss(xmm0, dword[A2 - 0x80]);
        movss(dword[B - 0x70], xmm0);
        movss(xmm0, dword[A2 + LDA * 1 - 0x80]);
        movss(dword[B - 0x6c], xmm0);
        movss(xmm0, dword[A2 + LDA * 2 - 0x80]);
        movss(dword[B - 0x68], xmm0);
        movss(xmm0, dword[A2 + LDA3 * 1 - 0x80]);
        movss(dword[B - 0x64], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -32);
        dec(I);
        jg(labels[23], T_NEAR);
        align(4);

        L(labels[24]);
        test(M, 0x4);
        jle(labels[25], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        movss(dword[B - 0x7c], xmm0);
        movss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        movss(dword[B - 0x78], xmm0);
        movss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        movss(dword[B - 0x74], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -16);
        align(4);

        L(labels[25]);
        test(M, 0x2);
        jle(labels[26], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        movss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(labels[26]);
        test(M, 0x1);
        jle(labels[27], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        movss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[27]);
        sub(N, 0x1);
        align(4);

        L(labels[28]);
        jmp(labels[5], T_NEAR);
        align(4);

        L(labels[30]);
        xorps(xmm3, xmm4);
        ucomiss(xmm6, xmm3);
        jne(labels[57], T_NEAR);
        movaps(xmm6, xmm4);
        cmp(N, 0x8);
        jl(labels[38], T_NEAR);
        align(4);

        L(labels[31]);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[33], T_NEAR);
        align(4);

        L(labels[32]);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x10], xmm0);
        movups(xmm0, xword[A2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B], xmm0);
        movups(xmm0, xword[A2 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x10], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x20], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x30], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x40], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x50], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x60], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B + 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -256);
        dec(I);
        jg(labels[32], T_NEAR);
        align(4);

        L(labels[33]);
        test(M, 0x4);
        jle(labels[35], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -128);
        align(4);

        L(labels[35]);
        test(M, 0x2);
        jle(labels[36], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(labels[36]);
        test(M, 0x1);
        jle(labels[37], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        sub(B, -32);
        align(4);

        L(labels[37]);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(labels[31], T_NEAR);
        align(4);

        L(labels[38]);
        cmp(N, 0x4);
        jl(labels[44], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[40], T_NEAR);
        align(4);

        L(labels[39]);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -128);
        dec(I);
        jg(labels[39], T_NEAR);
        align(4);

        L(labels[40]);
        test(M, 0x4);
        jle(labels[41], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -64);
        align(4);

        L(labels[41]);
        test(M, 0x2);
        jle(labels[42], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(labels[42]);
        test(M, 0x1);
        jle(labels[43], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[43]);
        sub(N, 0x4);
        align(4);

        L(labels[44]);
        cmp(N, 0x2);
        jl(labels[50], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[46], T_NEAR);
        align(4);

        L(labels[45]);
        movsd(xmm0, qword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        movsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x70], xmm0);
        movsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x68], xmm0);
        movsd(xmm0, qword[A2 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x60], xmm0);
        movsd(xmm0, qword[A2 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x58], xmm0);
        movsd(xmm0, qword[A2 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x50], xmm0);
        movsd(xmm0, qword[A2 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x48], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -64);
        dec(I);
        jg(labels[45], T_NEAR);
        align(4);

        L(labels[46]);
        test(M, 0x4);
        jle(labels[47], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        movsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x70], xmm0);
        movsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x68], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -32);
        align(4);

        L(labels[47]);
        test(M, 0x2);
        jle(labels[48], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(labels[48]);
        test(M, 0x1);
        jle(labels[49], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[49]);
        sub(N, 0x2);
        align(4);

        L(labels[50]);
        cmp(N, 0x1);
        jl(labels[56], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[52], T_NEAR);
        align(4);

        L(labels[51]);
        movss(xmm0, dword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        movss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x78], xmm0);
        movss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x74], xmm0);
        movss(xmm0, dword[A2 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x70], xmm0);
        movss(xmm0, dword[A2 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x6c], xmm0);
        movss(xmm0, dword[A2 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x68], xmm0);
        movss(xmm0, dword[A2 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x64], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -32);
        dec(I);
        jg(labels[51], T_NEAR);
        align(4);

        L(labels[52]);
        test(M, 0x4);
        jle(labels[53], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        movss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x78], xmm0);
        movss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x74], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -16);
        align(4);

        L(labels[53]);
        test(M, 0x2);
        jle(labels[54], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(labels[54]);
        test(M, 0x1);
        jle(labels[55], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[55]);
        sub(N, 0x1);
        align(4);

        L(labels[56]);
        jmp(labels[5], T_NEAR);
        align(4);

        L(labels[57]);
        cmp(N, 0x8);
        jl(labels[64], T_NEAR);
        align(4);

        L(labels[58]);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[60], T_NEAR);
        align(4);

        L(labels[59]);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x10], xmm0);
        movups(xmm0, xword[A2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B], xmm0);
        movups(xmm0, xword[A2 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x10], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x20], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x30], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x40], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x50], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x60], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B + 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -256);
        dec(I);
        jg(labels[59], T_NEAR);
        align(4);

        L(labels[60]);
        test(M, 0x4);
        jle(labels[61], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -128);
        align(4);

        L(labels[61]);
        test(M, 0x2);
        jle(labels[62], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(labels[62]);
        test(M, 0x1);
        jle(labels[63], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 - 0x70]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        sub(B, -32);
        align(4);

        L(labels[63]);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(labels[58], T_NEAR);
        align(4);

        L(labels[64]);
        cmp(N, 0x4);
        jl(labels[70], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[66], T_NEAR);
        align(4);

        L(labels[65]);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        movups(xmm0, xword[A2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x40], xmm0);
        movups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x30], xmm0);
        movups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x20], xmm0);
        movups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -128);
        dec(I);
        jg(labels[65], T_NEAR);
        align(4);

        L(labels[66]);
        test(M, 0x4);
        jle(labels[67], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        movups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x60], xmm0);
        movups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -64);
        align(4);

        L(labels[67]);
        test(M, 0x2);
        jle(labels[68], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(labels[68]);
        test(M, 0x1);
        jle(labels[69], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[69]);
        sub(N, 0x4);
        align(4);

        L(labels[70]);
        cmp(N, 0x2);
        jl(labels[76], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[72], T_NEAR);
        align(4);

        L(labels[71]);
        movsd(xmm0, qword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        movsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x70], xmm0);
        movsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x68], xmm0);
        movsd(xmm0, qword[A2 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x60], xmm0);
        movsd(xmm0, qword[A2 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x58], xmm0);
        movsd(xmm0, qword[A2 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x50], xmm0);
        movsd(xmm0, qword[A2 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x48], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -64);
        dec(I);
        jg(labels[71], T_NEAR);
        align(4);

        L(labels[72]);
        test(M, 0x4);
        jle(labels[73], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        movsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x70], xmm0);
        movsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x68], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -32);
        align(4);

        L(labels[73]);
        test(M, 0x2);
        jle(labels[74], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(labels[74]);
        test(M, 0x1);
        jle(labels[75], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[75]);
        sub(N, 0x2);
        align(4);

        L(labels[76]);
        cmp(N, 0x1);
        jl(labels[5], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[1], T_NEAR);
        align(4);

        L(labels[0]);
        movss(xmm0, dword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        movss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x78], xmm0);
        movss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x74], xmm0);
        movss(xmm0, dword[A2 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x70], xmm0);
        movss(xmm0, dword[A2 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x6c], xmm0);
        movss(xmm0, dword[A2 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x68], xmm0);
        movss(xmm0, dword[A2 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x64], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -32);
        dec(I);
        jg(labels[0], T_NEAR);
        align(4);

        L(labels[1]);
        test(M, 0x4);
        jle(labels[2], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        movss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x78], xmm0);
        movss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x74], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -16);
        align(4);

        L(labels[2]);
        test(M, 0x2);
        jle(labels[3], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(labels[3]);
        test(M, 0x1);
        jle(labels[4], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[4]);
        sub(N, 0x1);
        align(4);

        L(labels[5]);

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
