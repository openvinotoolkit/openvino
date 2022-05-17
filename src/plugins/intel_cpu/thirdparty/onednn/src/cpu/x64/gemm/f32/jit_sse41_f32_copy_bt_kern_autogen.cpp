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

jit_sse41_f32_copy_bt_kern::jit_sse41_f32_copy_bt_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_sse41_f32_copy_bt_kern::generate() {

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
        std::vector<Xbyak::Label> labels(59);
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
        jne(labels[16], T_NEAR);
        cmp(N, 0x4);
        jl(labels[3], T_NEAR);
        align(4);

        L(labels[25]);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[58], T_NEAR);
        align(4);

        L(labels[32]);
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
        jg(labels[32], T_NEAR);
        align(4);

        L(labels[58]);
        test(M, 0x4);
        jle(labels[0], T_NEAR);
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

        L(labels[0]);
        test(M, 0x2);
        jle(labels[1], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        movups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(labels[1]);
        test(M, 0x1);
        jle(labels[2], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        movups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[2]);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(labels[25], T_NEAR);
        align(4);

        L(labels[3]);
        cmp(N, 0x2);
        jl(labels[9], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[5], T_NEAR);
        align(4);

        L(labels[4]);
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
        jg(labels[4], T_NEAR);
        align(4);

        L(labels[5]);
        test(M, 0x4);
        jle(labels[6], T_NEAR);
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

        L(labels[6]);
        test(M, 0x2);
        jle(labels[7], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        movlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(labels[7]);
        test(M, 0x1);
        jle(labels[8], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        movlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[8]);
        sub(N, 0x2);
        align(4);

        L(labels[9]);
        cmp(N, 0x1);
        jl(labels[15], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[11], T_NEAR);
        align(4);

        L(labels[10]);
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
        jg(labels[10], T_NEAR);
        align(4);

        L(labels[11]);
        test(M, 0x4);
        jle(labels[12], T_NEAR);
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

        L(labels[12]);
        test(M, 0x2);
        jle(labels[13], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        movss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(labels[13]);
        test(M, 0x1);
        jle(labels[14], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        movss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[14]);
        sub(N, 0x1);
        align(4);

        L(labels[15]);
        jmp(labels[57], T_NEAR);
        align(4);

        L(labels[16]);
        xorps(xmm3, xmm4);
        ucomiss(xmm6, xmm3);
        jne(labels[38], T_NEAR);
        movaps(xmm6, xmm4);
        cmp(N, 0x4);
        jl(labels[23], T_NEAR);
        align(4);

        L(labels[17]);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[19], T_NEAR);
        align(4);

        L(labels[18]);
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
        jg(labels[18], T_NEAR);
        align(4);

        L(labels[19]);
        test(M, 0x4);
        jle(labels[20], T_NEAR);
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

        L(labels[20]);
        test(M, 0x2);
        jle(labels[21], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(labels[21]);
        test(M, 0x1);
        jle(labels[22], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[22]);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(labels[17], T_NEAR);
        align(4);

        L(labels[23]);
        cmp(N, 0x2);
        jl(labels[30], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[26], T_NEAR);
        align(4);

        L(labels[24]);
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
        jg(labels[24], T_NEAR);
        align(4);

        L(labels[26]);
        test(M, 0x4);
        jle(labels[27], T_NEAR);
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

        L(labels[27]);
        test(M, 0x2);
        jle(labels[28], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(labels[28]);
        test(M, 0x1);
        jle(labels[29], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[29]);
        sub(N, 0x2);
        align(4);

        L(labels[30]);
        cmp(N, 0x1);
        jl(labels[37], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[33], T_NEAR);
        align(4);

        L(labels[31]);
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
        jg(labels[31], T_NEAR);
        align(4);

        L(labels[33]);
        test(M, 0x4);
        jle(labels[34], T_NEAR);
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

        L(labels[34]);
        test(M, 0x2);
        jle(labels[35], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(labels[35]);
        test(M, 0x1);
        jle(labels[36], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        xorps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[36]);
        sub(N, 0x1);
        align(4);

        L(labels[37]);
        jmp(labels[57], T_NEAR);
        align(4);

        L(labels[38]);
        cmp(N, 0x4);
        jl(labels[45], T_NEAR);
        align(4);

        L(labels[39]);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[41], T_NEAR);
        align(4);

        L(labels[40]);
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
        jg(labels[40], T_NEAR);
        align(4);

        L(labels[41]);
        test(M, 0x4);
        jle(labels[42], T_NEAR);
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

        L(labels[42]);
        test(M, 0x2);
        jle(labels[43], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        movups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(labels[43]);
        test(M, 0x1);
        jle(labels[44], T_NEAR);
        movups(xmm0, xword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[44]);
        sub(N, 0x4);
        cmp(N, 0x4);
        jge(labels[39], T_NEAR);
        align(4);

        L(labels[45]);
        cmp(N, 0x2);
        jl(labels[51], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[47], T_NEAR);
        align(4);

        L(labels[46]);
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
        jg(labels[46], T_NEAR);
        align(4);

        L(labels[47]);
        test(M, 0x4);
        jle(labels[48], T_NEAR);
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

        L(labels[48]);
        test(M, 0x2);
        jle(labels[49], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        movsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(labels[49]);
        test(M, 0x1);
        jle(labels[50], T_NEAR);
        movsd(xmm0, qword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[50]);
        sub(N, 0x2);
        align(4);

        L(labels[51]);
        cmp(N, 0x1);
        jl(labels[57], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[53], T_NEAR);
        align(4);

        L(labels[52]);
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
        jg(labels[52], T_NEAR);
        align(4);

        L(labels[53]);
        test(M, 0x4);
        jle(labels[54], T_NEAR);
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

        L(labels[54]);
        test(M, 0x2);
        jle(labels[55], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        movss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(labels[55]);
        test(M, 0x1);
        jle(labels[56], T_NEAR);
        movss(xmm0, dword[A1 - 0x80]);
        mulps(xmm0, xmm6);
        movss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[56]);
        sub(N, 0x1);
        align(4);

        L(labels[57]);

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
