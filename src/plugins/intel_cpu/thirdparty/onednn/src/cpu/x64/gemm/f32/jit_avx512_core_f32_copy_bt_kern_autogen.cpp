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

jit_avx512_core_f32_copy_bt_kern::jit_avx512_core_f32_copy_bt_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_avx512_core_f32_copy_bt_kern::generate() {

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
#define A1 rsi
#define A2 r10
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
        vbroadcastss(zmm6, dword[ALPHA]);
        vpcmpeqb(xmm3, xmm3, xmm3);
        vpsrld(xmm3, xmm3, 0x17);
        vpslld(xmm3, xmm3, 0x19);
        vpsrld(xmm3, xmm3, 0x2);
        vpcmpeqb(xmm4, xmm4, xmm4);
        vpslld(xmm4, xmm4, 0x1f);
        vbroadcastss(zmm4, xmm4);
        vucomiss(xmm6, xmm3);
        jne(labels[26], T_NEAR);
        cmp(N, 0x8);
        jl(labels[7], T_NEAR);
        align(4);

        L(labels[29]);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[70], T_NEAR);
        align(4);

        L(labels[36]);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vmovups(yword[B - 0x20], ymm0);
        vmovups(ymm0, yword[A2 - 0x80]);
        vmovups(yword[B], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 1 - 0x80]);
        vmovups(yword[B + 0x20], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 2 - 0x80]);
        vmovups(yword[B + 0x40], ymm0);
        vmovups(ymm0, yword[A2 + LDA3 * 1 - 0x80]);
        vmovups(yword[B + 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -256);
        dec(I);
        jg(labels[36], T_NEAR);
        align(4);

        L(labels[70]);
        test(M, 0x4);
        jle(labels[4], T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vmovups(yword[B - 0x20], ymm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -128);
        align(4);

        L(labels[4]);
        test(M, 0x2);
        jle(labels[5], T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmovups(yword[B - 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(labels[5]);
        test(M, 0x1);
        jle(labels[6], T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmovups(yword[B - 0x80], ymm0);
        sub(B, -32);
        align(4);

        L(labels[6]);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(labels[29], T_NEAR);
        align(4);

        L(labels[7]);
        cmp(N, 0x4);
        jl(labels[13], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[9], T_NEAR);
        align(4);

        L(labels[8]);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmovups(xword[B - 0x70], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        vmovups(xword[B - 0x60], xmm0);
        vmovups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        vmovups(xword[B - 0x50], xmm0);
        vmovups(xmm0, xword[A2 - 0x80]);
        vmovups(xword[B - 0x40], xmm0);
        vmovups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        vmovups(xword[B - 0x30], xmm0);
        vmovups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        vmovups(xword[B - 0x20], xmm0);
        vmovups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        vmovups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -128);
        dec(I);
        jg(labels[8], T_NEAR);
        align(4);

        L(labels[9]);
        test(M, 0x4);
        jle(labels[10], T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmovups(xword[B - 0x70], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        vmovups(xword[B - 0x60], xmm0);
        vmovups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        vmovups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -64);
        align(4);

        L(labels[10]);
        test(M, 0x2);
        jle(labels[11], T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmovups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(labels[11]);
        test(M, 0x1);
        jle(labels[12], T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmovups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[12]);
        sub(N, 0x4);
        align(4);

        L(labels[13]);
        cmp(N, 0x2);
        jl(labels[19], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[15], T_NEAR);
        align(4);

        L(labels[14]);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmovlps(qword[B - 0x78], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        vmovlps(qword[B - 0x70], xmm0);
        vmovsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        vmovlps(qword[B - 0x68], xmm0);
        vmovsd(xmm0, qword[A2 - 0x80]);
        vmovlps(qword[B - 0x60], xmm0);
        vmovsd(xmm0, qword[A2 + LDA * 1 - 0x80]);
        vmovlps(qword[B - 0x58], xmm0);
        vmovsd(xmm0, qword[A2 + LDA * 2 - 0x80]);
        vmovlps(qword[B - 0x50], xmm0);
        vmovsd(xmm0, qword[A2 + LDA3 * 1 - 0x80]);
        vmovlps(qword[B - 0x48], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -64);
        dec(I);
        jg(labels[14], T_NEAR);
        align(4);

        L(labels[15]);
        test(M, 0x4);
        jle(labels[16], T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmovlps(qword[B - 0x78], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        vmovlps(qword[B - 0x70], xmm0);
        vmovsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        vmovlps(qword[B - 0x68], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -32);
        align(4);

        L(labels[16]);
        test(M, 0x2);
        jle(labels[17], T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmovlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(labels[17]);
        test(M, 0x1);
        jle(labels[18], T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmovlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[18]);
        sub(N, 0x2);
        align(4);

        L(labels[19]);
        cmp(N, 0x1);
        jl(labels[25], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[21], T_NEAR);
        align(4);

        L(labels[20]);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmovss(dword[B - 0x7c], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        vmovss(dword[B - 0x78], xmm0);
        vmovss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        vmovss(dword[B - 0x74], xmm0);
        vmovss(xmm0, dword[A2 - 0x80]);
        vmovss(dword[B - 0x70], xmm0);
        vmovss(xmm0, dword[A2 + LDA * 1 - 0x80]);
        vmovss(dword[B - 0x6c], xmm0);
        vmovss(xmm0, dword[A2 + LDA * 2 - 0x80]);
        vmovss(dword[B - 0x68], xmm0);
        vmovss(xmm0, dword[A2 + LDA3 * 1 - 0x80]);
        vmovss(dword[B - 0x64], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -32);
        dec(I);
        jg(labels[20], T_NEAR);
        align(4);

        L(labels[21]);
        test(M, 0x4);
        jle(labels[22], T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmovss(dword[B - 0x7c], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        vmovss(dword[B - 0x78], xmm0);
        vmovss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        vmovss(dword[B - 0x74], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -16);
        align(4);

        L(labels[22]);
        test(M, 0x2);
        jle(labels[23], T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmovss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(labels[23]);
        test(M, 0x1);
        jle(labels[24], T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmovss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[24]);
        sub(N, 0x1);
        align(4);

        L(labels[25]);
        jmp(labels[3], T_NEAR);
        align(4);

        L(labels[26]);
        vxorps(xmm3, xmm3, xmm4);
        vucomiss(xmm6, xmm3);
        jne(labels[54], T_NEAR);
        vmovaps(zmm6, zmm4);
        cmp(N, 0x8);
        jl(labels[34], T_NEAR);
        align(4);

        L(labels[27]);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[30], T_NEAR);
        align(4);

        L(labels[28]);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        vmovups(ymm0, yword[A2 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x20], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 2 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x40], ymm0);
        vmovups(ymm0, yword[A2 + LDA3 * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -256);
        dec(I);
        jg(labels[28], T_NEAR);
        align(4);

        L(labels[30]);
        test(M, 0x4);
        jle(labels[31], T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -128);
        align(4);

        L(labels[31]);
        test(M, 0x2);
        jle(labels[32], T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(labels[32]);
        test(M, 0x1);
        jle(labels[33], T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        sub(B, -32);
        align(4);

        L(labels[33]);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(labels[27], T_NEAR);
        align(4);

        L(labels[34]);
        cmp(N, 0x4);
        jl(labels[41], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[37], T_NEAR);
        align(4);

        L(labels[35]);
        vmovups(xmm0, xword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x60], xmm0);
        vmovups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x50], xmm0);
        vmovups(xmm0, xword[A2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x40], xmm0);
        vmovups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x30], xmm0);
        vmovups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x20], xmm0);
        vmovups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -128);
        dec(I);
        jg(labels[35], T_NEAR);
        align(4);

        L(labels[37]);
        test(M, 0x4);
        jle(labels[38], T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x60], xmm0);
        vmovups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -64);
        align(4);

        L(labels[38]);
        test(M, 0x2);
        jle(labels[39], T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(labels[39]);
        test(M, 0x1);
        jle(labels[40], T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[40]);
        sub(N, 0x4);
        align(4);

        L(labels[41]);
        cmp(N, 0x2);
        jl(labels[47], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[43], T_NEAR);
        align(4);

        L(labels[42]);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x70], xmm0);
        vmovsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x68], xmm0);
        vmovsd(xmm0, qword[A2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x60], xmm0);
        vmovsd(xmm0, qword[A2 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x58], xmm0);
        vmovsd(xmm0, qword[A2 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x50], xmm0);
        vmovsd(xmm0, qword[A2 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x48], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -64);
        dec(I);
        jg(labels[42], T_NEAR);
        align(4);

        L(labels[43]);
        test(M, 0x4);
        jle(labels[44], T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x70], xmm0);
        vmovsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x68], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -32);
        align(4);

        L(labels[44]);
        test(M, 0x2);
        jle(labels[45], T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(labels[45]);
        test(M, 0x1);
        jle(labels[46], T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[46]);
        sub(N, 0x2);
        align(4);

        L(labels[47]);
        cmp(N, 0x1);
        jl(labels[53], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[49], T_NEAR);
        align(4);

        L(labels[48]);
        vmovss(xmm0, dword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x78], xmm0);
        vmovss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x74], xmm0);
        vmovss(xmm0, dword[A2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x70], xmm0);
        vmovss(xmm0, dword[A2 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x6c], xmm0);
        vmovss(xmm0, dword[A2 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x68], xmm0);
        vmovss(xmm0, dword[A2 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x64], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -32);
        dec(I);
        jg(labels[48], T_NEAR);
        align(4);

        L(labels[49]);
        test(M, 0x4);
        jle(labels[50], T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x78], xmm0);
        vmovss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x74], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -16);
        align(4);

        L(labels[50]);
        test(M, 0x2);
        jle(labels[51], T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(labels[51]);
        test(M, 0x1);
        jle(labels[52], T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[52]);
        sub(N, 0x1);
        align(4);

        L(labels[53]);
        jmp(labels[3], T_NEAR);
        align(4);

        L(labels[54]);
        cmp(N, 0x8);
        jl(labels[61], T_NEAR);
        align(4);

        L(labels[55]);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x20);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[57], T_NEAR);
        align(4);

        L(labels[56]);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        vmovups(ymm0, yword[A2 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x20], ymm0);
        vmovups(ymm0, yword[A2 + LDA * 2 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x40], ymm0);
        vmovups(ymm0, yword[A2 + LDA3 * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B + 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -256);
        dec(I);
        jg(labels[56], T_NEAR);
        align(4);

        L(labels[57]);
        test(M, 0x4);
        jle(labels[58], T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 2 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x40], ymm0);
        vmovups(ymm0, yword[A1 + LDA3 * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x20], ymm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -128);
        align(4);

        L(labels[58]);
        test(M, 0x2);
        jle(labels[59], T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(ymm0, yword[A1 + LDA * 1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -64);
        align(4);

        L(labels[59]);
        test(M, 0x1);
        jle(labels[60], T_NEAR);
        vmovups(ymm0, yword[A1 - 0x80]);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        sub(B, -32);
        align(4);

        L(labels[60]);
        sub(N, 0x8);
        cmp(N, 0x8);
        jge(labels[55], T_NEAR);
        align(4);

        L(labels[61]);
        cmp(N, 0x4);
        jl(labels[67], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x10);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[63], T_NEAR);
        align(4);

        L(labels[62]);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x60], xmm0);
        vmovups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x50], xmm0);
        vmovups(xmm0, xword[A2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x40], xmm0);
        vmovups(xmm0, xword[A2 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x30], xmm0);
        vmovups(xmm0, xword[A2 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x20], xmm0);
        vmovups(xmm0, xword[A2 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x10], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -128);
        dec(I);
        jg(labels[62], T_NEAR);
        align(4);

        L(labels[63]);
        test(M, 0x4);
        jle(labels[64], T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x60], xmm0);
        vmovups(xmm0, xword[A1 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x50], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -64);
        align(4);

        L(labels[64]);
        test(M, 0x2);
        jle(labels[65], T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xmm0, xword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x70], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -32);
        align(4);

        L(labels[65]);
        test(M, 0x1);
        jle(labels[66], T_NEAR);
        vmovups(xmm0, xword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        sub(B, -16);
        align(4);

        L(labels[66]);
        sub(N, 0x4);
        align(4);

        L(labels[67]);
        cmp(N, 0x2);
        jl(labels[74], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x8);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[69], T_NEAR);
        align(4);

        L(labels[68]);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x70], xmm0);
        vmovsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x68], xmm0);
        vmovsd(xmm0, qword[A2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x60], xmm0);
        vmovsd(xmm0, qword[A2 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x58], xmm0);
        vmovsd(xmm0, qword[A2 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x50], xmm0);
        vmovsd(xmm0, qword[A2 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x48], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -64);
        dec(I);
        jg(labels[68], T_NEAR);
        align(4);

        L(labels[69]);
        test(M, 0x4);
        jle(labels[71], T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x70], xmm0);
        vmovsd(xmm0, qword[A1 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x68], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -32);
        align(4);

        L(labels[71]);
        test(M, 0x2);
        jle(labels[72], T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovsd(xmm0, qword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x78], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -16);
        align(4);

        L(labels[72]);
        test(M, 0x1);
        jle(labels[73], T_NEAR);
        vmovsd(xmm0, qword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        sub(B, -8);
        align(4);

        L(labels[73]);
        sub(N, 0x2);
        align(4);

        L(labels[74]);
        cmp(N, 0x1);
        jl(labels[3], T_NEAR);
        mov(A1, A);
        lea(A2, ptr[A1 + LDA * 4]);
        add(A, 0x4);
        mov(I, M);
        sar(I, 0x3);
        jle(labels[76], T_NEAR);
        align(4);

        L(labels[75]);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x78], xmm0);
        vmovss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x74], xmm0);
        vmovss(xmm0, dword[A2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x70], xmm0);
        vmovss(xmm0, dword[A2 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x6c], xmm0);
        vmovss(xmm0, dword[A2 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x68], xmm0);
        vmovss(xmm0, dword[A2 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x64], xmm0);
        lea(A1, ptr[A1 + LDA * 8]);
        lea(A2, ptr[A2 + LDA * 8]);
        sub(B, -32);
        dec(I);
        jg(labels[75], T_NEAR);
        align(4);

        L(labels[76]);
        test(M, 0x4);
        jle(labels[0], T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 2 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x78], xmm0);
        vmovss(xmm0, dword[A1 + LDA3 * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x74], xmm0);
        lea(A1, ptr[A1 + LDA * 4]);
        sub(B, -16);
        align(4);

        L(labels[0]);
        test(M, 0x2);
        jle(labels[1], T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(xmm0, dword[A1 + LDA * 1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x7c], xmm0);
        lea(A1, ptr[A1 + LDA * 2]);
        sub(B, -8);
        align(4);

        L(labels[1]);
        test(M, 0x1);
        jle(labels[2], T_NEAR);
        vmovss(xmm0, dword[A1 - 0x80]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        sub(B, -4);
        align(4);

        L(labels[2]);
        sub(N, 0x1);
        align(4);

        L(labels[3]);

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
