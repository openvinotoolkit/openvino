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

jit_avx_f32_copy_at_kern::jit_avx_f32_copy_at_kern()
    : jit_generator(nullptr, F32_COPY_KERNEL_CODE_SIZE) {}

void jit_avx_f32_copy_at_kern::generate() {

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
        std::vector<Xbyak::Label> labels(80);
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
        vbroadcastss(ymm6, dword[ALPHA]);
        vpcmpeqb(xmm3, xmm3, xmm3);
        vpsrld(xmm3, xmm3, 0x17);
        vpslld(xmm3, xmm3, 0x19);
        vpsrld(xmm3, xmm3, 0x2);
        vpcmpeqb(xmm4, xmm4, xmm4);
        vpslld(xmm4, xmm4, 0x1f);
        vperm2f128(ymm4, ymm4, ymm4, 0x20);
        vucomiss(xmm6, xmm3);
        jne(labels[29], T_NEAR);
        cmp(N, 0x10);
        jl(labels[52], T_NEAR);
        align(4);

        L(labels[45]);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x10);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[55], T_NEAR);
        align(4);

        L(labels[34]);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm4, xword[A1 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A1 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x40], ymm1);
        vmovups(yword[B], ymm2);
        vmovups(yword[B + 0x40], ymm3);
        vmovups(xmm0, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A2 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(yword[B - 0x20], ymm1);
        vmovups(yword[B + 0x20], ymm2);
        vmovups(yword[B + 0x60], ymm3);
        sub(A1, -16);
        sub(B, -256);
        dec(I);
        jg(labels[34], T_NEAR);
        align(4);

        L(labels[55]);
        test(M, 0x2);
        jle(labels[54], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x40], ymm1);
        vmovsd(xmm0, qword[A2]);
        vmovsd(xmm1, qword[A2 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A2 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(yword[B - 0x20], ymm1);
        sub(A1, -8);
        sub(B, -128);
        align(4);

        L(labels[54]);
        test(M, 0x1);
        jle(labels[53], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vmovups(yword[B - 0x80], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vmovups(yword[B - 0x60], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -64);
        align(4);

        L(labels[53]);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(labels[45], T_NEAR);
        align(4);

        L(labels[52]);
        cmp(N, 0x8);
        jl(labels[47], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[50], T_NEAR);
        align(4);

        L(labels[51]);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm4, xword[A1 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A1 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x60], ymm1);
        vmovups(yword[B - 0x40], ymm2);
        vmovups(yword[B - 0x20], ymm3);
        sub(A1, -16);
        sub(B, -128);
        dec(I);
        jg(labels[51], T_NEAR);
        align(4);

        L(labels[50]);
        test(M, 0x2);
        jle(labels[49], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x60], ymm1);
        sub(A1, -8);
        sub(B, -64);
        align(4);

        L(labels[49]);
        test(M, 0x1);
        jle(labels[48], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vmovups(yword[B - 0x80], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -32);
        align(4);

        L(labels[48]);
        sub(N, 0x8);
        align(4);

        L(labels[47]);
        cmp(N, 0x4);
        jl(labels[41], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[44], T_NEAR);
        align(4);

        L(labels[46]);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm1, xword[A1 + LDA * 1]);
        vmovups(xmm2, xword[A1 + LDA * 2]);
        vmovups(xmm3, xword[A1 + LDA3 * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm5, xmm0, xmm1);
        vunpcklps(xmm1, xmm2, xmm3);
        vunpckhps(xmm3, xmm2, xmm3);
        vunpcklpd(xmm0, xmm4, xmm1);
        vunpckhpd(xmm1, xmm4, xmm1);
        vunpcklpd(xmm2, xmm5, xmm3);
        vunpckhpd(xmm3, xmm5, xmm3);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xword[B - 0x70], xmm1);
        vmovups(xword[B - 0x60], xmm2);
        vmovups(xword[B - 0x50], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -16);
        sub(B, -64);
        dec(I);
        jg(labels[46], T_NEAR);
        align(4);

        L(labels[44]);
        test(M, 0x2);
        jle(labels[43], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm1, xmm0, xmm1);
        vunpcklpd(xmm0, xmm4, xmm1);
        vunpckhpd(xmm1, xmm4, xmm1);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xword[B - 0x70], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -8);
        sub(B, -32);
        align(4);

        L(labels[43]);
        test(M, 0x1);
        jle(labels[42], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vmovups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -4);
        sub(B, -16);
        align(4);

        L(labels[42]);
        sub(N, 0x4);
        align(4);

        L(labels[41]);
        cmp(N, 0x2);
        jl(labels[36], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[39], T_NEAR);
        align(4);

        L(labels[40]);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm1, xword[A1 + LDA * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm1, xmm0, xmm1);
        vmovaps(xmm0, xmm4);
        vmovlps(qword[B - 0x80], xmm0);
        vmovhps(qword[B - 0x78], xmm0);
        vmovlps(qword[B - 0x70], xmm1);
        vmovhps(qword[B - 0x68], xmm1);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -16);
        sub(B, -32);
        dec(I);
        jg(labels[40], T_NEAR);
        align(4);

        L(labels[39]);
        test(M, 0x2);
        jle(labels[38], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovlps(qword[B - 0x80], xmm0);
        vmovhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(labels[38]);
        test(M, 0x1);
        jle(labels[37], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(labels[37]);
        sub(N, 0x2);
        align(4);

        L(labels[36]);
        cmp(N, 0x1);
        jl(labels[30], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[33], T_NEAR);
        align(4);

        L(labels[35]);
        vmovups(xmm0, xword[A1]);
        vpshufd(xmm1, xmm0, 0x55);
        vpshufd(xmm2, xmm0, 0xaa);
        vpshufd(xmm3, xmm0, 0xff);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        vmovss(dword[B - 0x78], xmm2);
        vmovss(dword[B - 0x74], xmm3);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -16);
        sub(B, -16);
        dec(I);
        jg(labels[35], T_NEAR);
        align(4);

        L(labels[33]);
        test(M, 0x2);
        jle(labels[32], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vpshufd(xmm1, xmm0, 0x55);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(labels[32]);
        test(M, 0x1);
        jle(labels[31], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(labels[31]);
        sub(N, 0x1);
        align(4);

        L(labels[30]);
        jmp(labels[56], T_NEAR);
        align(4);

        L(labels[29]);
        vxorps(xmm3, xmm3, xmm4);
        vucomiss(xmm6, xmm3);
        jne(labels[2], T_NEAR);
        vmovaps(ymm6, ymm4);
        cmp(N, 0x10);
        jl(labels[23], T_NEAR);
        align(4);

        L(labels[28]);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x10);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[26], T_NEAR);
        align(4);

        L(labels[27]);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm4, xword[A1 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A1 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vxorps(ymm0, ymm6, ymm0);
        vxorps(ymm1, ymm6, ymm1);
        vxorps(ymm2, ymm6, ymm2);
        vxorps(ymm3, ymm6, ymm3);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x40], ymm1);
        vmovups(yword[B], ymm2);
        vmovups(yword[B + 0x40], ymm3);
        vmovups(xmm0, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A2 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vxorps(ymm0, ymm6, ymm0);
        vxorps(ymm1, ymm6, ymm1);
        vxorps(ymm2, ymm6, ymm2);
        vxorps(ymm3, ymm6, ymm3);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(yword[B - 0x20], ymm1);
        vmovups(yword[B + 0x20], ymm2);
        vmovups(yword[B + 0x60], ymm3);
        sub(A1, -16);
        sub(B, -256);
        dec(I);
        jg(labels[27], T_NEAR);
        align(4);

        L(labels[26]);
        test(M, 0x2);
        jle(labels[25], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vxorps(ymm0, ymm6, ymm0);
        vxorps(ymm1, ymm6, ymm1);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x40], ymm1);
        vmovsd(xmm0, qword[A2]);
        vmovsd(xmm1, qword[A2 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A2 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vxorps(ymm0, ymm6, ymm0);
        vxorps(ymm1, ymm6, ymm1);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(yword[B - 0x20], ymm1);
        sub(A1, -8);
        sub(B, -128);
        align(4);

        L(labels[25]);
        test(M, 0x1);
        jle(labels[24], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -64);
        align(4);

        L(labels[24]);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(labels[28], T_NEAR);
        align(4);

        L(labels[23]);
        cmp(N, 0x8);
        jl(labels[18], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[21], T_NEAR);
        align(4);

        L(labels[22]);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm4, xword[A1 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A1 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vxorps(ymm0, ymm6, ymm0);
        vxorps(ymm1, ymm6, ymm1);
        vxorps(ymm2, ymm6, ymm2);
        vxorps(ymm3, ymm6, ymm3);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x60], ymm1);
        vmovups(yword[B - 0x40], ymm2);
        vmovups(yword[B - 0x20], ymm3);
        sub(A1, -16);
        sub(B, -128);
        dec(I);
        jg(labels[22], T_NEAR);
        align(4);

        L(labels[21]);
        test(M, 0x2);
        jle(labels[20], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vxorps(ymm0, ymm6, ymm0);
        vxorps(ymm1, ymm6, ymm1);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x60], ymm1);
        sub(A1, -8);
        sub(B, -64);
        align(4);

        L(labels[20]);
        test(M, 0x1);
        jle(labels[19], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vxorps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -32);
        align(4);

        L(labels[19]);
        sub(N, 0x8);
        align(4);

        L(labels[18]);
        cmp(N, 0x4);
        jl(labels[13], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[16], T_NEAR);
        align(4);

        L(labels[17]);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm1, xword[A1 + LDA * 1]);
        vmovups(xmm2, xword[A1 + LDA * 2]);
        vmovups(xmm3, xword[A1 + LDA3 * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm5, xmm0, xmm1);
        vunpcklps(xmm1, xmm2, xmm3);
        vunpckhps(xmm3, xmm2, xmm3);
        vunpcklpd(xmm0, xmm4, xmm1);
        vunpckhpd(xmm1, xmm4, xmm1);
        vunpcklpd(xmm2, xmm5, xmm3);
        vunpckhpd(xmm3, xmm5, xmm3);
        vxorps(xmm0, xmm6, xmm0);
        vxorps(xmm1, xmm6, xmm1);
        vxorps(xmm2, xmm6, xmm2);
        vxorps(xmm3, xmm6, xmm3);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xword[B - 0x70], xmm1);
        vmovups(xword[B - 0x60], xmm2);
        vmovups(xword[B - 0x50], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -16);
        sub(B, -64);
        dec(I);
        jg(labels[17], T_NEAR);
        align(4);

        L(labels[16]);
        test(M, 0x2);
        jle(labels[15], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm1, xmm0, xmm1);
        vunpcklpd(xmm0, xmm4, xmm1);
        vunpckhpd(xmm1, xmm4, xmm1);
        vxorps(xmm0, xmm6, xmm0);
        vxorps(xmm1, xmm6, xmm1);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xword[B - 0x70], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -8);
        sub(B, -32);
        align(4);

        L(labels[15]);
        test(M, 0x1);
        jle(labels[14], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vxorps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -4);
        sub(B, -16);
        align(4);

        L(labels[14]);
        sub(N, 0x4);
        align(4);

        L(labels[13]);
        cmp(N, 0x2);
        jl(labels[8], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[11], T_NEAR);
        align(4);

        L(labels[12]);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm1, xword[A1 + LDA * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm1, xmm0, xmm1);
        vmovaps(xmm0, xmm4);
        vxorps(xmm0, xmm6, xmm0);
        vxorps(xmm1, xmm6, xmm1);
        vmovlps(qword[B - 0x80], xmm0);
        vmovhps(qword[B - 0x78], xmm0);
        vmovlps(qword[B - 0x70], xmm1);
        vmovhps(qword[B - 0x68], xmm1);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -16);
        sub(B, -32);
        dec(I);
        jg(labels[12], T_NEAR);
        align(4);

        L(labels[11]);
        test(M, 0x2);
        jle(labels[10], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(labels[10]);
        test(M, 0x1);
        jle(labels[9], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vxorps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(labels[9]);
        sub(N, 0x2);
        align(4);

        L(labels[8]);
        cmp(N, 0x1);
        jl(labels[3], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[6], T_NEAR);
        align(4);

        L(labels[7]);
        vmovups(xmm0, xword[A1]);
        vxorps(xmm0, xmm6, xmm0);
        vpshufd(xmm1, xmm0, 0x55);
        vpshufd(xmm2, xmm0, 0xaa);
        vpshufd(xmm3, xmm0, 0xff);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        vmovss(dword[B - 0x78], xmm2);
        vmovss(dword[B - 0x74], xmm3);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -16);
        sub(B, -16);
        dec(I);
        jg(labels[7], T_NEAR);
        align(4);

        L(labels[6]);
        test(M, 0x2);
        jle(labels[5], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vxorps(xmm0, xmm6, xmm0);
        vpshufd(xmm1, xmm0, 0x55);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(labels[5]);
        test(M, 0x1);
        jle(labels[4], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vxorps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(labels[4]);
        sub(N, 0x1);
        align(4);

        L(labels[3]);
        jmp(labels[56], T_NEAR);
        align(4);

        L(labels[2]);
        cmp(N, 0x10);
        jl(labels[76], T_NEAR);
        align(4);

        L(labels[1]);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x10);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[79], T_NEAR);
        align(4);

        L(labels[0]);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm4, xword[A1 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A1 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vmulps(ymm0, ymm6, ymm0);
        vmulps(ymm1, ymm6, ymm1);
        vmulps(ymm2, ymm6, ymm2);
        vmulps(ymm3, ymm6, ymm3);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x40], ymm1);
        vmovups(yword[B], ymm2);
        vmovups(yword[B + 0x40], ymm3);
        vmovups(xmm0, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A2 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vmulps(ymm0, ymm6, ymm0);
        vmulps(ymm1, ymm6, ymm1);
        vmulps(ymm2, ymm6, ymm2);
        vmulps(ymm3, ymm6, ymm3);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(yword[B - 0x20], ymm1);
        vmovups(yword[B + 0x20], ymm2);
        vmovups(yword[B + 0x60], ymm3);
        sub(A1, -16);
        sub(B, -256);
        dec(I);
        jg(labels[0], T_NEAR);
        align(4);

        L(labels[79]);
        test(M, 0x2);
        jle(labels[78], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vmulps(ymm0, ymm6, ymm0);
        vmulps(ymm1, ymm6, ymm1);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x40], ymm1);
        vmovsd(xmm0, qword[A2]);
        vmovsd(xmm1, qword[A2 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A2 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vmulps(ymm0, ymm6, ymm0);
        vmulps(ymm1, ymm6, ymm1);
        vmovups(yword[B - 0x60], ymm0);
        vmovups(yword[B - 0x20], ymm1);
        sub(A1, -8);
        sub(B, -128);
        align(4);

        L(labels[78]);
        test(M, 0x1);
        jle(labels[77], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A2 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x60], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -64);
        align(4);

        L(labels[77]);
        sub(N, 0x10);
        cmp(N, 0x10);
        jge(labels[1], T_NEAR);
        align(4);

        L(labels[76]);
        cmp(N, 0x8);
        jl(labels[71], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x8);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[74], T_NEAR);
        align(4);

        L(labels[75]);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm4, xword[A1 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm4, 0x20);
        lea(A2, ptr[A1 + LDA * 1]);
        vmovups(xmm1, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm1, ymm1, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm2, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm2, ymm2, ymm4, 0x20);
        add(A2, LDA);
        vmovups(xmm3, xword[A2]);
        vmovups(xmm4, xword[A2 + LDA * 4]);
        vperm2f128(ymm3, ymm3, ymm4, 0x20);
        add(A2, LDA);
        lea(A2, ptr[A2 + LDA * 4]);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm5, ymm0, ymm1);
        vunpcklps(ymm1, ymm2, ymm3);
        vunpckhps(ymm3, ymm2, ymm3);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vunpcklpd(ymm2, ymm5, ymm3);
        vunpckhpd(ymm3, ymm5, ymm3);
        vmulps(ymm0, ymm6, ymm0);
        vmulps(ymm1, ymm6, ymm1);
        vmulps(ymm2, ymm6, ymm2);
        vmulps(ymm3, ymm6, ymm3);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x60], ymm1);
        vmovups(yword[B - 0x40], ymm2);
        vmovups(yword[B - 0x20], ymm3);
        sub(A1, -16);
        sub(B, -128);
        dec(I);
        jg(labels[75], T_NEAR);
        align(4);

        L(labels[74]);
        test(M, 0x2);
        jle(labels[73], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovsd(xmm2, qword[A2]);
        vmovsd(xmm3, qword[A2 + LDA * 1]);
        vmovhps(xmm2, xmm2, qword[A2 + LDA * 2]);
        vmovhps(xmm3, xmm3, qword[A2 + LDA3 * 1]);
        lea(A2, ptr[A2 + LDA * 4]);
        vperm2f128(ymm0, ymm0, ymm2, 0x20);
        vperm2f128(ymm1, ymm1, ymm3, 0x20);
        vunpcklps(ymm4, ymm0, ymm1);
        vunpckhps(ymm1, ymm0, ymm1);
        vunpcklpd(ymm0, ymm4, ymm1);
        vunpckhpd(ymm1, ymm4, ymm1);
        vmulps(ymm0, ymm6, ymm0);
        vmulps(ymm1, ymm6, ymm1);
        vmovups(yword[B - 0x80], ymm0);
        vmovups(yword[B - 0x60], ymm1);
        sub(A1, -8);
        sub(B, -64);
        align(4);

        L(labels[73]);
        test(M, 0x1);
        jle(labels[72], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm4, xmm0, xmm2);
        lea(A2, ptr[A1 + LDA * 4]);
        vmovss(xmm0, dword[A2]);
        vmovss(xmm1, dword[A2 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A2 + LDA * 2]);
        vmovss(xmm3, dword[A2 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vperm2f128(ymm0, ymm0, ymm4, 0x2);
        vmulps(ymm0, ymm6, ymm0);
        vmovups(yword[B - 0x80], ymm0);
        lea(A2, ptr[A2 + LDA * 4]);
        sub(A1, -4);
        sub(B, -32);
        align(4);

        L(labels[72]);
        sub(N, 0x8);
        align(4);

        L(labels[71]);
        cmp(N, 0x4);
        jl(labels[66], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x4);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[69], T_NEAR);
        align(4);

        L(labels[70]);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm1, xword[A1 + LDA * 1]);
        vmovups(xmm2, xword[A1 + LDA * 2]);
        vmovups(xmm3, xword[A1 + LDA3 * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm5, xmm0, xmm1);
        vunpcklps(xmm1, xmm2, xmm3);
        vunpckhps(xmm3, xmm2, xmm3);
        vunpcklpd(xmm0, xmm4, xmm1);
        vunpckhpd(xmm1, xmm4, xmm1);
        vunpcklpd(xmm2, xmm5, xmm3);
        vunpckhpd(xmm3, xmm5, xmm3);
        vmulps(xmm0, xmm6, xmm0);
        vmulps(xmm1, xmm6, xmm1);
        vmulps(xmm2, xmm6, xmm2);
        vmulps(xmm3, xmm6, xmm3);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xword[B - 0x70], xmm1);
        vmovups(xword[B - 0x60], xmm2);
        vmovups(xword[B - 0x50], xmm3);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -16);
        sub(B, -64);
        dec(I);
        jg(labels[70], T_NEAR);
        align(4);

        L(labels[69]);
        test(M, 0x2);
        jle(labels[68], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vmovhps(xmm0, xmm0, qword[A1 + LDA * 2]);
        vmovhps(xmm1, xmm1, qword[A1 + LDA3 * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm1, xmm0, xmm1);
        vunpcklpd(xmm0, xmm4, xmm1);
        vunpckhpd(xmm1, xmm4, xmm1);
        vmulps(xmm0, xmm6, xmm0);
        vmulps(xmm1, xmm6, xmm1);
        vmovups(xword[B - 0x80], xmm0);
        vmovups(xword[B - 0x70], xmm1);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -8);
        sub(B, -32);
        align(4);

        L(labels[68]);
        test(M, 0x1);
        jle(labels[67], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmovss(xmm2, dword[A1 + LDA * 2]);
        vmovss(xmm3, dword[A1 + LDA3 * 1]);
        vunpcklps(xmm2, xmm2, xmm3);
        vunpcklpd(xmm0, xmm0, xmm2);
        vmulps(xmm0, xmm6, xmm0);
        vmovups(xword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 4]);
        sub(A1, -4);
        sub(B, -16);
        align(4);

        L(labels[67]);
        sub(N, 0x4);
        align(4);

        L(labels[66]);
        cmp(N, 0x2);
        jl(labels[61], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x2);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[64], T_NEAR);
        align(4);

        L(labels[65]);
        vmovups(xmm0, xword[A1]);
        vmovups(xmm1, xword[A1 + LDA * 1]);
        vunpcklps(xmm4, xmm0, xmm1);
        vunpckhps(xmm1, xmm0, xmm1);
        vmovaps(xmm0, xmm4);
        vmulps(xmm0, xmm6, xmm0);
        vmulps(xmm1, xmm6, xmm1);
        vmovlps(qword[B - 0x80], xmm0);
        vmovhps(qword[B - 0x78], xmm0);
        vmovlps(qword[B - 0x70], xmm1);
        vmovhps(qword[B - 0x68], xmm1);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -16);
        sub(B, -32);
        dec(I);
        jg(labels[65], T_NEAR);
        align(4);

        L(labels[64]);
        test(M, 0x2);
        jle(labels[63], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmovsd(xmm1, qword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        vmovhps(qword[B - 0x78], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -8);
        sub(B, -16);
        align(4);

        L(labels[63]);
        test(M, 0x1);
        jle(labels[62], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmovss(xmm1, dword[A1 + LDA * 1]);
        vunpcklps(xmm0, xmm0, xmm1);
        vmulps(xmm0, xmm6, xmm0);
        vmovlps(qword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 2]);
        sub(A1, -4);
        sub(B, -8);
        align(4);

        L(labels[62]);
        sub(N, 0x2);
        align(4);

        L(labels[61]);
        cmp(N, 0x1);
        jl(labels[56], T_NEAR);
        mov(A1, A);
        mov(I, LDA);
        imul(I, I, 0x1);
        add(A, I);
        mov(I, M);
        sar(I, 0x2);
        jle(labels[59], T_NEAR);
        align(4);

        L(labels[60]);
        vmovups(xmm0, xword[A1]);
        vmulps(xmm0, xmm6, xmm0);
        vpshufd(xmm1, xmm0, 0x55);
        vpshufd(xmm2, xmm0, 0xaa);
        vpshufd(xmm3, xmm0, 0xff);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        vmovss(dword[B - 0x78], xmm2);
        vmovss(dword[B - 0x74], xmm3);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -16);
        sub(B, -16);
        dec(I);
        jg(labels[60], T_NEAR);
        align(4);

        L(labels[59]);
        test(M, 0x2);
        jle(labels[58], T_NEAR);
        vmovsd(xmm0, qword[A1]);
        vmulps(xmm0, xmm6, xmm0);
        vpshufd(xmm1, xmm0, 0x55);
        vmovss(dword[B - 0x80], xmm0);
        vmovss(dword[B - 0x7c], xmm1);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -8);
        sub(B, -8);
        align(4);

        L(labels[58]);
        test(M, 0x1);
        jle(labels[57], T_NEAR);
        vmovss(xmm0, dword[A1]);
        vmulps(xmm0, xmm6, xmm0);
        vmovss(dword[B - 0x80], xmm0);
        lea(A2, ptr[A1 + LDA * 1]);
        sub(A1, -4);
        sub(B, -4);
        align(4);

        L(labels[57]);
        sub(N, 0x1);
        align(4);

        L(labels[56]);

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
