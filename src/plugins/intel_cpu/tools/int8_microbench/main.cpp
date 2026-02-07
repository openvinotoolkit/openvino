// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cpu/aarch64/brgemm/brgemm.hpp>

#include "nodes/kernels/aarch64/brgemm_int8_kernel.hpp"
#include "nodes/kernels/aarch64/jit_int8_conv_kernel.hpp"

#if defined(__linux__)
#    include <sys/auxv.h>
#endif
#if defined(__linux__) && defined(__aarch64__)
#    include <asm/hwcap.h>
#endif
#include <cpu/aarch64/cpu_isa_traits.hpp>
#if defined(OV_CPU_WITH_ACL)
#    include <arm_compute/core/QuantizationInfo.h>
#    include <arm_compute/core/TensorInfo.h>
#    include <arm_compute/core/Types.h>
#    include <arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h>
#    include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>
#    include <arm_compute/runtime/Tensor.h>
#    include <arm_compute/runtime/TensorAllocator.h>
#    include <arm_compute/runtime/NEON/NEFunctions.h>
#    include <arm_compute/core/utils/misc/ShapeCalculator.h>
#endif
#if defined(OV_CPU_WITH_KLEIDIAI)
#    include "kai/kai_common.h"
#    include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#    include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#    include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"
#    include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#    include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.h"
#endif

namespace {

struct Args {
    bool run_gemm = true;
    bool run_conv = true;
    bool use_acl = true;
    bool use_our = true;
    bool use_kleidiai = true;
    bool src_signed = false;
    size_t M = 128;
    size_t N = 128;
    size_t K = 256;
    size_t Nn = 1;
    size_t H = 28;
    size_t W = 28;
    size_t IC = 64;
    size_t OC = 64;
    size_t KH = 1;
    size_t KW = 1;
    size_t stride = 1;
    size_t warmup = 5;
    size_t iters = 50;
};

bool has_asimd_dotprod() {
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP_ASIMDDP)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return false;
#endif
}

bool has_i8mm() {
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP2_I8MM)
    return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
    return false;
#endif
}

bool has_sve() {
#if defined(__aarch64__)
    return dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::sve_128);
#else
    return false;
#endif
}

size_t packed_block_stride(size_t K, size_t oc_block) {
    const size_t k_blocks = K / 16;
    const size_t k_tail = K % 16;
    return k_blocks * oc_block * 16 + k_tail * oc_block;
}

size_t round_up(size_t value, size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

size_t packed_block_stride_mmla(size_t K, size_t oc_block) {
    const size_t Kp = round_up(K, 8);
    return (Kp / 8) * oc_block * 8;
}

void pack_dot_block(const int8_t* src, size_t K, size_t oc_block, int8_t* dst) {
    const size_t k_blocks = K / 16;
    const size_t k_tail = K % 16;
    size_t offset = 0;
    for (size_t kb = 0; kb < k_blocks; ++kb) {
        const size_t k_off = kb * 16;
        for (size_t oc = 0; oc < oc_block; ++oc) {
            const int8_t* s = src + oc * K + k_off;
            std::copy_n(s, 16, dst + offset);
            offset += 16;
        }
    }
    const size_t k_base = k_blocks * 16;
    for (size_t kt = 0; kt < k_tail; ++kt) {
        for (size_t oc = 0; oc < oc_block; ++oc) {
            dst[offset++] = src[oc * K + k_base + kt];
        }
    }
}

void pack_mmla_block(const int8_t* src, size_t K, size_t Kp, size_t oc_block, int8_t* dst) {
    const size_t k_blocks = Kp / 8;
    size_t offset = 0;
    for (size_t kb = 0; kb < k_blocks; ++kb) {
        const size_t k_off = kb * 8;
        const size_t valid = k_off < K ? std::min<size_t>(8, K - k_off) : 0;
        for (size_t oc = 0; oc + 1 < oc_block; oc += 2) {
            const int8_t* s0 = src + oc * K + k_off;
            const int8_t* s1 = src + (oc + 1) * K + k_off;
            if (valid == 8) {
                std::memcpy(dst + offset, s0, 8);
                std::memcpy(dst + offset + 8, s1, 8);
            } else {
                std::memset(dst + offset, 0, 16);
                if (valid > 0) {
                    std::memcpy(dst + offset, s0, valid);
                    std::memcpy(dst + offset + 8, s1, valid);
                }
            }
            offset += 16;
        }
    }
}

void print_usage() {
    std::cout << "ov_cpu_int8_microbench [options]\n"
              << "  --gemm/--conv                 Select GEMM or conv (default: both)\n"
              << "  --no-acl/--no-our/--no-kleidiai Disable ACL, our kernels, or KleidiAI\n"
              << "  --signed-src                   Use signed src (i8). Default: u8\n"
              << "  --m/--n/--k                    GEMM sizes (default 128x128x256)\n"
              << "  --n/--h/--w/--ic/--oc/--kh/--kw Conv sizes (default N1 H28 W28 IC64 OC64 KH1 KW1)\n"
              << "  --stride                       Conv stride (default 1)\n"
              << "  --warmup/--iters               Warmup and measure iterations\n";
}

bool parse_size_arg(const std::string& arg, const char* name, size_t& value) {
    if (arg.rfind(name, 0) == 0) {
        value = static_cast<size_t>(std::stoul(arg.substr(std::strlen(name))));
        return true;
    }
    return false;
}

Args parse_args(int argc, char** argv) {
    Args args;
    bool gemm_set = false;
    bool conv_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help") {
            print_usage();
            std::exit(0);
        } else if (arg == "--gemm") {
            args.run_gemm = true;
            gemm_set = true;
        } else if (arg == "--conv") {
            args.run_conv = true;
            conv_set = true;
        } else if (arg == "--no-acl") {
            args.use_acl = false;
        } else if (arg == "--no-our") {
            args.use_our = false;
        } else if (arg == "--no-kleidiai") {
            args.use_kleidiai = false;
        } else if (arg == "--signed-src") {
            args.src_signed = true;
        } else if (parse_size_arg(arg, "--m=", args.M) || parse_size_arg(arg, "--n=", args.N) ||
                   parse_size_arg(arg, "--k=", args.K) || parse_size_arg(arg, "--N=", args.Nn) ||
                   parse_size_arg(arg, "--H=", args.H) || parse_size_arg(arg, "--W=", args.W) ||
                   parse_size_arg(arg, "--IC=", args.IC) || parse_size_arg(arg, "--OC=", args.OC) ||
                   parse_size_arg(arg, "--KH=", args.KH) || parse_size_arg(arg, "--KW=", args.KW) ||
                   parse_size_arg(arg, "--stride=", args.stride) || parse_size_arg(arg, "--warmup=", args.warmup) ||
                   parse_size_arg(arg, "--iters=", args.iters)) {
            continue;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            print_usage();
            std::exit(1);
        }
    }
    if (gemm_set && !conv_set) {
        args.run_conv = false;
    } else if (conv_set && !gemm_set) {
        args.run_gemm = false;
    }
    return args;
}

template <typename T>
void fill_random(std::vector<T>& data, int lo, int hi, std::mt19937& gen) {
    std::uniform_int_distribution<int> dist(lo, hi);
    for (auto& v : data) {
        v = static_cast<T>(dist(gen));
    }
}

double to_ms(std::chrono::nanoseconds ns) {
    return static_cast<double>(ns.count()) / 1.0e6;
}

struct TimerResult {
    double ms = 0.0;
    double gops = 0.0;
};

constexpr size_t kBrgemmMB = 4;
constexpr size_t kBrgemmNB = 4;

TimerResult time_loop(size_t iters, double ops, const std::function<void()>& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iters; ++i) {
        fn();
    }
    auto end = std::chrono::high_resolution_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    const double ms = to_ms(ns) / static_cast<double>(iters);
    const double gops = (ops / 1.0e9) / (ms / 1.0e3);
    return {ms, gops};
}

TimerResult bench_our_gemm(const Args& args, bool use_block4, bool use_dot) {
    const size_t M = args.M;
    const size_t N = args.N;
    const size_t K = args.K;

    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N);
    std::mt19937 gen(42);
    fill_random(A, args.src_signed ? -10 : 0, 10, gen);
    fill_random(B, -10, 10, gen);

    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x4 block(args.src_signed);
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x4_dot block_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x4_udot block_udot;
    dot.create_ker();
    block.create_ker();
    block_dot.create_ker();
    block_udot.create_ker();
    const auto dot_ker = dot.ker();
    const auto block_ker = block.ker();
    const auto block_dot_ker = block_dot.ker();
    const auto block_udot_ker = block_udot.ker();
    const bool use_dot_s8 = use_dot && args.src_signed;
    const bool use_dot_u8 = use_dot && !args.src_signed;

    auto run = [&]() {
        for (size_t m = 0; m < M; ++m) {
            const uint8_t* a_ptr = A.data() + m * K;
            int32_t* c_ptr = C.data() + m * N;
            size_t n = 0;
                if (use_block4) {
                    for (; n + 4 <= N; n += 4) {
                        const int8_t* b_ptr = B.data() + n * K;
                        if (use_dot_s8) {
                            const int8_t* a_ptr_s8 = reinterpret_cast<const int8_t*>(a_ptr);
                            block_dot_ker(a_ptr_s8, b_ptr, c_ptr + n, K, K, 0);
                        } else if (use_dot_u8) {
                            block_udot_ker(a_ptr, b_ptr, c_ptr + n, K, K, 0);
                        } else {
                            block_ker(a_ptr, b_ptr, c_ptr + n, K, K, 0);
                        }
                    }
                }
            for (; n < N; ++n) {
                const int8_t* b_ptr = B.data() + n * K;
                dot_ker(a_ptr, b_ptr, c_ptr + n, K, 0);
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_brgemm_gemm(const Args& args) {
    if (!has_sve()) {
        return {-1.0, -1.0};
    }
    const size_t M = args.M;
    const size_t N = args.N;
    const size_t K = args.K;

    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N);
    std::mt19937 gen(42);
    fill_random(A, args.src_signed ? -10 : 0, 10, gen);
    fill_random(B, -10, 10, gen);

    std::vector<int8_t> B_packed(K * N);
    for (size_t k = 0; k < K; ++k) {
        for (size_t n = 0; n < N; ++n) {
            B_packed[k * N + n] = B[n * K + k];
        }
    }

    try {
        ov::intel_cpu::aarch64::BrgemmInt8Kernel brg(kBrgemmMB, kBrgemmNB, K, K, N, N, args.src_signed);
        ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
        dot.create_ker();
        const auto dot_ker = dot.ker();
        const bool use_onednn_brgemm = brg.uses_brgemm();

        auto run = [&]() {
            const size_t M_full = (M / kBrgemmMB) * kBrgemmMB;
            const size_t N_full = (N / kBrgemmNB) * kBrgemmNB;
            for (size_t m = 0; m < M_full; m += kBrgemmMB) {
                const uint8_t* a_block = A.data() + m * K;
                for (size_t n = 0; n < N_full; n += kBrgemmNB) {
                    const int8_t* b_block = use_onednn_brgemm ? (B_packed.data() + n) : (B.data() + n * K);
                    int32_t* c_block = C.data() + m * N + n;
                    brg.execute(a_block, b_block, c_block);
                }
                for (size_t n = N_full; n < N; ++n) {
                    const int8_t* b_ptr = B.data() + n * K;
                    for (size_t mi = 0; mi < kBrgemmMB; ++mi) {
                        const uint8_t* a_ptr = A.data() + (m + mi) * K;
                        int32_t* c_ptr = C.data() + (m + mi) * N + n;
                        dot_ker(a_ptr, b_ptr, c_ptr, K, 0);
                    }
                }
            }
            for (size_t m = M_full; m < M; ++m) {
                const uint8_t* a_ptr = A.data() + m * K;
                for (size_t n = 0; n < N; ++n) {
                    const int8_t* b_ptr = B.data() + n * K;
                    int32_t* c_ptr = C.data() + m * N + n;
                    dot_ker(a_ptr, b_ptr, c_ptr, K, 0);
                }
            }
        };

        for (size_t i = 0; i < args.warmup; ++i) {
            run();
        }

        const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        return time_loop(args.iters, ops, run);
    } catch (const std::exception&) {
        return {-1.0, -1.0};
    }
}

TimerResult bench_our_gemm_block8_dot(const Args& args) {
    const size_t M = args.M;
    const size_t N = args.N;
    const size_t K = args.K;

    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N);
    std::mt19937 gen(42);
    fill_random(A, args.src_signed ? -10 : 0, 10, gen);
    fill_random(B, -10, 10, gen);

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_dot block8_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_udot block8_udot;
    block8_dot.create_ker();
    block8_udot.create_ker();
    const auto block8_dot_ker = block8_dot.ker();
    const auto block8_udot_ker = block8_udot.ker();

    auto run = [&]() {
        for (size_t m = 0; m < M; ++m) {
            const uint8_t* a_ptr_u8 = A.data() + m * K;
            const int8_t* a_ptr_s8 = reinterpret_cast<const int8_t*>(a_ptr_u8);
            int32_t* c_ptr = C.data() + m * N;
            for (size_t n = 0; n + 8 <= N; n += 8) {
                const int8_t* b_ptr = B.data() + n * K;
                if (args.src_signed) {
                    block8_dot_ker(a_ptr_s8, b_ptr, c_ptr + n, K, K, 0);
                } else {
                    block8_udot_ker(a_ptr_u8, b_ptr, c_ptr + n, K, K, 0);
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_gemm_block8_dot_packed(const Args& args) {
    const size_t M = args.M;
    const size_t N = args.N;
    const size_t K = args.K;

    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N);
    std::mt19937 gen(42);
    fill_random(A, args.src_signed ? -10 : 0, 10, gen);
    fill_random(B, -10, 10, gen);

    const size_t n_blocks = N / 8;
    const size_t stride8 = packed_block_stride(K, 8);
    std::vector<int8_t> B_packed(n_blocks * stride8);
    for (size_t nb = 0; nb < n_blocks; ++nb) {
        const int8_t* src_block = B.data() + nb * 8 * K;
        int8_t* dst_block = B_packed.data() + nb * stride8;
        pack_dot_block(src_block, K, 8, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_dot_packed block8_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_udot_packed block8_udot;
    block8_dot.create_ker();
    block8_udot.create_ker();
    const auto block8_dot_ker = block8_dot.ker();
    const auto block8_udot_ker = block8_udot.ker();

    auto run = [&]() {
        for (size_t m = 0; m < M; ++m) {
            const uint8_t* a_ptr_u8 = A.data() + m * K;
            const int8_t* a_ptr_s8 = reinterpret_cast<const int8_t*>(a_ptr_u8);
            int32_t* c_ptr = C.data() + m * N;
            for (size_t n = 0; n + 8 <= N; n += 8) {
                const int8_t* b_ptr = B_packed.data() + (n / 8) * stride8;
                if (args.src_signed) {
                    block8_dot_ker(a_ptr_s8, b_ptr, c_ptr + n, K, 0, 0);
                } else {
                    block8_udot_ker(a_ptr_u8, b_ptr, c_ptr + n, K, 0, 0);
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_gemm_block4x4_dot(const Args& args) {
    const size_t M = args.M;
    const size_t N = args.N;
    const size_t K = args.K;

    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N);
    std::mt19937 gen(42);
    fill_random(A, -10, 10, gen);
    fill_random(B, -10, 10, gen);

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_dot block4x4_dot;
    block4x4_dot.create_ker();
    const auto block4x4_dot_ker = block4x4_dot.ker();

    auto run = [&]() {
        for (size_t m = 0; m + 4 <= M; m += 4) {
            const int8_t* a_ptrs[4] = {
                reinterpret_cast<const int8_t*>(A.data() + (m + 0) * K),
                reinterpret_cast<const int8_t*>(A.data() + (m + 1) * K),
                reinterpret_cast<const int8_t*>(A.data() + (m + 2) * K),
                reinterpret_cast<const int8_t*>(A.data() + (m + 3) * K),
            };
            for (size_t n = 0; n + 4 <= N; n += 4) {
                const int8_t* b_ptr = B.data() + n * K;
                int32_t* c_ptr = C.data() + m * N + n;
                block4x4_dot_ker(a_ptrs, b_ptr, c_ptr, K, K, N * sizeof(int32_t), 0);
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_gemm_block4x4_dot_packed(const Args& args) {
    const size_t M = args.M;
    const size_t N = args.N;
    const size_t K = args.K;

    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N);
    std::mt19937 gen(42);
    fill_random(A, args.src_signed ? -10 : 0, 10, gen);
    fill_random(B, -10, 10, gen);

    const size_t n_blocks = N / 4;
    const size_t stride4 = packed_block_stride(K, 4);
    std::vector<int8_t> B_packed(n_blocks * stride4);
    for (size_t nb = 0; nb < n_blocks; ++nb) {
        const int8_t* src_block = B.data() + nb * 4 * K;
        int8_t* dst_block = B_packed.data() + nb * stride4;
        pack_dot_block(src_block, K, 4, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_dot_packed block4x4_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_udot_packed block4x4_udot;
    block4x4_dot.create_ker();
    block4x4_udot.create_ker();
    const auto block4x4_dot_ker = block4x4_dot.ker();
    const auto block4x4_udot_ker = block4x4_udot.ker();

    auto run = [&]() {
        for (size_t m = 0; m + 4 <= M; m += 4) {
            const uint8_t* a_ptrs_u8[4] = {
                A.data() + (m + 0) * K,
                A.data() + (m + 1) * K,
                A.data() + (m + 2) * K,
                A.data() + (m + 3) * K,
            };
            const int8_t* a_ptrs_s8[4] = {
                reinterpret_cast<const int8_t*>(a_ptrs_u8[0]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[1]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[2]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[3]),
            };
            for (size_t n = 0; n + 4 <= N; n += 4) {
                const int8_t* b_ptr = B_packed.data() + (n / 4) * stride4;
                int32_t* c_ptr = C.data() + m * N + n;
                if (args.src_signed) {
                    block4x4_dot_ker(a_ptrs_s8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                } else {
                    block4x4_udot_ker(a_ptrs_u8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_gemm_block4x4_mmla_packed(const Args& args) {
    const size_t M = args.M;
    const size_t N = args.N;
    const size_t K = args.K;

    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N);
    std::mt19937 gen(42);
    fill_random(A, args.src_signed ? -10 : 0, 10, gen);
    fill_random(B, -10, 10, gen);

    const size_t Kp = round_up(K, 8);
    const size_t n_blocks = N / 4;
    const size_t stride4 = packed_block_stride_mmla(K, 4);
    std::vector<int8_t> B_packed(n_blocks * stride4);
    for (size_t nb = 0; nb < n_blocks; ++nb) {
        const int8_t* src_block = B.data() + nb * 4 * K;
        int8_t* dst_block = B_packed.data() + nb * stride4;
        pack_mmla_block(src_block, K, Kp, 4, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_smmla_packed block4x4_smmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_usmmla_packed block4x4_usmmla;
    block4x4_smmla.create_ker();
    block4x4_usmmla.create_ker();
    const auto block4x4_smmla_ker = block4x4_smmla.ker();
    const auto block4x4_usmmla_ker = block4x4_usmmla.ker();

    auto run = [&]() {
        for (size_t m = 0; m + 4 <= M; m += 4) {
            const uint8_t* a_ptrs_u8[4] = {
                A.data() + (m + 0) * K,
                A.data() + (m + 1) * K,
                A.data() + (m + 2) * K,
                A.data() + (m + 3) * K,
            };
            const int8_t* a_ptrs_s8[4] = {
                reinterpret_cast<const int8_t*>(a_ptrs_u8[0]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[1]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[2]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[3]),
            };
            for (size_t n = 0; n + 4 <= N; n += 4) {
                const int8_t* b_ptr = B_packed.data() + (n / 4) * stride4;
                int32_t* c_ptr = C.data() + m * N + n;
                if (args.src_signed) {
                    block4x4_smmla_ker(a_ptrs_s8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                } else {
                    block4x4_usmmla_ker(a_ptrs_u8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_gemm_block4x8_mmla_packed(const Args& args) {
    const size_t M = args.M;
    const size_t N = args.N;
    const size_t K = args.K;

    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N);
    std::mt19937 gen(42);
    fill_random(A, args.src_signed ? -10 : 0, 10, gen);
    fill_random(B, -10, 10, gen);

    const size_t Kp = round_up(K, 8);
    const size_t n_blocks = N / 8;
    const size_t stride8 = packed_block_stride_mmla(K, 8);
    std::vector<int8_t> B_packed(n_blocks * stride8);
    for (size_t nb = 0; nb < n_blocks; ++nb) {
        const int8_t* src_block = B.data() + nb * 8 * K;
        int8_t* dst_block = B_packed.data() + nb * stride8;
        pack_mmla_block(src_block, K, Kp, 8, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_smmla_packed block4x8_smmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_usmmla_packed block4x8_usmmla;
    block4x8_smmla.create_ker();
    block4x8_usmmla.create_ker();
    const auto block4x8_smmla_ker = block4x8_smmla.ker();
    const auto block4x8_usmmla_ker = block4x8_usmmla.ker();

    auto run = [&]() {
        for (size_t m = 0; m + 4 <= M; m += 4) {
            const uint8_t* a_ptrs_u8[4] = {
                A.data() + (m + 0) * K,
                A.data() + (m + 1) * K,
                A.data() + (m + 2) * K,
                A.data() + (m + 3) * K,
            };
            const int8_t* a_ptrs_s8[4] = {
                reinterpret_cast<const int8_t*>(a_ptrs_u8[0]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[1]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[2]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[3]),
            };
            for (size_t n = 0; n + 8 <= N; n += 8) {
                const int8_t* b_ptr = B_packed.data() + (n / 8) * stride8;
                int32_t* c_ptr = C.data() + m * N + n;
                if (args.src_signed) {
                    block4x8_smmla_ker(a_ptrs_s8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                } else {
                    block4x8_usmmla_ker(a_ptrs_u8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_gemm_block4x16_mmla_packed(const Args& args) {
    const size_t M = args.M;
    const size_t N = args.N;
    const size_t K = args.K;

    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N);
    std::mt19937 gen(42);
    fill_random(A, args.src_signed ? -10 : 0, 10, gen);
    fill_random(B, -10, 10, gen);

    const size_t Kp = round_up(K, 8);
    const size_t n_blocks = N / 16;
    const size_t stride16 = packed_block_stride_mmla(K, 16);
    std::vector<int8_t> B_packed(n_blocks * stride16);
    for (size_t nb = 0; nb < n_blocks; ++nb) {
        const int8_t* src_block = B.data() + nb * 16 * K;
        int8_t* dst_block = B_packed.data() + nb * stride16;
        pack_mmla_block(src_block, K, Kp, 16, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_smmla_packed block4x16_smmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_usmmla_packed block4x16_usmmla;
    block4x16_smmla.create_ker();
    block4x16_usmmla.create_ker();
    const auto block4x16_smmla_ker = block4x16_smmla.ker();
    const auto block4x16_usmmla_ker = block4x16_usmmla.ker();

    auto run = [&]() {
        for (size_t m = 0; m + 4 <= M; m += 4) {
            const uint8_t* a_ptrs_u8[4] = {
                A.data() + (m + 0) * K,
                A.data() + (m + 1) * K,
                A.data() + (m + 2) * K,
                A.data() + (m + 3) * K,
            };
            const int8_t* a_ptrs_s8[4] = {
                reinterpret_cast<const int8_t*>(a_ptrs_u8[0]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[1]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[2]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[3]),
            };
            for (size_t n = 0; n + 16 <= N; n += 16) {
                const int8_t* b_ptr = B_packed.data() + (n / 16) * stride16;
                int32_t* c_ptr = C.data() + m * N + n;
                if (args.src_signed) {
                    block4x16_smmla_ker(a_ptrs_s8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                } else {
                    block4x16_usmmla_ker(a_ptrs_u8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_1x1(const Args& args, bool use_block4, bool use_dot) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC);
    std::vector<int32_t> dst(N * H * W * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x4 block(args.src_signed);
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x4_dot block_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x4_udot block_udot;
    dot.create_ker();
    block.create_ker();
    block_dot.create_ker();
    block_udot.create_ker();
    const auto dot_ker = dot.ker();
    const auto block_ker = block.ker();
    const auto block_dot_ker = block_dot.ker();
    const auto block_udot_ker = block_udot.ker();
    const bool use_dot_s8 = use_dot && args.src_signed;
    const bool use_dot_u8 = use_dot && !args.src_signed;

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    const size_t src_off = ((n * H + h) * W + w) * IC;
                    const uint8_t* src_ptr = src.data() + src_off;
                    const size_t dst_off = ((n * H + h) * W + w) * OC;
                    int32_t* dst_ptr = dst.data() + dst_off;
                    size_t oc = 0;
                    if (use_block4) {
                        for (; oc + 4 <= OC; oc += 4) {
                            const int8_t* wei_ptr = wei.data() + oc * IC;
                            if (use_dot_s8) {
                                const int8_t* src_ptr_s8 = reinterpret_cast<const int8_t*>(src_ptr);
                                block_dot_ker(src_ptr_s8, wei_ptr, dst_ptr + oc, IC, IC, 0);
                            } else if (use_dot_u8) {
                                block_udot_ker(src_ptr, wei_ptr, dst_ptr + oc, IC, IC, 0);
                            } else {
                                block_ker(src_ptr, wei_ptr, dst_ptr + oc, IC, IC, 0);
                            }
                        }
                    }
                    for (; oc < OC; ++oc) {
                        const int8_t* wei_ptr = wei.data() + oc * IC;
                        dot_ker(src_ptr, wei_ptr, dst_ptr + oc, IC, 0);
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(H) * static_cast<double>(W) *
                       static_cast<double>(IC) * static_cast<double>(OC);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_1x1_brgemm(const Args& args) {
    if (!has_sve()) {
        return {-1.0, -1.0};
    }
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC);
    std::vector<int32_t> dst(N * H * W * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    std::vector<int8_t> packed_wei(IC * OC);
    for (size_t ic = 0; ic < IC; ++ic) {
        for (size_t oc = 0; oc < OC; ++oc) {
            packed_wei[ic * OC + oc] = wei[oc * IC + ic];
        }
    }

    try {
        ov::intel_cpu::aarch64::BrgemmInt8Kernel brg(kBrgemmMB, kBrgemmNB, IC, IC, OC, OC, args.src_signed);
        ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
        dot.create_ker();
        const auto dot_ker = dot.ker();
        const bool use_onednn_brgemm = brg.uses_brgemm();

        auto run = [&]() {
            const size_t w_full = (W / kBrgemmMB) * kBrgemmMB;
            const size_t oc_full = (OC / kBrgemmNB) * kBrgemmNB;
            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    size_t w = 0;
                    for (; w < w_full; w += kBrgemmMB) {
                        const uint8_t* src_block = src.data() + ((n * H + h) * W + w) * IC;
                        int32_t* dst_block = dst.data() + ((n * H + h) * W + w) * OC;
                        for (size_t oc = 0; oc < oc_full; oc += kBrgemmNB) {
                            const int8_t* wei_block =
                                use_onednn_brgemm ? (packed_wei.data() + oc) : (wei.data() + oc * IC);
                            brg.execute(src_block, wei_block, dst_block + oc);
                        }
                        for (size_t oc = oc_full; oc < OC; ++oc) {
                            const int8_t* wei_ptr = wei.data() + oc * IC;
                            for (size_t mi = 0; mi < kBrgemmMB; ++mi) {
                                const uint8_t* src_ptr = src_block + mi * IC;
                                int32_t* dst_ptr = dst_block + mi * OC + oc;
                                dot_ker(src_ptr, wei_ptr, dst_ptr, IC, 0);
                            }
                        }
                    }
                    for (; w < W; ++w) {
                        const uint8_t* src_ptr = src.data() + ((n * H + h) * W + w) * IC;
                        int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC;
                        for (size_t oc = 0; oc < OC; ++oc) {
                            const int8_t* wei_ptr = wei.data() + oc * IC;
                            dot_ker(src_ptr, wei_ptr, dst_ptr + oc, IC, 0);
                        }
                    }
                }
            }
        };

        for (size_t i = 0; i < args.warmup; ++i) {
            run();
        }

        const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(H) * static_cast<double>(W) *
                           static_cast<double>(IC) * static_cast<double>(OC);
        return time_loop(args.iters, ops, run);
    } catch (const std::exception&) {
        return {-1.0, -1.0};
    }
}

TimerResult bench_our_conv_1x1_block8_dot(const Args& args) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC);
    std::vector<int32_t> dst(N * H * W * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_dot block8_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_udot block8_udot;
    block8_dot.create_ker();
    block8_udot.create_ker();
    const auto block8_dot_ker = block8_dot.ker();
    const auto block8_udot_ker = block8_udot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    const size_t src_off = ((n * H + h) * W + w) * IC;
                    const uint8_t* src_ptr_u8 = src.data() + src_off;
                    const int8_t* src_ptr_s8 = reinterpret_cast<const int8_t*>(src_ptr_u8);
                    const size_t dst_off = ((n * H + h) * W + w) * OC;
                    int32_t* dst_ptr = dst.data() + dst_off;
                    for (size_t oc = 0; oc + 8 <= OC; oc += 8) {
                        const int8_t* wei_ptr = wei.data() + oc * IC;
                        if (args.src_signed) {
                            block8_dot_ker(src_ptr_s8, wei_ptr, dst_ptr + oc, IC, IC, 0);
                        } else {
                            block8_udot_ker(src_ptr_u8, wei_ptr, dst_ptr + oc, IC, IC, 0);
                        }
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(H) * static_cast<double>(W) *
                       static_cast<double>(IC) * static_cast<double>(OC);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_1x1_block8_dot_packed(const Args& args) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC);
    std::vector<int32_t> dst(N * H * W * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    const size_t oc_blocks = OC / 8;
    const size_t stride8 = packed_block_stride(IC, 8);
    std::vector<int8_t> wei_packed(oc_blocks * stride8);
    for (size_t ocb = 0; ocb < oc_blocks; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 8 * IC;
        int8_t* dst_block = wei_packed.data() + ocb * stride8;
        pack_dot_block(src_block, IC, 8, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_dot_packed block8_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_udot_packed block8_udot;
    block8_dot.create_ker();
    block8_udot.create_ker();
    const auto block8_dot_ker = block8_dot.ker();
    const auto block8_udot_ker = block8_udot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    const size_t src_off = ((n * H + h) * W + w) * IC;
                    const uint8_t* src_ptr_u8 = src.data() + src_off;
                    const int8_t* src_ptr_s8 = reinterpret_cast<const int8_t*>(src_ptr_u8);
                    const size_t dst_off = ((n * H + h) * W + w) * OC;
                    int32_t* dst_ptr = dst.data() + dst_off;
                    for (size_t oc = 0; oc + 8 <= OC; oc += 8) {
                        const int8_t* wei_ptr = wei_packed.data() + (oc / 8) * stride8;
                        if (args.src_signed) {
                            block8_dot_ker(src_ptr_s8, wei_ptr, dst_ptr + oc, IC, 0, 0);
                        } else {
                            block8_udot_ker(src_ptr_u8, wei_ptr, dst_ptr + oc, IC, 0, 0);
                        }
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(H) * static_cast<double>(W) *
                       static_cast<double>(IC) * static_cast<double>(OC);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_kxk(const Args& args) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t KH = args.KH;
    const size_t KW = args.KW;
    const size_t OH = H - KH + 1;
    const size_t OW = W - KW + 1;

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC * KH * KW);
    std::vector<int32_t> dst(N * OH * OW * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x4 block(args.src_signed);
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x4_dot block_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_dot block8_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x4_udot block_udot;
    dot.create_ker();
    block.create_ker();
    block_dot.create_ker();
    block8_dot.create_ker();
    block_udot.create_ker();
    const auto dot_ker = dot.ker();
    const auto block_ker = block.ker();
    const auto block_dot_ker = block_dot.ker();
    const auto block8_dot_ker = block8_dot.ker();
    const auto block_udot_ker = block_udot.ker();
    const bool use_dot_s8 = args.src_signed && has_asimd_dotprod();
    const bool use_dot_u8 = !args.src_signed && has_asimd_dotprod();

    std::vector<int8_t> packed_wei(KH * KW * OC * IC);
    for (size_t kh = 0; kh < KH; ++kh) {
        for (size_t kw = 0; kw < KW; ++kw) {
            for (size_t oc = 0; oc < OC; ++oc) {
                for (size_t ic = 0; ic < IC; ++ic) {
                    const size_t dst_idx = ((kh * KW + kw) * OC + oc) * IC + ic;
                    packed_wei[dst_idx] = wei[((oc * IC + ic) * KH + kh) * KW + kw];
                }
            }
        }
    }

    const size_t stride_w = IC;
    const size_t stride_h = W * IC;

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            const uint8_t* src_n = src.data() + n * H * W * IC;
            for (size_t h = 0; h < OH; ++h) {
                for (size_t w = 0; w < OW; ++w) {
                    int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC;
                    size_t oc = 0;
                    if (use_dot_s8) {
                        for (; oc + 8 <= OC; oc += 8) {
                            bool wrote = false;
                            for (size_t kh = 0; kh < KH; ++kh) {
                                for (size_t kw = 0; kw < KW; ++kw) {
                                    const uint8_t* src_ptr =
                                        src_n + (h + kh) * stride_h + (w + kw) * stride_w;
                                    const int8_t* wei_ptr =
                                        packed_wei.data() + ((kh * KW + kw) * OC + oc) * IC;
                                    block8_dot_ker(reinterpret_cast<const int8_t*>(src_ptr),
                                                   wei_ptr,
                                                   dst_ptr + oc,
                                                   IC,
                                                   IC,
                                                   wrote ? 1 : 0);
                                    wrote = true;
                                }
                            }
                        }
                    }
                    for (; oc + 4 <= OC; oc += 4) {
                        bool wrote = false;
                        for (size_t kh = 0; kh < KH; ++kh) {
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const uint8_t* src_ptr =
                                    src_n + (h + kh) * stride_h + (w + kw) * stride_w;
                                const int8_t* wei_ptr = packed_wei.data() + ((kh * KW + kw) * OC + oc) * IC;
                                if (use_dot_s8) {
                                    block_dot_ker(reinterpret_cast<const int8_t*>(src_ptr),
                                                  wei_ptr,
                                                  dst_ptr + oc,
                                                  IC,
                                                  IC,
                                                  wrote ? 1 : 0);
                                } else if (use_dot_u8) {
                                    block_udot_ker(src_ptr, wei_ptr, dst_ptr + oc, IC, IC, wrote ? 1 : 0);
                                } else {
                                    block_ker(src_ptr, wei_ptr, dst_ptr + oc, IC, IC, wrote ? 1 : 0);
                                }
                                wrote = true;
                            }
                        }
                    }
                    for (; oc < OC; ++oc) {
                        bool wrote = false;
                        for (size_t kh = 0; kh < KH; ++kh) {
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const uint8_t* src_ptr =
                                    src_n + (h + kh) * stride_h + (w + kw) * stride_w;
                                const int8_t* wei_ptr = packed_wei.data() + ((kh * KW + kw) * OC + oc) * IC;
                                dot_ker(src_ptr, wei_ptr, dst_ptr + oc, IC, wrote ? 1 : 0);
                                wrote = true;
                            }
                        }
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(OH) * static_cast<double>(OW) *
                       static_cast<double>(IC) * static_cast<double>(OC) * static_cast<double>(KH) *
                       static_cast<double>(KW);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_kxk_mmla_packed(const Args& args, size_t oc_block) {
    if (!has_i8mm() || (args.IC % 8 != 0) || (oc_block != 4 && oc_block != 8 && oc_block != 16)) {
        return {-1.0, -1.0};
    }
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t KH = args.KH;
    const size_t KW = args.KW;
    if (OC % oc_block != 0) {
        return {-1.0, -1.0};
    }
    const size_t OH = H - KH + 1;
    const size_t OW = W - KW + 1;
    const size_t stride_w = IC;
    const size_t stride_h = W * IC;
    const size_t ic_padded = round_up(IC, 8);

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC * KH * KW);
    std::vector<int32_t> dst(N * OH * OW * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    const bool use_s8 = args.src_signed;
    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(use_s8);
    dot.create_ker();
    const auto dot_ker = dot.ker();

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_smmla_packed ker4_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_smmla_packed ker8_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_smmla_packed ker16_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_usmmla_packed ker4_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_usmmla_packed ker8_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_usmmla_packed ker16_u8;
    ker4_s8.create_ker();
    ker8_s8.create_ker();
    ker16_s8.create_ker();
    ker4_u8.create_ker();
    ker8_u8.create_ker();
    ker16_u8.create_ker();

    std::vector<int8_t> wei_col(KH * KW * OC * IC);
    for (size_t kh = 0; kh < KH; ++kh) {
        for (size_t kw = 0; kw < KW; ++kw) {
            for (size_t oc = 0; oc < OC; ++oc) {
                for (size_t ic = 0; ic < IC; ++ic) {
                    const size_t dst_idx = ((kh * KW + kw) * OC + oc) * IC + ic;
                    wei_col[dst_idx] = wei[((oc * IC + ic) * KH + kh) * KW + kw];
                }
            }
        }
    }

    const size_t oc_blocks = OC / oc_block;
    const size_t stride_pack = packed_block_stride_mmla(IC, oc_block);
    if (stride_pack == 0) {
        return {-1.0, -1.0};
    }
    std::vector<int8_t> packed_wei(KH * KW * oc_blocks * stride_pack, 0);
    for (size_t kh = 0; kh < KH; ++kh) {
        for (size_t kw = 0; kw < KW; ++kw) {
            const size_t khkw = kh * KW + kw;
            for (size_t ocb = 0; ocb < oc_blocks; ++ocb) {
                const int8_t* src_block = wei_col.data() + (khkw * OC + ocb * oc_block) * IC;
                int8_t* dst_block = packed_wei.data() + (khkw * oc_blocks + ocb) * stride_pack;
                pack_mmla_block(src_block, IC, ic_padded, oc_block, dst_block);
            }
        }
    }

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            const uint8_t* src_n = src.data() + n * H * W * IC;
            for (size_t h = 0; h < OH; ++h) {
                size_t w = 0;
                for (; w + 4 <= OW; w += 4) {
                    for (size_t oc = 0; oc < OC; oc += oc_block) {
                        int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC + oc;
                        for (size_t kh = 0; kh < KH; ++kh) {
                            const size_t src_h_off = (h + kh) * stride_h;
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const size_t khkw = kh * KW + kw;
                                const size_t src_w_off = (w + kw) * stride_w;
                                const uint8_t* src_ptrs_u8[4] = {
                                    src_n + src_h_off + src_w_off + 0 * stride_w,
                                    src_n + src_h_off + src_w_off + 1 * stride_w,
                                    src_n + src_h_off + src_w_off + 2 * stride_w,
                                    src_n + src_h_off + src_w_off + 3 * stride_w,
                                };
                                const int8_t* wei_ptr =
                                    packed_wei.data() + (khkw * oc_blocks + (oc / oc_block)) * stride_pack;
                                const size_t accum = (kh == 0 && kw == 0) ? 0 : 1;
                                if (use_s8) {
                                    const int8_t* src_ptrs_s8[4] = {
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                                    };
                                    if (oc_block == 16) {
                                        ker16_s8.ker()(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), accum);
                                    } else if (oc_block == 8) {
                                        ker8_s8.ker()(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), accum);
                                    } else {
                                        ker4_s8.ker()(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), accum);
                                    }
                                } else {
                                    if (oc_block == 16) {
                                        ker16_u8.ker()(src_ptrs_u8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), accum);
                                    } else if (oc_block == 8) {
                                        ker8_u8.ker()(src_ptrs_u8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), accum);
                                    } else {
                                        ker4_u8.ker()(src_ptrs_u8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), accum);
                                    }
                                }
                            }
                        }
                    }
                }
                for (; w < OW; ++w) {
                    int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC;
                    for (size_t oc = 0; oc < OC; ++oc) {
                        bool wrote = false;
                        for (size_t kh = 0; kh < KH; ++kh) {
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const uint8_t* src_ptr =
                                    src_n + (h + kh) * stride_h + (w + kw) * stride_w;
                                const int8_t* wei_ptr = wei_col.data() + ((kh * KW + kw) * OC + oc) * IC;
                                dot_ker(src_ptr, wei_ptr, dst_ptr + oc, IC, wrote ? 1 : 0);
                                wrote = true;
                            }
                        }
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(OH) * static_cast<double>(OW) *
                       static_cast<double>(IC) * static_cast<double>(OC) * static_cast<double>(KH) *
                       static_cast<double>(KW);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_kxk_mmla_packed_fused(const Args& args, size_t oc_block) {
    if (!has_i8mm() || (oc_block != 4 && oc_block != 8 && oc_block != 16)) {
        return {-1.0, -1.0};
    }
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t KH = args.KH;
    const size_t KW = args.KW;
    if (OC % oc_block != 0) {
        return {-1.0, -1.0};
    }
    const size_t OH = H - KH + 1;
    const size_t OW = W - KW + 1;
    const size_t stride_w = IC;
    const size_t stride_h = W * IC;
    const size_t k_total = IC * KH * KW;
    const size_t k_total_padded = round_up(k_total, 8);

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC * KH * KW);
    std::vector<int32_t> dst(N * OH * OW * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    const bool use_s8 = args.src_signed;
    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(use_s8);
    dot.create_ker();
    const auto dot_ker = dot.ker();

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_smmla_packed ker4_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_smmla_packed ker8_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_smmla_packed ker16_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_usmmla_packed ker4_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_usmmla_packed ker8_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_usmmla_packed ker16_u8;
    ker4_s8.create_ker();
    ker8_s8.create_ker();
    ker16_s8.create_ker();
    ker4_u8.create_ker();
    ker8_u8.create_ker();
    ker16_u8.create_ker();

    std::vector<int8_t> fused_col(OC * k_total);
    for (size_t oc = 0; oc < OC; ++oc) {
        size_t dst_idx = oc * k_total;
        for (size_t kh = 0; kh < KH; ++kh) {
            for (size_t kw = 0; kw < KW; ++kw) {
                for (size_t ic = 0; ic < IC; ++ic) {
                    fused_col[dst_idx++] = wei[((oc * IC + ic) * KH + kh) * KW + kw];
                }
            }
        }
    }

    const size_t oc_blocks = OC / oc_block;
    const size_t stride_pack = packed_block_stride_mmla(k_total, oc_block);
    std::vector<int8_t> packed_wei(oc_blocks * stride_pack, 0);
    for (size_t ocb = 0; ocb < oc_blocks; ++ocb) {
        const int8_t* src_block = fused_col.data() + ocb * oc_block * k_total;
        int8_t* dst_block = packed_wei.data() + ocb * stride_pack;
        pack_mmla_block(src_block, k_total, k_total_padded, oc_block, dst_block);
    }

    std::vector<uint8_t> packed_src_u8;
    std::vector<int8_t> packed_src_s8;
    if (use_s8) {
        packed_src_s8.resize(4 * k_total_padded);
    } else {
        packed_src_u8.resize(4 * k_total_padded);
    }

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            const uint8_t* src_n = src.data() + n * H * W * IC;
            for (size_t h = 0; h < OH; ++h) {
                size_t w = 0;
                for (; w + 4 <= OW; w += 4) {
                    if (use_s8) {
                        int8_t* pack_ptr = packed_src_s8.data();
                        for (size_t m = 0; m < 4; ++m) {
                            const size_t base = m * k_total_padded;
                            if (k_total_padded != k_total) {
                                std::memset(pack_ptr + base, 0, k_total_padded);
                            }
                            for (size_t kh = 0; kh < KH; ++kh) {
                                const size_t src_h_off = (h + kh) * stride_h;
                                const size_t src_w_off = (w + m) * stride_w;
                                const uint8_t* src_ptr_u8 = src_n + src_h_off + src_w_off;
                                const int8_t* src_ptr = reinterpret_cast<const int8_t*>(src_ptr_u8);
                                const size_t dst_off = base + kh * KW * IC;
                                std::memcpy(pack_ptr + dst_off, src_ptr, KW * IC);
                            }
                        }
                    } else {
                        uint8_t* pack_ptr = packed_src_u8.data();
                        for (size_t m = 0; m < 4; ++m) {
                            const size_t base = m * k_total_padded;
                            if (k_total_padded != k_total) {
                                std::memset(pack_ptr + base, 0, k_total_padded);
                            }
                            for (size_t kh = 0; kh < KH; ++kh) {
                                const size_t src_h_off = (h + kh) * stride_h;
                                const size_t src_w_off = (w + m) * stride_w;
                                const uint8_t* src_ptr = src_n + src_h_off + src_w_off;
                                const size_t dst_off = base + kh * KW * IC;
                                std::memcpy(pack_ptr + dst_off, src_ptr, KW * IC);
                            }
                        }
                    }

                    const int8_t* src_ptrs_s8[4] = {
                        packed_src_s8.empty() ? nullptr : packed_src_s8.data() + 0 * k_total_padded,
                        packed_src_s8.empty() ? nullptr : packed_src_s8.data() + 1 * k_total_padded,
                        packed_src_s8.empty() ? nullptr : packed_src_s8.data() + 2 * k_total_padded,
                        packed_src_s8.empty() ? nullptr : packed_src_s8.data() + 3 * k_total_padded,
                    };
                    const uint8_t* src_ptrs_u8[4] = {
                        packed_src_u8.empty() ? nullptr : packed_src_u8.data() + 0 * k_total_padded,
                        packed_src_u8.empty() ? nullptr : packed_src_u8.data() + 1 * k_total_padded,
                        packed_src_u8.empty() ? nullptr : packed_src_u8.data() + 2 * k_total_padded,
                        packed_src_u8.empty() ? nullptr : packed_src_u8.data() + 3 * k_total_padded,
                    };

                    for (size_t oc = 0; oc < OC; oc += oc_block) {
                        int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC + oc;
                        const int8_t* wei_ptr = packed_wei.data() + (oc / oc_block) * stride_pack;
                        if (use_s8) {
                            if (oc_block == 16) {
                                ker16_s8.ker()(src_ptrs_s8,
                                               wei_ptr,
                                               dst_ptr,
                                               k_total_padded,
                                               0,
                                               OC * sizeof(int32_t),
                                               0);
                            } else if (oc_block == 8) {
                                ker8_s8.ker()(src_ptrs_s8,
                                              wei_ptr,
                                              dst_ptr,
                                              k_total_padded,
                                              0,
                                              OC * sizeof(int32_t),
                                              0);
                            } else {
                                ker4_s8.ker()(src_ptrs_s8,
                                              wei_ptr,
                                              dst_ptr,
                                              k_total_padded,
                                              0,
                                              OC * sizeof(int32_t),
                                              0);
                            }
                        } else {
                            if (oc_block == 16) {
                                ker16_u8.ker()(src_ptrs_u8,
                                               wei_ptr,
                                               dst_ptr,
                                               k_total_padded,
                                               0,
                                               OC * sizeof(int32_t),
                                               0);
                            } else if (oc_block == 8) {
                                ker8_u8.ker()(src_ptrs_u8,
                                              wei_ptr,
                                              dst_ptr,
                                              k_total_padded,
                                              0,
                                              OC * sizeof(int32_t),
                                              0);
                            } else {
                                ker4_u8.ker()(src_ptrs_u8,
                                              wei_ptr,
                                              dst_ptr,
                                              k_total_padded,
                                              0,
                                              OC * sizeof(int32_t),
                                              0);
                            }
                        }
                    }
                }
                for (; w < OW; ++w) {
                    int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC;
                    for (size_t oc = 0; oc < OC; ++oc) {
                        bool wrote = false;
                        for (size_t kh = 0; kh < KH; ++kh) {
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const uint8_t* src_ptr =
                                    src_n + (h + kh) * stride_h + (w + kw) * stride_w;
                                const int8_t* wei_ptr = fused_col.data() + oc * k_total + (kh * KW + kw) * IC;
                                dot_ker(src_ptr, wei_ptr, dst_ptr + oc, IC, wrote ? 1 : 0);
                                wrote = true;
                            }
                        }
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(OH) * static_cast<double>(OW) *
                       static_cast<double>(IC) * static_cast<double>(OC) * static_cast<double>(KH) *
                       static_cast<double>(KW);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_kxk_brgemm(const Args& args) {
    using dnnl::impl::cpu::aarch64::brgemm_batch_element_t;

    if (!has_sve()) {
        return {-1.0, -1.0};
    }
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t KH = args.KH;
    const size_t KW = args.KW;
    const size_t OH = H - KH + 1;
    const size_t OW = W - KW + 1;

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC * KH * KW);
    std::vector<int32_t> dst(N * OH * OW * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    std::vector<int8_t> packed_wei_dot(KH * KW * OC * IC);
    std::vector<int8_t> packed_wei_brgemm(KH * KW * IC * OC);
    for (size_t kh = 0; kh < KH; ++kh) {
        for (size_t kw = 0; kw < KW; ++kw) {
            for (size_t oc = 0; oc < OC; ++oc) {
                for (size_t ic = 0; ic < IC; ++ic) {
                    const int8_t v = wei[((oc * IC + ic) * KH + kh) * KW + kw];
                    packed_wei_dot[((kh * KW + kw) * OC + oc) * IC + ic] = v;
                    packed_wei_brgemm[(kh * KW + kw) * IC * OC + ic * OC + oc] = v;
                }
            }
        }
    }

    try {
        ov::intel_cpu::aarch64::BrgemmInt8Kernel brg(kBrgemmMB, kBrgemmNB, IC, IC, OC, OC, args.src_signed);
        ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
        dot.create_ker();
        const auto dot_ker = dot.ker();
        const bool use_onednn_brgemm = brg.uses_brgemm();

        std::vector<brgemm_batch_element_t> batch(KH * KW);
        std::vector<const int8_t*> batch_bases(KH * KW);
        const int8_t* packed_wei_base = use_onednn_brgemm ? packed_wei_brgemm.data() : packed_wei_dot.data();
        const size_t packed_wei_stride = use_onednn_brgemm ? (IC * OC) : (OC * IC);

        auto run = [&]() {
            const size_t ow_full = (OW / kBrgemmMB) * kBrgemmMB;
            const size_t oc_full = (OC / kBrgemmNB) * kBrgemmNB;
            for (size_t n = 0; n < N; ++n) {
                const uint8_t* src_n = src.data() + n * H * W * IC;
                for (size_t h = 0; h < OH; ++h) {
                    size_t w = 0;
                    for (; w < ow_full; w += kBrgemmMB) {
                        int32_t* dst_block = dst.data() + ((n * OH + h) * OW + w) * OC;
                        size_t idx = 0;
                        for (size_t kh = 0; kh < KH; ++kh) {
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const uint8_t* src_ptr = src_n + ((h + kh) * W + (w + kw)) * IC;
                                batch[idx].ptr.A = src_ptr;
                                batch_bases[idx] = packed_wei_base + (kh * KW + kw) * packed_wei_stride;
                                ++idx;
                            }
                        }
                        for (size_t oc = 0; oc < oc_full; oc += kBrgemmNB) {
                            for (size_t bi = 0; bi < batch.size(); ++bi) {
                                batch[bi].ptr.B = use_onednn_brgemm ? (batch_bases[bi] + oc) : (batch_bases[bi] + oc * IC);
                            }
                            brg.execute_batch(batch.data(), static_cast<int>(batch.size()), dst_block + oc);
                        }
                        for (size_t oc = oc_full; oc < OC; ++oc) {
                            int32_t* dst_ptr = dst_block + oc;
                            bool wrote = false;
                            for (size_t kh = 0; kh < KH; ++kh) {
                                for (size_t kw = 0; kw < KW; ++kw) {
                                    const uint8_t* src_ptr = src_n + ((h + kh) * W + (w + kw)) * IC;
                                    const int8_t* wei_ptr =
                                        packed_wei_dot.data() + ((kh * KW + kw) * OC + oc) * IC;
                                    dot_ker(src_ptr, wei_ptr, dst_ptr, IC, wrote ? 1 : 0);
                                    wrote = true;
                                }
                            }
                        }
                    }
                    for (; w < OW; ++w) {
                        int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC;
                        for (size_t oc = 0; oc < OC; ++oc) {
                            bool wrote = false;
                            for (size_t kh = 0; kh < KH; ++kh) {
                                for (size_t kw = 0; kw < KW; ++kw) {
                                    const uint8_t* src_ptr = src_n + ((h + kh) * W + (w + kw)) * IC;
                                    const int8_t* wei_ptr =
                                        packed_wei_dot.data() + ((kh * KW + kw) * OC + oc) * IC;
                                    dot_ker(src_ptr, wei_ptr, dst_ptr + oc, IC, wrote ? 1 : 0);
                                    wrote = true;
                                }
                            }
                        }
                    }
                }
            }
        };

        for (size_t i = 0; i < args.warmup; ++i) {
            run();
        }

        const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(OH) * static_cast<double>(OW) *
                           static_cast<double>(IC) * static_cast<double>(OC) * static_cast<double>(KH) *
                           static_cast<double>(KW);
        return time_loop(args.iters, ops, run);
    } catch (const std::exception&) {
        return {-1.0, -1.0};
    }
}

TimerResult bench_our_conv_1x1_block4x4_dot(const Args& args) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC);
    std::vector<int32_t> dst(N * H * W * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_dot block4x4_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_udot block4x4_udot;
    block4x4_dot.create_ker();
    block4x4_udot.create_ker();
    const auto block4x4_dot_ker = block4x4_dot.ker();
    const auto block4x4_udot_ker = block4x4_udot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                size_t w = 0;
                for (; w + 4 <= W; w += 4) {
                    const size_t src_off0 = ((n * H + h) * W + w + 0) * IC;
                    const size_t src_off1 = ((n * H + h) * W + w + 1) * IC;
                    const size_t src_off2 = ((n * H + h) * W + w + 2) * IC;
                    const size_t src_off3 = ((n * H + h) * W + w + 3) * IC;
                    const uint8_t* src_ptrs_u8[4] = {
                        src.data() + src_off0,
                        src.data() + src_off1,
                        src.data() + src_off2,
                        src.data() + src_off3,
                    };
                    const size_t dst_off = ((n * H + h) * W + w) * OC;
                    int32_t* dst_ptr = dst.data() + dst_off;
                    for (size_t oc = 0; oc + 4 <= OC; oc += 4) {
                        const int8_t* wei_ptr = wei.data() + oc * IC;
                        if (args.src_signed) {
                            const int8_t* src_ptrs_s8[4] = {
                                reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                                reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                                reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                                reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                            };
                            block4x4_dot_ker(src_ptrs_s8, wei_ptr, dst_ptr + oc, IC, IC, OC * sizeof(int32_t), 0);
                        } else {
                            block4x4_udot_ker(src_ptrs_u8, wei_ptr, dst_ptr + oc, IC, IC, OC * sizeof(int32_t), 0);
                        }
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(H) * static_cast<double>(W) *
                       static_cast<double>(IC) * static_cast<double>(OC);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_1x1_block4x4_dot_packed(const Args& args) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC);
    std::vector<int32_t> dst(N * H * W * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    const size_t oc_blocks = OC / 4;
    const size_t stride4 = packed_block_stride(IC, 4);
    std::vector<int8_t> wei_packed(oc_blocks * stride4);
    for (size_t ocb = 0; ocb < oc_blocks; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 4 * IC;
        int8_t* dst_block = wei_packed.data() + ocb * stride4;
        pack_dot_block(src_block, IC, 4, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_dot_packed block4x4_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_udot_packed block4x4_udot;
    block4x4_dot.create_ker();
    block4x4_udot.create_ker();
    const auto block4x4_dot_ker = block4x4_dot.ker();
    const auto block4x4_udot_ker = block4x4_udot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w + 4 <= W; w += 4) {
                    const size_t base = ((n * H + h) * W + w) * IC;
                    const uint8_t* src_ptrs_u8[4] = {
                        src.data() + base + 0 * IC,
                        src.data() + base + 1 * IC,
                        src.data() + base + 2 * IC,
                        src.data() + base + 3 * IC,
                    };
                    const int8_t* src_ptrs_s8[4] = {
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                    };
                    for (size_t oc = 0; oc + 4 <= OC; oc += 4) {
                        const int8_t* wei_ptr = wei_packed.data() + (oc / 4) * stride4;
                        int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC + oc;
                        if (args.src_signed) {
                            block4x4_dot_ker(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), 0);
                        } else {
                            block4x4_udot_ker(src_ptrs_u8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), 0);
                        }
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(H) * static_cast<double>(W) *
                       static_cast<double>(IC) * static_cast<double>(OC);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_1x1_block4x4_mmla_packed(const Args& args) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC);
    std::vector<int32_t> dst(N * H * W * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    const size_t oc_blocks = OC / 4;
    const size_t ic_padded = round_up(IC, 8);
    const size_t stride4 = packed_block_stride_mmla(IC, 4);
    std::vector<int8_t> wei_packed(oc_blocks * stride4);
    for (size_t ocb = 0; ocb < oc_blocks; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 4 * IC;
        int8_t* dst_block = wei_packed.data() + ocb * stride4;
        pack_mmla_block(src_block, IC, ic_padded, 4, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_smmla_packed block4x4_smmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_usmmla_packed block4x4_usmmla;
    block4x4_smmla.create_ker();
    block4x4_usmmla.create_ker();
    const auto block4x4_smmla_ker = block4x4_smmla.ker();
    const auto block4x4_usmmla_ker = block4x4_usmmla.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w + 4 <= W; w += 4) {
                    const size_t base = ((n * H + h) * W + w) * IC;
                    const uint8_t* src_ptrs_u8[4] = {
                        src.data() + base + 0 * IC,
                        src.data() + base + 1 * IC,
                        src.data() + base + 2 * IC,
                        src.data() + base + 3 * IC,
                    };
                    const int8_t* src_ptrs_s8[4] = {
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                    };
                    for (size_t oc = 0; oc + 4 <= OC; oc += 4) {
                        const int8_t* wei_ptr = wei_packed.data() + (oc / 4) * stride4;
                        int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC + oc;
                        if (args.src_signed) {
                            block4x4_smmla_ker(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), 0);
                        } else {
                            block4x4_usmmla_ker(src_ptrs_u8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), 0);
                        }
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(H) * static_cast<double>(W) *
                       static_cast<double>(IC) * static_cast<double>(OC);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_1x1_block4x8_mmla_packed(const Args& args) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC);
    std::vector<int32_t> dst(N * H * W * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    const size_t oc_blocks = OC / 8;
    const size_t ic_padded = round_up(IC, 8);
    const size_t stride8 = packed_block_stride_mmla(IC, 8);
    std::vector<int8_t> wei_packed(oc_blocks * stride8);
    for (size_t ocb = 0; ocb < oc_blocks; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 8 * IC;
        int8_t* dst_block = wei_packed.data() + ocb * stride8;
        pack_mmla_block(src_block, IC, ic_padded, 8, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_smmla_packed block4x8_smmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_usmmla_packed block4x8_usmmla;
    block4x8_smmla.create_ker();
    block4x8_usmmla.create_ker();
    const auto block4x8_smmla_ker = block4x8_smmla.ker();
    const auto block4x8_usmmla_ker = block4x8_usmmla.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w + 4 <= W; w += 4) {
                    const size_t base = ((n * H + h) * W + w) * IC;
                    const uint8_t* src_ptrs_u8[4] = {
                        src.data() + base + 0 * IC,
                        src.data() + base + 1 * IC,
                        src.data() + base + 2 * IC,
                        src.data() + base + 3 * IC,
                    };
                    const int8_t* src_ptrs_s8[4] = {
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                    };
                    for (size_t oc = 0; oc + 8 <= OC; oc += 8) {
                        const int8_t* wei_ptr = wei_packed.data() + (oc / 8) * stride8;
                        int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC + oc;
                        if (args.src_signed) {
                            block4x8_smmla_ker(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), 0);
                        } else {
                            block4x8_usmmla_ker(src_ptrs_u8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), 0);
                        }
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(H) * static_cast<double>(W) *
                       static_cast<double>(IC) * static_cast<double>(OC);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_1x1_block4x16_mmla_packed(const Args& args) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC);
    std::vector<int32_t> dst(N * H * W * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    const size_t oc_blocks = OC / 16;
    const size_t ic_padded = round_up(IC, 8);
    const size_t stride16 = packed_block_stride_mmla(IC, 16);
    std::vector<int8_t> wei_packed(oc_blocks * stride16);
    for (size_t ocb = 0; ocb < oc_blocks; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 16 * IC;
        int8_t* dst_block = wei_packed.data() + ocb * stride16;
        pack_mmla_block(src_block, IC, ic_padded, 16, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_smmla_packed block4x16_smmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_usmmla_packed block4x16_usmmla;
    block4x16_smmla.create_ker();
    block4x16_usmmla.create_ker();
    const auto block4x16_smmla_ker = block4x16_smmla.ker();
    const auto block4x16_usmmla_ker = block4x16_usmmla.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w + 4 <= W; w += 4) {
                    const size_t base = ((n * H + h) * W + w) * IC;
                    const uint8_t* src_ptrs_u8[4] = {
                        src.data() + base + 0 * IC,
                        src.data() + base + 1 * IC,
                        src.data() + base + 2 * IC,
                        src.data() + base + 3 * IC,
                    };
                    const int8_t* src_ptrs_s8[4] = {
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                        reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                    };
                    for (size_t oc = 0; oc + 16 <= OC; oc += 16) {
                        const int8_t* wei_ptr = wei_packed.data() + (oc / 16) * stride16;
                        int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC + oc;
                        if (args.src_signed) {
                            block4x16_smmla_ker(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), 0);
                        } else {
                            block4x16_usmmla_ker(src_ptrs_u8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), 0);
                        }
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(H) * static_cast<double>(W) *
                       static_cast<double>(IC) * static_cast<double>(OC);
    return time_loop(args.iters, ops, run);
}

#if defined(OV_CPU_WITH_KLEIDIAI)
static constexpr kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel kKleidiI8Dotprod = {
    kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
    kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod};

static constexpr kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel kKleidiI8I8mm = {
    kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
    kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm};

TimerResult bench_kleidiai_gemm(const Args& args, bool& supported) {
    supported = false;
    const size_t M = args.M;
    const size_t N = args.N;
    const size_t K = args.K;
    if (K % 8 != 0 || (!has_i8mm() && !has_asimd_dotprod())) {
        return {};
    }

    const auto* ukernel = has_i8mm() ? &kKleidiI8I8mm : &kKleidiI8Dotprod;
    supported = true;

    std::vector<float> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<float> C(M * N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_a(-10.0f, 10.0f);
    for (auto& v : A) {
        v = dist_a(gen);
    }
    fill_random(B, -10, 10, gen);

    const size_t mr = ukernel->get_mr();
    const size_t nr = ukernel->get_nr();
    const size_t kr = ukernel->get_kr();
    const size_t sr = ukernel->get_sr();
    const size_t m_step = 16;
    const size_t n_step = ukernel->get_n_step();
    const size_t lhs_stride = K * sizeof(float);
    const size_t dst_stride_row = N * sizeof(float);
    const size_t dst_stride_col = sizeof(float);

    const size_t lhs_block_bytes = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m_step, K, mr, kr, sr);
    const size_t m_blocks = (M + m_step - 1) / m_step;
    std::vector<uint8_t> lhs_packed(lhs_block_bytes * m_blocks);

    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);
    std::vector<int8_t> rhs_packed(rhs_packed_size);
    std::vector<float> rhs_scales(N, 1.0f);
    std::vector<float> rhs_bias(N, 0.0f);
    kai_rhs_pack_qsi8cx_params params{};
    params.lhs_zero_point = 1;
    params.scale_multiplier = 1.0f;

    kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(1,
                                             N,
                                             K,
                                             nr,
                                             kr,
                                             sr,
                                             B.data(),
                                             rhs_bias.data(),
                                             rhs_scales.data(),
                                             rhs_packed.data(),
                                             0,
                                             &params);

    for (size_t m_blk = 0; m_blk < m_blocks; ++m_blk) {
        const size_t M_iter = std::min(M - m_blk * m_step, m_step);
        auto* lhs_packed_block = lhs_packed.data() + m_blk * lhs_block_bytes;
        kai_run_lhs_quant_pack_qai8dxp_f32(M_iter,
                                           K,
                                           mr,
                                           kr,
                                           sr,
                                           0,
                                           A.data() + m_blk * m_step * K,
                                           lhs_stride,
                                           lhs_packed_block);
    }

    const size_t lhs_packed_offset = ukernel->get_lhs_packed_offset(0, K);

    auto run = [&]() {
        for (size_t m_blk = 0; m_blk < m_blocks; ++m_blk) {
            const size_t M_iter = std::min(M - m_blk * m_step, m_step);
            const auto* lhs_packed_block = lhs_packed.data() + m_blk * lhs_block_bytes;
            for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
                const size_t N_iter = std::min(N - n_idx, n_step);
                const size_t rhs_offset = ukernel->get_rhs_packed_offset(n_idx, K);
                const size_t dst_offset = ukernel->get_dst_offset(m_blk * m_step, n_idx, dst_stride_row);
                const void* lhs_ptr = lhs_packed_block + lhs_packed_offset;
                const void* rhs_ptr = rhs_packed.data() + rhs_offset;
                float* dst_ptr = C.data() + dst_offset / sizeof(float);
                ukernel->run_matmul(M_iter,
                                    N_iter,
                                    K,
                                    lhs_ptr,
                                    rhs_ptr,
                                    dst_ptr,
                                    dst_stride_row,
                                    dst_stride_col,
                                    std::numeric_limits<float>::lowest(),
                                    std::numeric_limits<float>::max());
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return time_loop(args.iters, ops, run);
}
#endif

#if defined(OV_CPU_WITH_ACL)
TimerResult bench_acl_gemm(const Args& args) {
    using namespace arm_compute;

    const size_t M = args.M;
    const size_t N = args.N;
    const size_t K = args.K;

    const DataType src_dt = args.src_signed ? DataType::QASYMM8_SIGNED : DataType::QASYMM8;
    const QuantizationInfo qinfo(1.0F, 0);

    Tensor a;
    Tensor b;
    Tensor c;
    a.allocator()->init(TensorInfo(TensorShape(K, M), 1, src_dt, DataLayout::NHWC));
    b.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::QASYMM8_SIGNED, DataLayout::NHWC));
    c.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::S32, DataLayout::NHWC));
    a.info()->set_quantization_info(qinfo);
    b.info()->set_quantization_info(qinfo);
    c.info()->set_quantization_info(qinfo);

    NEGEMMLowpMatrixMultiplyCore gemm;
    gemm.configure(&a, &b, nullptr, &c, GEMMInfo());

    a.allocator()->allocate();
    b.allocator()->allocate();
    c.allocator()->allocate();

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist_a(args.src_signed ? -10 : 0, 10);
    std::uniform_int_distribution<int> dist_b(-10, 10);
    auto* a_ptr_u8 = reinterpret_cast<uint8_t*>(a.buffer());
    auto* a_ptr_s8 = reinterpret_cast<int8_t*>(a.buffer());
    auto* b_ptr = reinterpret_cast<int8_t*>(b.buffer());
    for (size_t i = 0; i < M * K; ++i) {
        if (args.src_signed) {
            a_ptr_s8[i] = static_cast<int8_t>(dist_a(gen));
        } else {
            a_ptr_u8[i] = static_cast<uint8_t>(dist_a(gen));
        }
    }
    for (size_t i = 0; i < N * K; ++i) {
        b_ptr[i] = static_cast<int8_t>(dist_b(gen));
    }

    for (size_t i = 0; i < args.warmup; ++i) {
        gemm.run();
    }

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return time_loop(args.iters, ops, [&]() { gemm.run(); });
}

TimerResult bench_acl_conv_1x1(const Args& args) {
    using namespace arm_compute;
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;

    const DataType src_dt = args.src_signed ? DataType::QASYMM8_SIGNED : DataType::QASYMM8;
    const QuantizationInfo qinfo(1.0F, 0);

    Tensor src;
    Tensor wei;
    Tensor dst;
    const TensorInfo src_info(TensorShape(IC, W, H, N), 1, src_dt, DataLayout::NHWC);
    const TensorInfo wei_info(TensorShape(IC, args.KW, args.KH, OC), 1, DataType::QASYMM8_SIGNED, DataLayout::NHWC);
    TensorInfo dst_info;
    const PadStrideInfo ps_info(args.stride, args.stride, 0, 0);
    dst_info = TensorInfo(misc::shape_calculator::compute_deep_convolution_shape(src_info, wei_info, ps_info),
                          1,
                          src_dt,
                          DataLayout::NHWC);

    src.allocator()->init(src_info);
    wei.allocator()->init(wei_info);
    dst.allocator()->init(dst_info);
    src.info()->set_quantization_info(qinfo);
    wei.info()->set_quantization_info(qinfo);
    dst.info()->set_quantization_info(qinfo);

    NEConvolutionLayer conv;
    conv.configure(&src, &wei, nullptr, &dst, ps_info);

    src.allocator()->allocate();
    wei.allocator()->allocate();
    dst.allocator()->allocate();

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist_s(args.src_signed ? -10 : 0, 10);
    std::uniform_int_distribution<int> dist_w(-10, 10);
    auto* src_ptr_u8 = reinterpret_cast<uint8_t*>(src.buffer());
    auto* src_ptr_s8 = reinterpret_cast<int8_t*>(src.buffer());
    auto* wei_ptr = reinterpret_cast<int8_t*>(wei.buffer());
    const size_t src_elems = N * H * W * IC;
    const size_t wei_elems = IC * args.KH * args.KW * OC;
    for (size_t i = 0; i < src_elems; ++i) {
        if (args.src_signed) {
            src_ptr_s8[i] = static_cast<int8_t>(dist_s(gen));
        } else {
            src_ptr_u8[i] = static_cast<uint8_t>(dist_s(gen));
        }
    }
    for (size_t i = 0; i < wei_elems; ++i) {
        wei_ptr[i] = static_cast<int8_t>(dist_w(gen));
    }

    for (size_t i = 0; i < args.warmup; ++i) {
        conv.run();
    }

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(H) * static_cast<double>(W) *
                       static_cast<double>(IC) * static_cast<double>(OC);
    return time_loop(args.iters, ops, [&]() { conv.run(); });
}
#endif

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    std::cout << "INT8 microbench (AArch64)\n";
    std::cout << "src=" << (args.src_signed ? "s8" : "u8") << " weights=s8\n";

    if (args.run_gemm) {
        std::cout << "\nGEMM M=" << args.M << " N=" << args.N << " K=" << args.K << "\n";
        if (args.use_our) {
            const auto dot = bench_our_gemm(args, false, false);
            const auto block4 = bench_our_gemm(args, true, false);
            const auto brgemm = bench_our_brgemm_gemm(args);
            std::cout << "  our_dot     : " << dot.ms << " ms, " << dot.gops << " GOPS\n";
            std::cout << "  our_block4  : " << block4.ms << " ms, " << block4.gops << " GOPS\n";
            if (brgemm.ms >= 0.0) {
                std::cout << "  our_brgemm4x4 : " << brgemm.ms << " ms, " << brgemm.gops << " GOPS\n";
            } else {
                std::cout << "  our_brgemm4x4 : unsupported (brgemm unavailable)\n";
            }
            if (has_asimd_dotprod()) {
                const auto block4_dot = bench_our_gemm(args, true, true);
                std::cout << "  our_block4_dot : " << block4_dot.ms << " ms, " << block4_dot.gops << " GOPS\n";
                if (args.N % 8 == 0) {
                    const auto block8_dot = bench_our_gemm_block8_dot(args);
                    std::cout << "  our_block8_dot : " << block8_dot.ms << " ms, " << block8_dot.gops << " GOPS\n";
                    const auto block8_dot_packed = bench_our_gemm_block8_dot_packed(args);
                    std::cout << "  our_block8_dot_packed : " << block8_dot_packed.ms << " ms, "
                              << block8_dot_packed.gops << " GOPS\n";
                } else {
                    std::cout << "  our_block8_dot : unsupported (N multiple of 8)\n";
                    std::cout << "  our_block8_dot_packed : unsupported (N multiple of 8)\n";
                }
                if (args.M % 4 == 0 && args.N % 4 == 0) {
                    const auto block4x4_dot = bench_our_gemm_block4x4_dot(args);
                    std::cout << "  our_block4x4_dot : " << block4x4_dot.ms << " ms, " << block4x4_dot.gops
                              << " GOPS\n";
                    const auto block4x4_dot_packed = bench_our_gemm_block4x4_dot_packed(args);
                    std::cout << "  our_block4x4_dot_packed : " << block4x4_dot_packed.ms << " ms, "
                              << block4x4_dot_packed.gops << " GOPS\n";
                } else {
                    std::cout << "  our_block4x4_dot : unsupported (M,N multiple of 4)\n";
                    std::cout << "  our_block4x4_dot_packed : unsupported (M,N multiple of 4)\n";
                }
            }
            if (has_i8mm()) {
                if (args.M % 4 == 0 && args.N % 4 == 0 && args.K % 8 == 0) {
                    const auto block4x4_mmla = bench_our_gemm_block4x4_mmla_packed(args);
                    std::cout << "  our_block4x4_mmla_packed : " << block4x4_mmla.ms << " ms, "
                              << block4x4_mmla.gops << " GOPS\n";
                } else {
                    std::cout << "  our_block4x4_mmla_packed : unsupported (M,N multiple of 4, K multiple of 8)\n";
                }
                if (args.M % 4 == 0 && args.N % 8 == 0 && args.K % 8 == 0) {
                    const auto block4x8_mmla = bench_our_gemm_block4x8_mmla_packed(args);
                    std::cout << "  our_block4x8_mmla_packed : " << block4x8_mmla.ms << " ms, "
                              << block4x8_mmla.gops << " GOPS\n";
                } else {
                    std::cout << "  our_block4x8_mmla_packed : unsupported (M multiple of 4, N multiple of 8, K multiple of 8)\n";
                }
                if (args.M % 4 == 0 && args.N % 16 == 0 && args.K % 8 == 0) {
                    const auto block4x16_mmla = bench_our_gemm_block4x16_mmla_packed(args);
                    std::cout << "  our_block4x16_mmla_packed : " << block4x16_mmla.ms << " ms, "
                              << block4x16_mmla.gops << " GOPS\n";
                } else {
                    std::cout << "  our_block4x16_mmla_packed : unsupported (M multiple of 4, N multiple of 16, K multiple of 8)\n";
                }
            }
        }
#if defined(OV_CPU_WITH_ACL)
        if (args.use_acl) {
            const auto acl = bench_acl_gemm(args);
            std::cout << "  acl_gemm    : " << acl.ms << " ms, " << acl.gops << " GOPS\n";
        }
#else
        if (args.use_acl) {
            std::cout << "  acl_gemm    : unavailable (built without ACL)\n";
        }
#endif
#if defined(OV_CPU_WITH_KLEIDIAI)
        if (args.use_kleidiai) {
            bool supported = false;
            const auto kleidiai = bench_kleidiai_gemm(args, supported);
            if (!supported) {
                std::cout << "  kleidiai_gemm : unsupported (requires dotprod/i8mm, K multiple of 8)\n";
            } else {
                std::cout << "  kleidiai_gemm : " << kleidiai.ms << " ms, " << kleidiai.gops << " GOPS\n";
            }
        }
#else
        if (args.use_kleidiai) {
            std::cout << "  kleidiai_gemm : unavailable (built without KleidiAI)\n";
        }
#endif
    }

    if (args.run_conv) {
        std::cout << "\nCONV N=" << args.Nn << " H=" << args.H << " W=" << args.W << " IC=" << args.IC
                  << " OC=" << args.OC << " KH=" << args.KH << " KW=" << args.KW << " stride=" << args.stride
                  << "\n";
        if (args.stride != 1) {
            std::cout << "  our_conv    : unsupported (only stride=1)\n";
        } else if (args.KH == 1 && args.KW == 1) {
            if (args.use_our) {
                const auto dot = bench_our_conv_1x1(args, false, false);
                const auto block4 = bench_our_conv_1x1(args, true, false);
                const auto brgemm = bench_our_conv_1x1_brgemm(args);
                std::cout << "  our_conv1x1 : " << dot.ms << " ms, " << dot.gops << " GOPS\n";
                std::cout << "  our_conv1x1_block4 : " << block4.ms << " ms, " << block4.gops << " GOPS\n";
                if (brgemm.ms >= 0.0) {
                    std::cout << "  our_conv1x1_brgemm4x4 : " << brgemm.ms << " ms, " << brgemm.gops
                              << " GOPS\n";
                } else {
                    std::cout << "  our_conv1x1_brgemm4x4 : unsupported (brgemm unavailable)\n";
                }
                if (has_asimd_dotprod()) {
                    const auto block4_dot = bench_our_conv_1x1(args, true, true);
                    std::cout << "  our_conv1x1_block4_dot : " << block4_dot.ms << " ms, " << block4_dot.gops
                              << " GOPS\n";
                    if (args.OC % 8 == 0) {
                        const auto block8_dot = bench_our_conv_1x1_block8_dot(args);
                        std::cout << "  our_conv1x1_block8_dot : " << block8_dot.ms << " ms, " << block8_dot.gops
                                  << " GOPS\n";
                        const auto block8_dot_packed = bench_our_conv_1x1_block8_dot_packed(args);
                        std::cout << "  our_conv1x1_block8_dot_packed : " << block8_dot_packed.ms << " ms, "
                                  << block8_dot_packed.gops << " GOPS\n";
                    } else {
                        std::cout << "  our_conv1x1_block8_dot : unsupported (OC multiple of 8)\n";
                        std::cout << "  our_conv1x1_block8_dot_packed : unsupported (OC multiple of 8)\n";
                    }
                    if (args.W % 4 == 0 && args.OC % 4 == 0) {
                        const auto block4x4_dot = bench_our_conv_1x1_block4x4_dot(args);
                        std::cout << "  our_conv1x1_block4x4_dot : " << block4x4_dot.ms << " ms, "
                                  << block4x4_dot.gops << " GOPS\n";
                        const auto block4x4_dot_packed = bench_our_conv_1x1_block4x4_dot_packed(args);
                        std::cout << "  our_conv1x1_block4x4_dot_packed : " << block4x4_dot_packed.ms << " ms, "
                                  << block4x4_dot_packed.gops << " GOPS\n";
                    } else {
                        std::cout << "  our_conv1x1_block4x4_dot : unsupported (W,OC multiple of 4)\n";
                        std::cout << "  our_conv1x1_block4x4_dot_packed : unsupported (W,OC multiple of 4)\n";
                    }
                }
                if (has_i8mm()) {
                    if (args.W % 4 == 0 && args.OC % 4 == 0 && args.IC % 8 == 0) {
                        const auto block4x4_mmla = bench_our_conv_1x1_block4x4_mmla_packed(args);
                        std::cout << "  our_conv1x1_block4x4_mmla_packed : " << block4x4_mmla.ms << " ms, "
                                  << block4x4_mmla.gops << " GOPS\n";
                    } else {
                        std::cout << "  our_conv1x1_block4x4_mmla_packed : unsupported (W,OC multiple of 4, IC multiple of 8)\n";
                    }
                    if (args.W % 4 == 0 && args.OC % 8 == 0 && args.IC % 8 == 0) {
                        const auto block4x8_mmla = bench_our_conv_1x1_block4x8_mmla_packed(args);
                        std::cout << "  our_conv1x1_block4x8_mmla_packed : " << block4x8_mmla.ms << " ms, "
                                  << block4x8_mmla.gops << " GOPS\n";
                    } else {
                        std::cout << "  our_conv1x1_block4x8_mmla_packed : unsupported (W multiple of 4, OC multiple of 8, IC multiple of 8)\n";
                    }
                    if (args.W % 4 == 0 && args.OC % 16 == 0 && args.IC % 8 == 0) {
                        const auto block4x16_mmla = bench_our_conv_1x1_block4x16_mmla_packed(args);
                        std::cout << "  our_conv1x1_block4x16_mmla_packed : " << block4x16_mmla.ms << " ms, "
                                  << block4x16_mmla.gops << " GOPS\n";
                    } else {
                        std::cout << "  our_conv1x1_block4x16_mmla_packed : unsupported (W multiple of 4, OC multiple of 16, IC multiple of 8)\n";
                    }
                }
            }
        } else if ((args.KH == args.KW) && (args.KH == 3 || args.KH == 5)) {
            if (args.use_our) {
                const auto conv_kxk = bench_our_conv_kxk(args);
                const auto conv_kxk_mmla4 = bench_our_conv_kxk_mmla_packed(args, 4);
                const auto conv_kxk_mmla8 = bench_our_conv_kxk_mmla_packed(args, 8);
                const auto conv_kxk_mmla16 = bench_our_conv_kxk_mmla_packed(args, 16);
                const auto conv_kxk_mmla4_fused = bench_our_conv_kxk_mmla_packed_fused(args, 4);
                const auto conv_kxk_mmla8_fused = bench_our_conv_kxk_mmla_packed_fused(args, 8);
                const auto conv_kxk_mmla16_fused = bench_our_conv_kxk_mmla_packed_fused(args, 16);
                const auto conv_kxk_brgemm = bench_our_conv_kxk_brgemm(args);
                std::cout << "  our_conv" << args.KH << "x" << args.KW << " : " << conv_kxk.ms << " ms, "
                          << conv_kxk.gops << " GOPS\n";
                if (conv_kxk_mmla4.ms >= 0.0) {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x4 : " << conv_kxk_mmla4.ms
                              << " ms, " << conv_kxk_mmla4.gops << " GOPS\n";
                } else {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla4x4 : unsupported (OC multiple of 4, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_mmla8.ms >= 0.0) {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x8 : " << conv_kxk_mmla8.ms
                              << " ms, " << conv_kxk_mmla8.gops << " GOPS\n";
                } else {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla4x8 : unsupported (OC multiple of 8, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_mmla16.ms >= 0.0) {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x16 : " << conv_kxk_mmla16.ms
                              << " ms, " << conv_kxk_mmla16.gops << " GOPS\n";
                } else {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla4x16 : unsupported (OC multiple of 16, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_mmla4_fused.ms >= 0.0) {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x4_fused : "
                              << conv_kxk_mmla4_fused.ms << " ms, " << conv_kxk_mmla4_fused.gops << " GOPS\n";
                } else {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla4x4_fused : unsupported (OC multiple of 4, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_mmla8_fused.ms >= 0.0) {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x8_fused : "
                              << conv_kxk_mmla8_fused.ms << " ms, " << conv_kxk_mmla8_fused.gops << " GOPS\n";
                } else {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla4x8_fused : unsupported (OC multiple of 8, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_mmla16_fused.ms >= 0.0) {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x16_fused : "
                              << conv_kxk_mmla16_fused.ms << " ms, " << conv_kxk_mmla16_fused.gops << " GOPS\n";
                } else {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla4x16_fused : unsupported (OC multiple of 16, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_brgemm.ms >= 0.0) {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW << "_brgemm4x4 : " << conv_kxk_brgemm.ms
                              << " ms, " << conv_kxk_brgemm.gops << " GOPS\n";
                } else {
                    std::cout << "  our_conv" << args.KH << "x" << args.KW
                              << "_brgemm4x4 : unsupported (brgemm unavailable)\n";
                }
            }
        } else {
            std::cout << "  our_conv    : unsupported (only 1x1, 3x3, 5x5)\n";
        }
#if defined(OV_CPU_WITH_ACL)
        if (args.use_acl) {
            try {
                const auto acl = bench_acl_conv_1x1(args);
                std::cout << "  acl_conv    : " << acl.ms << " ms, " << acl.gops << " GOPS\n";
            } catch (const std::exception& ex) {
                std::cout << "  acl_conv    : failed (" << ex.what() << ")\n";
            }
        }
#else
        if (args.use_acl) {
            std::cout << "  acl_conv    : unavailable (built without ACL)\n";
        }
#endif
    }

    return 0;
}
