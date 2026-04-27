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
#include <sstream>
#include <string>
#include <vector>

#include <cpu/aarch64/brgemm/brgemm.hpp>
#include "utils/precision_support.h"
#include "nodes/kernels/aarch64/brgemm_int8_kernel.hpp"
#include "nodes/kernels/aarch64/brgemm_kernels/int8_brgemm_kernels.hpp"
#if defined(OV_CPU_WITH_ACL)
#    include <arm_compute/core/QuantizationInfo.h>
#    include <arm_compute/core/TensorInfo.h>
#    include <arm_compute/core/Types.h>
#    include <arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h>
#    include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>
#    include <arm_compute/runtime/Scheduler.h>
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
    bool pack_only = false;
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
    size_t pad = 0;
    size_t dilation = 1;
    size_t groups = 1;
    size_t warmup = 5;
    size_t iters = 50;
};

size_t conv_out_dim(size_t in, size_t k, size_t stride, size_t pad, size_t dilation) {
    const size_t eff_k = dilation * (k - 1) + 1;
    const size_t padded = in + 2 * pad;
    if (padded < eff_k) {
        return 0;
    }
    return (padded - eff_k) / stride + 1;
}

bool has_asimd_dotprod() {
    return ov::intel_cpu::hasIntDotProductSupport();
}

bool has_i8mm() {
    return ov::intel_cpu::hasInt8MMSupport();
}

bool has_sve() {
    switch (ov::intel_cpu::getAarch64Int8Isa()) {
    case ov::intel_cpu::Aarch64Int8Isa::sve:
    case ov::intel_cpu::Aarch64Int8Isa::sve_i8mm:
    case ov::intel_cpu::Aarch64Int8Isa::sve2_i8mm:
        return true;
    default:
        return false;
    }
}

size_t packed_block_stride(size_t K, size_t oc_block) {
    const size_t k_blocks = K / 16;
    const size_t k_tail = K % 16;
    return k_blocks * oc_block * 16 + k_tail * oc_block;
}

size_t round_up(size_t value, size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

template <typename T>
T* align_ptr(T* ptr, size_t alignment) {
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    const auto aligned = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<T*>(aligned);
}

size_t packed_block_stride_dot_interleaved(size_t K, size_t oc_block) {
    const size_t Kp = round_up(K, 4);
    return (Kp / 4) * oc_block * 4;
}

size_t packed_block_stride_mmla(size_t K, size_t oc_block) {
    const size_t Kp = round_up(K, 8);
    return (Kp / 8) * oc_block * 8;
}

std::vector<int32_t> build_wei_comp_1x1(const std::vector<int8_t>& wei, size_t OC, size_t IC) {
    std::vector<int32_t> comp(OC, 0);
    for (size_t oc = 0; oc < OC; ++oc) {
        const int8_t* row = wei.data() + oc * IC;
        int32_t sum = 0;
        for (size_t ic = 0; ic < IC; ++ic) {
            sum += static_cast<int32_t>(row[ic]);
        }
        comp[oc] = sum;
    }
    return comp;
}

std::vector<int32_t> build_bias_comp_1x1(const std::vector<int8_t>& wei, size_t OC, size_t IC) {
    auto comp = build_wei_comp_1x1(wei, OC, IC);
    for (auto& v : comp) {
        v *= 128;
    }
    return comp;
}

inline void add_1x1_comp_block(int32_t* dst_ptr, size_t rows, size_t row_stride, const int32_t* comp, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        int32_t* dst_row = dst_ptr + r * row_stride;
        for (size_t c = 0; c < cols; ++c) {
            dst_row[c] += comp[c];
        }
    }
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

void pack_dot_block_interleaved4(const int8_t* src, size_t K, size_t oc_block, int8_t* dst) {
    const size_t k_blocks = round_up(K, 4) / 4;
    size_t offset = 0;
    for (size_t kb = 0; kb < k_blocks; ++kb) {
        const size_t k_off = kb * 4;
        for (size_t oc = 0; oc < oc_block; oc += 4) {
            for (size_t lane = 0; lane < 4; ++lane) {
                const size_t oc_idx = oc + lane;
                for (size_t t = 0; t < 4; ++t) {
                    const size_t k = k_off + t;
                    const size_t dst_idx = offset + lane * 4 + t;
                    if (oc_idx < oc_block && k < K) {
                        dst[dst_idx] = src[oc_idx * K + k];
                    } else {
                        dst[dst_idx] = 0;
                    }
                }
            }
            offset += 16;
        }
    }
}

void pack_lhs_row_dot4x16_interleaved4(const uint8_t* src, size_t K, uint8_t* dst) {
    const size_t k_blocks = round_up(K, 4) / 4;
    for (size_t kb = 0; kb < k_blocks; ++kb) {
        const size_t k_base = kb * 4;
        const uint8_t v0 = k_base < K ? src[k_base] : 0;
        const uint8_t v1 = (k_base + 1) < K ? src[k_base + 1] : 0;
        const uint8_t v2 = (k_base + 2) < K ? src[k_base + 2] : 0;
        const uint8_t v3 = (k_base + 3) < K ? src[k_base + 3] : 0;
        const uint32_t r0 = static_cast<uint32_t>(v0) * 0x01010101u;
        const uint32_t r1 = static_cast<uint32_t>(v1) * 0x01010101u;
        const uint32_t r2 = static_cast<uint32_t>(v2) * 0x01010101u;
        const uint32_t r3 = static_cast<uint32_t>(v3) * 0x01010101u;
        std::memcpy(dst, &r0, sizeof(r0));
        std::memcpy(dst + 4, &r1, sizeof(r1));
        std::memcpy(dst + 8, &r2, sizeof(r2));
        std::memcpy(dst + 12, &r3, sizeof(r3));
        dst += 16;
    }
}

void pack_lhs_4x16_dot_interleaved4(const uint8_t* src0,
                                   const uint8_t* src1,
                                   const uint8_t* src2,
                                   const uint8_t* src3,
                                   size_t K,
                                   uint8_t* dst,
                                   size_t dst_stride) {
    pack_lhs_row_dot4x16_interleaved4(src0, K, dst);
    pack_lhs_row_dot4x16_interleaved4(src1, K, dst + dst_stride);
    pack_lhs_row_dot4x16_interleaved4(src2, K, dst + 2 * dst_stride);
    pack_lhs_row_dot4x16_interleaved4(src3, K, dst + 3 * dst_stride);
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
              << "  --pack-only                   Measure fused conv packing only\n"
              << "  --signed-src                   Use signed src (i8). Default: u8\n"
              << "  --m/--n/--k                    GEMM sizes (default 128x128x256)\n"
              << "  --n/--h/--w/--ic/--oc/--kh/--kw Conv sizes (default N1 H28 W28 IC64 OC64 KH1 KW1)\n"
              << "  --stride                       Conv stride (default 1)\n"
              << "  --pad                          Conv symmetric padding (default 0)\n"
              << "  --dilation                     Conv dilation (default 1)\n"
              << "  --groups                       Conv groups (default 1)\n"
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
        } else if (arg == "--pack-only") {
            args.pack_only = true;
        } else if (arg == "--signed-src") {
            args.src_signed = true;
        } else if (parse_size_arg(arg, "--m=", args.M) || parse_size_arg(arg, "--n=", args.N) ||
                   parse_size_arg(arg, "--k=", args.K) || parse_size_arg(arg, "--N=", args.Nn) ||
                   parse_size_arg(arg, "--H=", args.H) || parse_size_arg(arg, "--W=", args.W) ||
                   parse_size_arg(arg, "--IC=", args.IC) || parse_size_arg(arg, "--OC=", args.OC) ||
                   parse_size_arg(arg, "--KH=", args.KH) || parse_size_arg(arg, "--KW=", args.KW) ||
                   parse_size_arg(arg, "--stride=", args.stride) || parse_size_arg(arg, "--pad=", args.pad) ||
                   parse_size_arg(arg, "--dilation=", args.dilation) || parse_size_arg(arg, "--groups=", args.groups) ||
                   parse_size_arg(arg, "--warmup=", args.warmup) ||
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
    if (args.pack_only) {
        args.run_gemm = false;
        args.run_conv = true;
        args.use_acl = false;
        args.use_kleidiai = false;
    }
    if (args.stride == 0 || args.dilation == 0) {
        std::cerr << "Invalid conv args: stride and dilation must be >= 1\n";
        std::exit(1);
    }
    if (args.groups == 0) {
        std::cerr << "Invalid conv args: groups must be >= 1\n";
        std::exit(1);
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

std::string src_pair_tag(bool src_signed) {
    return src_signed ? "s8s8" : "u8s8";
}

bool starts_with(const std::string& value, const char* prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::string canonical_bench_label(const std::string& raw, bool src_signed) {
    if (raw.find('[') != std::string::npos) {
        return raw;
    }
    if (starts_with(raw, "acl_")) {
        return "acl::" + raw.substr(4) + " [" + raw + "]";
    }
    if (starts_with(raw, "kleidiai_")) {
        return "kleidiai::" + raw.substr(9) + " [" + raw + "]";
    }
    if (!starts_with(raw, "our_")) {
        return raw;
    }

    if (raw == "our_brgemm4x4") {
        return "brgemm_wrapper_runtime_dispatch [" + raw + "]";
    }
    if (raw.find("_brgemm") != std::string::npos) {
        return "brgemm_wrapper::" + raw.substr(4) + " [" + raw + "]";
    }

    std::string family = "aarch64_neon_mla";
    if (raw.find("mmla") != std::string::npos) {
        family = "aarch64_neon_i8mm";
    } else if (raw.find("dot") != std::string::npos) {
        family = "aarch64_neon_dotprod";
    }

    std::string body = raw.substr(4);
    const auto replace_all = [](std::string& value, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = value.find(from, pos)) != std::string::npos) {
            value.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replace_all(body, "_pack_only", "_pack");
    replace_all(body, "_exec_u8", "_exec_bias_comp");
    return family + "_" + src_pair_tag(src_signed) + "_" + body + " [" + raw + "]";
}

std::string rewrite_benchmark_report(const std::string& report, bool src_signed) {
    std::istringstream input(report);
    std::ostringstream output;
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("  ", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                std::string raw_label = line.substr(2, colon - 2);
                while (!raw_label.empty() && raw_label.back() == ' ') {
                    raw_label.pop_back();
                }
                output << "  " << canonical_bench_label(raw_label, src_signed) << line.substr(colon);
                if (!input.eof()) {
                    output << '\n';
                }
                continue;
            }
        }
        output << line;
        if (!input.eof()) {
            output << '\n';
        }
    }
    return output.str();
}

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

TimerResult bench_our_gemm_block8x8_mmla_packed(const Args& args) {
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

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x8_smmla_packed block8x8_smmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x8_usmmla_packed block8x8_usmmla;
    block8x8_smmla.create_ker();
    block8x8_usmmla.create_ker();
    const auto block8x8_smmla_ker = block8x8_smmla.ker();
    const auto block8x8_usmmla_ker = block8x8_usmmla.ker();

    auto run = [&]() {
        for (size_t m = 0; m + 8 <= M; m += 8) {
            const uint8_t* a_ptrs_u8[8] = {
                A.data() + (m + 0) * K,
                A.data() + (m + 1) * K,
                A.data() + (m + 2) * K,
                A.data() + (m + 3) * K,
                A.data() + (m + 4) * K,
                A.data() + (m + 5) * K,
                A.data() + (m + 6) * K,
                A.data() + (m + 7) * K,
            };
            const int8_t* a_ptrs_s8[8] = {
                reinterpret_cast<const int8_t*>(a_ptrs_u8[0]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[1]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[2]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[3]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[4]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[5]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[6]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[7]),
            };
            for (size_t n = 0; n + 8 <= N; n += 8) {
                const int8_t* b_ptr = B_packed.data() + (n / 8) * stride8;
                int32_t* c_ptr = C.data() + m * N + n;
                if (args.src_signed) {
                    block8x8_smmla_ker(a_ptrs_s8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                } else {
                    block8x8_usmmla_ker(a_ptrs_u8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
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

TimerResult bench_our_gemm_block8x12_mmla_packed(const Args& args) {
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
    const size_t n_blocks = N / 12;
    const size_t stride12 = packed_block_stride_mmla(K, 12);
    std::vector<int8_t> B_packed(n_blocks * stride12);
    for (size_t nb = 0; nb < n_blocks; ++nb) {
        const int8_t* src_block = B.data() + nb * 12 * K;
        int8_t* dst_block = B_packed.data() + nb * stride12;
        pack_mmla_block(src_block, K, Kp, 12, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x12_smmla_packed block8x12_smmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x12_usmmla_packed block8x12_usmmla;
    block8x12_smmla.create_ker();
    block8x12_usmmla.create_ker();
    const auto block8x12_smmla_ker = block8x12_smmla.ker();
    const auto block8x12_usmmla_ker = block8x12_usmmla.ker();

    auto run = [&]() {
        for (size_t m = 0; m + 8 <= M; m += 8) {
            const uint8_t* a_ptrs_u8[8] = {
                A.data() + (m + 0) * K,
                A.data() + (m + 1) * K,
                A.data() + (m + 2) * K,
                A.data() + (m + 3) * K,
                A.data() + (m + 4) * K,
                A.data() + (m + 5) * K,
                A.data() + (m + 6) * K,
                A.data() + (m + 7) * K,
            };
            const int8_t* a_ptrs_s8[8] = {
                reinterpret_cast<const int8_t*>(a_ptrs_u8[0]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[1]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[2]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[3]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[4]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[5]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[6]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[7]),
            };
            for (size_t n = 0; n + 12 <= N; n += 12) {
                const int8_t* b_ptr = B_packed.data() + (n / 12) * stride12;
                int32_t* c_ptr = C.data() + m * N + n;
                if (args.src_signed) {
                    block8x12_smmla_ker(a_ptrs_s8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                } else {
                    block8x12_usmmla_ker(a_ptrs_u8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
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

TimerResult bench_our_gemm_mmla_tuned(const Args& args) {
    if (!has_i8mm() || (args.K % 8 != 0)) {
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

    const size_t Kp = round_up(K, 8);
    const bool use_s8 = args.src_signed;

    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(use_s8);
    dot.create_ker();
    const auto dot_ker = dot.ker();

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x8_smmla_packed ker8x8_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x8_usmmla_packed ker8x8_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x12_smmla_packed ker8x12_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x12_usmmla_packed ker8x12_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_smmla_packed ker4x16_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_usmmla_packed ker4x16_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_smmla_packed ker4x8_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_usmmla_packed ker4x8_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_smmla_packed ker4x4_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_usmmla_packed ker4x4_u8;
    ker8x8_s8.create_ker();
    ker8x8_u8.create_ker();
    ker8x12_s8.create_ker();
    ker8x12_u8.create_ker();
    ker4x16_s8.create_ker();
    ker4x16_u8.create_ker();
    ker4x8_s8.create_ker();
    ker4x8_u8.create_ker();
    ker4x4_s8.create_ker();
    ker4x4_u8.create_ker();

    const size_t n12_blocks = N / 12;
    const size_t n12_main = n12_blocks * 12;
    const size_t rem12 = N - n12_main;
    const bool use_block12 = (n12_blocks > 0);
    const size_t stride8 = packed_block_stride_mmla(K, 8);
    const size_t stride12 = packed_block_stride_mmla(K, 12);
    const size_t stride16 = packed_block_stride_mmla(K, 16);
    const size_t stride4 = packed_block_stride_mmla(K, 4);
    if (stride8 == 0 || stride4 == 0) {
        return {-1.0, -1.0};
    }

    std::vector<int8_t> B_packed8_main;
    std::vector<int8_t> B_packed12;
    std::vector<int8_t> B_packed8_tail12;
    std::vector<int8_t> B_packed4_tail12;
    std::vector<int8_t> B_packed16_tail;
    std::vector<int8_t> B_packed8_tail;
    std::vector<int8_t> B_packed4_tail;

    if (use_block12) {
        B_packed12.resize(n12_blocks * stride12);
        for (size_t nb = 0; nb < n12_blocks; ++nb) {
            const int8_t* src_block = B.data() + nb * 12 * K;
            int8_t* dst_block = B_packed12.data() + nb * stride12;
            pack_mmla_block(src_block, K, Kp, 12, dst_block);
        }
    } else {
        const size_t n_blocks8 = N / 8;
        B_packed8_main.resize(n_blocks8 * stride8);
        for (size_t nb = 0; nb < n_blocks8; ++nb) {
            const int8_t* src_block = B.data() + nb * 8 * K;
            int8_t* dst_block = B_packed8_main.data() + nb * stride8;
            pack_mmla_block(src_block, K, Kp, 8, dst_block);
        }
    }
    if (rem12 >= 8) {
        B_packed8_tail12.resize(stride8);
        const int8_t* src_block = B.data() + n12_main * K;
        pack_mmla_block(src_block, K, Kp, 8, B_packed8_tail12.data());
    }
    if (rem12 >= 4) {
        const size_t tail4_off = n12_main + ((rem12 >= 8) ? 8 : 0);
        B_packed4_tail12.resize(stride4);
        const int8_t* src_block = B.data() + tail4_off * K;
        pack_mmla_block(src_block, K, Kp, 4, B_packed4_tail12.data());
    }

    const size_t n_blocks16 = N / 16;
    const size_t n_blocks8_tail = (N % 16) / 8;
    const size_t n_blocks4_tail = (N % 8) / 4;
    if (n_blocks16 > 0) {
        B_packed16_tail.resize(n_blocks16 * stride16);
        for (size_t nb = 0; nb < n_blocks16; ++nb) {
            const int8_t* src_block = B.data() + nb * 16 * K;
            int8_t* dst_block = B_packed16_tail.data() + nb * stride16;
            pack_mmla_block(src_block, K, Kp, 16, dst_block);
        }
    }
    if (n_blocks8_tail > 0) {
        B_packed8_tail.resize(n_blocks8_tail * stride8);
        for (size_t nb = 0; nb < n_blocks8_tail; ++nb) {
            const int8_t* src_block = B.data() + (n_blocks16 * 16 + nb * 8) * K;
            int8_t* dst_block = B_packed8_tail.data() + nb * stride8;
            pack_mmla_block(src_block, K, Kp, 8, dst_block);
        }
    }
    if (n_blocks4_tail > 0) {
        B_packed4_tail.resize(n_blocks4_tail * stride4);
        for (size_t nb = 0; nb < n_blocks4_tail; ++nb) {
            const int8_t* src_block = B.data() + (n_blocks16 * 16 + n_blocks8_tail * 8 + nb * 4) * K;
            int8_t* dst_block = B_packed4_tail.data() + nb * stride4;
            pack_mmla_block(src_block, K, Kp, 4, dst_block);
        }
    }

    auto run = [&]() {
        size_t m = 0;
        for (; m + 8 <= M; m += 8) {
            const uint8_t* a_ptrs_u8[8] = {
                A.data() + (m + 0) * K,
                A.data() + (m + 1) * K,
                A.data() + (m + 2) * K,
                A.data() + (m + 3) * K,
                A.data() + (m + 4) * K,
                A.data() + (m + 5) * K,
                A.data() + (m + 6) * K,
                A.data() + (m + 7) * K,
            };
            const int8_t* a_ptrs_s8[8] = {
                reinterpret_cast<const int8_t*>(a_ptrs_u8[0]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[1]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[2]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[3]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[4]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[5]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[6]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8[7]),
            };
            const uint8_t* a_ptrs_u8_lo[4] = {
                a_ptrs_u8[0],
                a_ptrs_u8[1],
                a_ptrs_u8[2],
                a_ptrs_u8[3],
            };
            const uint8_t* a_ptrs_u8_hi[4] = {
                a_ptrs_u8[4],
                a_ptrs_u8[5],
                a_ptrs_u8[6],
                a_ptrs_u8[7],
            };
            const int8_t* a_ptrs_s8_lo[4] = {
                reinterpret_cast<const int8_t*>(a_ptrs_u8_lo[0]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8_lo[1]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8_lo[2]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8_lo[3]),
            };
            const int8_t* a_ptrs_s8_hi[4] = {
                reinterpret_cast<const int8_t*>(a_ptrs_u8_hi[0]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8_hi[1]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8_hi[2]),
                reinterpret_cast<const int8_t*>(a_ptrs_u8_hi[3]),
            };

            if (use_block12) {
                for (size_t n = 0; n + 12 <= n12_main; n += 12) {
                    const int8_t* b_ptr = B_packed12.data() + (n / 12) * stride12;
                    int32_t* c_ptr = C.data() + m * N + n;
                    if (use_s8) {
                        ker8x12_s8.ker()(a_ptrs_s8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                    } else {
                        ker8x12_u8.ker()(a_ptrs_u8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                    }
                }
                if (rem12 >= 8) {
                    const int8_t* b_ptr = B_packed8_tail12.data();
                    int32_t* c_ptr = C.data() + m * N + n12_main;
                    if (use_s8) {
                        ker8x8_s8.ker()(a_ptrs_s8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                    } else {
                        ker8x8_u8.ker()(a_ptrs_u8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                    }
                }
                if (rem12 >= 4) {
                    const size_t tail4_off = n12_main + ((rem12 >= 8) ? 8 : 0);
                    const int8_t* b_ptr = B_packed4_tail12.data();
                    int32_t* c_ptr0 = C.data() + m * N + tail4_off;
                    int32_t* c_ptr1 = C.data() + (m + 4) * N + tail4_off;
                    if (use_s8) {
                        ker4x4_s8.ker()(a_ptrs_s8_lo, b_ptr, c_ptr0, K, 0, N * sizeof(int32_t), 0);
                        ker4x4_s8.ker()(a_ptrs_s8_hi, b_ptr, c_ptr1, K, 0, N * sizeof(int32_t), 0);
                    } else {
                        ker4x4_u8.ker()(a_ptrs_u8_lo, b_ptr, c_ptr0, K, 0, N * sizeof(int32_t), 0);
                        ker4x4_u8.ker()(a_ptrs_u8_hi, b_ptr, c_ptr1, K, 0, N * sizeof(int32_t), 0);
                    }
                }
            } else {
                for (size_t n = 0; n + 8 <= N; n += 8) {
                    const int8_t* b_ptr = B_packed8_main.data() + (n / 8) * stride8;
                    int32_t* c_ptr = C.data() + m * N + n;
                    if (use_s8) {
                        ker8x8_s8.ker()(a_ptrs_s8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                    } else {
                        ker8x8_u8.ker()(a_ptrs_u8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                    }
                }
                if ((N % 8) >= 4) {
                    const size_t tail4_off = (N / 8) * 8;
                    const int8_t* b_ptr = B_packed4_tail12.data();
                    int32_t* c_ptr0 = C.data() + m * N + tail4_off;
                    int32_t* c_ptr1 = C.data() + (m + 4) * N + tail4_off;
                    if (use_s8) {
                        ker4x4_s8.ker()(a_ptrs_s8_lo, b_ptr, c_ptr0, K, 0, N * sizeof(int32_t), 0);
                        ker4x4_s8.ker()(a_ptrs_s8_hi, b_ptr, c_ptr1, K, 0, N * sizeof(int32_t), 0);
                    } else {
                        ker4x4_u8.ker()(a_ptrs_u8_lo, b_ptr, c_ptr0, K, 0, N * sizeof(int32_t), 0);
                        ker4x4_u8.ker()(a_ptrs_u8_hi, b_ptr, c_ptr1, K, 0, N * sizeof(int32_t), 0);
                    }
                }
            }
        }

        for (; m + 4 <= M; m += 4) {
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

            size_t n = 0;
            for (; n + 16 <= N; n += 16) {
                const int8_t* b_ptr = B_packed16_tail.data() + (n / 16) * stride16;
                int32_t* c_ptr = C.data() + m * N + n;
                if (use_s8) {
                    ker4x16_s8.ker()(a_ptrs_s8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                } else {
                    ker4x16_u8.ker()(a_ptrs_u8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                }
            }
            for (; n + 8 <= N; n += 8) {
                const size_t idx = (n - n_blocks16 * 16) / 8;
                const int8_t* b_ptr = B_packed8_tail.data() + idx * stride8;
                int32_t* c_ptr = C.data() + m * N + n;
                if (use_s8) {
                    ker4x8_s8.ker()(a_ptrs_s8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                } else {
                    ker4x8_u8.ker()(a_ptrs_u8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                }
            }
            for (; n + 4 <= N; n += 4) {
                const size_t tail4_idx = (n - n_blocks16 * 16 - n_blocks8_tail * 8) / 4;
                const int8_t* b_ptr = B_packed4_tail.data() + tail4_idx * stride4;
                int32_t* c_ptr = C.data() + m * N + n;
                if (use_s8) {
                    ker4x4_s8.ker()(a_ptrs_s8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                } else {
                    ker4x4_u8.ker()(a_ptrs_u8, b_ptr, c_ptr, K, 0, N * sizeof(int32_t), 0);
                }
            }
        }

        for (; m < M; ++m) {
            const uint8_t* a_ptr_u8 = A.data() + m * K;
            for (size_t n = 0; n < N; ++n) {
                int32_t* c_ptr = C.data() + m * N + n;
                const int8_t* b_ptr = B.data() + n * K;
                if (use_s8) {
                    dot_ker(a_ptr_u8, b_ptr, c_ptr, K, 0);
                } else {
                    dot_ker(a_ptr_u8, b_ptr, c_ptr, K, 0);
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

TimerResult bench_our_conv_1x1_brgemm_mnb(const Args& args, size_t mb, size_t nb) {
    if (!has_sve() || mb == 0 || nb == 0) {
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
        ov::intel_cpu::aarch64::BrgemmInt8Kernel brg(mb, nb, IC, IC, OC, OC, args.src_signed);
        ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
        dot.create_ker();
        const auto dot_ker = dot.ker();
        const bool use_onednn_brgemm = brg.uses_brgemm();

        auto run = [&]() {
            const size_t w_full = (W / mb) * mb;
            const size_t oc_full = (OC / nb) * nb;
            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    size_t w = 0;
                    for (; w < w_full; w += mb) {
                        const uint8_t* src_block = src.data() + ((n * H + h) * W + w) * IC;
                        int32_t* dst_block = dst.data() + ((n * H + h) * W + w) * OC;
                        for (size_t oc = 0; oc < oc_full; oc += nb) {
                            const int8_t* wei_block =
                                use_onednn_brgemm ? (packed_wei.data() + oc) : (wei.data() + oc * IC);
                            brg.execute(src_block, wei_block, dst_block + oc);
                        }
                        for (size_t oc = oc_full; oc < OC; ++oc) {
                            const int8_t* wei_ptr = wei.data() + oc * IC;
                            for (size_t mi = 0; mi < mb; ++mi) {
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

TimerResult bench_our_conv_1x1_block2x8_dot_packed(const Args& args) {
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

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_2x8_dot_packed_strided block2x8_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_2x8_udot_packed_strided block2x8_udot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_dot_packed block8_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_udot_packed block8_udot;
    block2x8_dot.create_ker();
    block2x8_udot.create_ker();
    block8_dot.create_ker();
    block8_udot.create_ker();
    const auto block2x8_dot_ker = block2x8_dot.ker();
    const auto block2x8_udot_ker = block2x8_udot.ker();
    const auto block8_dot_ker = block8_dot.ker();
    const auto block8_udot_ker = block8_udot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                size_t w = 0;
                for (; w + 2 <= W; w += 2) {
                    const size_t src_off = ((n * H + h) * W + w) * IC;
                    const uint8_t* src_ptr0 = src.data() + src_off;
                    const int8_t* src_ptr0_s8 = reinterpret_cast<const int8_t*>(src_ptr0);
                    const size_t dst_off = ((n * H + h) * W + w) * OC;
                    int32_t* dst_ptr = dst.data() + dst_off;
                    for (size_t oc = 0; oc + 8 <= OC; oc += 8) {
                        const int8_t* wei_ptr = wei_packed.data() + (oc / 8) * stride8;
                        if (args.src_signed) {
                            block2x8_dot_ker(src_ptr0_s8,
                                             wei_ptr,
                                             dst_ptr + oc,
                                             IC,
                                             IC,
                                             OC * sizeof(int32_t),
                                             0);
                        } else {
                            block2x8_udot_ker(src_ptr0, wei_ptr, dst_ptr + oc, IC, IC, OC * sizeof(int32_t), 0);
                        }
                    }
                }
                if (w < W) {
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

TimerResult bench_our_conv_1x1_block2x8_dot_packed_interleaved4(const Args& args) {
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
    const size_t stride8_i4 = packed_block_stride_dot_interleaved(IC, 8);
    std::vector<int8_t> wei_packed(oc_blocks * stride8);
    std::vector<int8_t> wei_packed_i4(oc_blocks * stride8_i4);
    for (size_t ocb = 0; ocb < oc_blocks; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 8 * IC;
        int8_t* dst_block = wei_packed.data() + ocb * stride8;
        int8_t* dst_block_i4 = wei_packed_i4.data() + ocb * stride8_i4;
        pack_dot_block(src_block, IC, 8, dst_block);
        pack_dot_block_interleaved4(src_block, IC, 8, dst_block_i4);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_2x8_dot_packed_strided_interleaved4 block2x8_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_2x8_udot_packed_strided_interleaved4 block2x8_udot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_dot_packed block8_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_udot_packed block8_udot;
    block2x8_dot.create_ker();
    block2x8_udot.create_ker();
    block8_dot.create_ker();
    block8_udot.create_ker();
    const auto block2x8_dot_ker = block2x8_dot.ker();
    const auto block2x8_udot_ker = block2x8_udot.ker();
    const auto block8_dot_ker = block8_dot.ker();
    const auto block8_udot_ker = block8_udot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                size_t w = 0;
                for (; w + 2 <= W; w += 2) {
                    const size_t src_off = ((n * H + h) * W + w) * IC;
                    const uint8_t* src_ptr0 = src.data() + src_off;
                    const int8_t* src_ptr0_s8 = reinterpret_cast<const int8_t*>(src_ptr0);
                    const size_t dst_off = ((n * H + h) * W + w) * OC;
                    int32_t* dst_ptr = dst.data() + dst_off;
                    for (size_t oc = 0; oc + 8 <= OC; oc += 8) {
                        const int8_t* wei_ptr = wei_packed_i4.data() + (oc / 8) * stride8_i4;
                        if (args.src_signed) {
                            block2x8_dot_ker(src_ptr0_s8,
                                             wei_ptr,
                                             dst_ptr + oc,
                                             IC,
                                             IC,
                                             OC * sizeof(int32_t),
                                             0);
                        } else {
                            block2x8_udot_ker(src_ptr0, wei_ptr, dst_ptr + oc, IC, IC, OC * sizeof(int32_t), 0);
                        }
                    }
                }
                if (w < W) {
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

TimerResult bench_our_conv_1x1_block2x16_dot_packed_interleaved4(const Args& args, bool executor_like_u8 = false) {
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
    const auto wei_comp = executor_like_u8 && !args.src_signed ? build_wei_comp_1x1(wei, OC, IC) : std::vector<int32_t>{};

    const size_t oc_blocks16 = OC / 16;
    const size_t oc_blocks8 = OC / 8;
    const size_t stride8 = packed_block_stride(IC, 8);
    const size_t stride16_i4 = packed_block_stride_dot_interleaved(IC, 16);
    std::vector<int8_t> wei_packed(oc_blocks8 * stride8);
    std::vector<int8_t> wei_packed_i4(oc_blocks16 * stride16_i4);
    for (size_t ocb = 0; ocb < oc_blocks8; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 8 * IC;
        int8_t* dst_block = wei_packed.data() + ocb * stride8;
        pack_dot_block(src_block, IC, 8, dst_block);
    }
    for (size_t ocb = 0; ocb < oc_blocks16; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 16 * IC;
        int8_t* dst_block_i4 = wei_packed_i4.data() + ocb * stride16_i4;
        pack_dot_block_interleaved4(src_block, IC, 16, dst_block_i4);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_2x16_dot_packed_strided_interleaved4 block2x16_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_2x16_udot_packed_strided_interleaved4 block2x16_udot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_dot_packed block8_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_udot_packed block8_udot;
    block2x16_dot.create_ker();
    block2x16_udot.create_ker();
    block8_dot.create_ker();
    block8_udot.create_ker();
    const auto block2x16_dot_ker = block2x16_dot.ker();
    const auto block2x16_udot_ker = block2x16_udot.ker();
    const auto block8_dot_ker = block8_dot.ker();
    const auto block8_udot_ker = block8_udot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                size_t w = 0;
                for (; w + 2 <= W; w += 2) {
                    const size_t src_off = ((n * H + h) * W + w) * IC;
                    const uint8_t* src_ptr0 = src.data() + src_off;
                    const int8_t* src_ptr0_s8 = reinterpret_cast<const int8_t*>(src_ptr0);
                    const size_t dst_off = ((n * H + h) * W + w) * OC;
                    int32_t* dst_ptr = dst.data() + dst_off;
                    for (size_t oc = 0; oc + 16 <= OC; oc += 16) {
                        const int8_t* wei_ptr = wei_packed_i4.data() + (oc / 16) * stride16_i4;
                        if (args.src_signed) {
                            block2x16_dot_ker(src_ptr0_s8,
                                              wei_ptr,
                                              dst_ptr + oc,
                                              IC,
                                              IC,
                                              OC * sizeof(int32_t),
                                              0);
                        } else {
                            block2x16_udot_ker(src_ptr0, wei_ptr, dst_ptr + oc, IC, IC, OC * sizeof(int32_t), 0);
                            if (executor_like_u8) {
                                add_1x1_comp_block(dst_ptr + oc, 2, OC, wei_comp.data() + oc, 16);
                            }
                        }
                    }
                }
                if (w < W) {
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
                            if (executor_like_u8) {
                                add_1x1_comp_block(dst_ptr + oc, 1, OC, wei_comp.data() + oc, 8);
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

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(H) * static_cast<double>(W) *
                       static_cast<double>(IC) * static_cast<double>(OC);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_1x1_block2x32_dot_packed_interleaved4(const Args& args, bool executor_like_u8 = false) {
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
    const auto wei_comp = executor_like_u8 && !args.src_signed ? build_wei_comp_1x1(wei, OC, IC) : std::vector<int32_t>{};

    const size_t oc_blocks32 = OC / 32;
    const size_t oc_blocks8 = OC / 8;
    const size_t stride8 = packed_block_stride(IC, 8);
    const size_t stride32_i4 = packed_block_stride_dot_interleaved(IC, 32);
    std::vector<int8_t> wei_packed(oc_blocks8 * stride8);
    std::vector<int8_t> wei_packed_i4(oc_blocks32 * stride32_i4);
    for (size_t ocb = 0; ocb < oc_blocks8; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 8 * IC;
        int8_t* dst_block = wei_packed.data() + ocb * stride8;
        pack_dot_block(src_block, IC, 8, dst_block);
    }
    for (size_t ocb = 0; ocb < oc_blocks32; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 32 * IC;
        int8_t* dst_block_i4 = wei_packed_i4.data() + ocb * stride32_i4;
        pack_dot_block_interleaved4(src_block, IC, 32, dst_block_i4);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_2x32_dot_packed_strided_interleaved4 block2x32_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_2x32_udot_packed_strided_interleaved4 block2x32_udot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_dot_packed block8_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_udot_packed block8_udot;
    block2x32_dot.create_ker();
    block2x32_udot.create_ker();
    block8_dot.create_ker();
    block8_udot.create_ker();
    const auto block2x32_dot_ker = block2x32_dot.ker();
    const auto block2x32_udot_ker = block2x32_udot.ker();
    const auto block8_dot_ker = block8_dot.ker();
    const auto block8_udot_ker = block8_udot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                size_t w = 0;
                for (; w + 2 <= W; w += 2) {
                    const size_t src_off = ((n * H + h) * W + w) * IC;
                    const uint8_t* src_ptr0 = src.data() + src_off;
                    const int8_t* src_ptr0_s8 = reinterpret_cast<const int8_t*>(src_ptr0);
                    const size_t dst_off = ((n * H + h) * W + w) * OC;
                    int32_t* dst_ptr = dst.data() + dst_off;
                    for (size_t oc = 0; oc + 32 <= OC; oc += 32) {
                        const int8_t* wei_ptr = wei_packed_i4.data() + (oc / 32) * stride32_i4;
                        if (args.src_signed) {
                            block2x32_dot_ker(src_ptr0_s8,
                                              wei_ptr,
                                              dst_ptr + oc,
                                              IC,
                                              IC,
                                              OC * sizeof(int32_t),
                                              0);
                        } else {
                            const int32_t* bias_block = executor_like_u8 ? wei_comp.data() + oc : nullptr;
                            block2x32_udot_ker(
                                src_ptr0, wei_ptr, dst_ptr + oc, IC, IC, OC * sizeof(int32_t), bias_block, 0);
                            if (executor_like_u8 && bias_block == nullptr) {
                                add_1x1_comp_block(dst_ptr + oc, 2, OC, wei_comp.data() + oc, 32);
                            }
                        }
                    }
                }
                if (w < W) {
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
                            if (executor_like_u8) {
                                add_1x1_comp_block(dst_ptr + oc, 1, OC, wei_comp.data() + oc, 8);
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

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(H) * static_cast<double>(W) *
                       static_cast<double>(IC) * static_cast<double>(OC);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_1x1_block4x16_dot_packed_interleaved4(const Args& args, bool executor_like_u8 = false) {
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
    const auto wei_comp = executor_like_u8 && !args.src_signed ? build_wei_comp_1x1(wei, OC, IC) : std::vector<int32_t>{};

    const size_t oc_blocks16 = OC / 16;
    const size_t oc_blocks8 = OC / 8;
    const size_t stride8 = packed_block_stride(IC, 8);
    const size_t stride16_i4 = packed_block_stride_dot_interleaved(IC, 16);
    std::vector<int8_t> wei_packed(oc_blocks8 * stride8);
    std::vector<int8_t> wei_packed_i4(oc_blocks16 * stride16_i4);
    for (size_t ocb = 0; ocb < oc_blocks8; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 8 * IC;
        int8_t* dst_block = wei_packed.data() + ocb * stride8;
        pack_dot_block(src_block, IC, 8, dst_block);
    }
    for (size_t ocb = 0; ocb < oc_blocks16; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 16 * IC;
        int8_t* dst_block_i4 = wei_packed_i4.data() + ocb * stride16_i4;
        pack_dot_block_interleaved4(src_block, IC, 16, dst_block_i4);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_dot_packed_strided_interleaved4 block4x16_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_udot_packed_strided_interleaved4 block4x16_udot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_dot_packed block8_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_udot_packed block8_udot;
    block4x16_dot.create_ker();
    block4x16_udot.create_ker();
    block8_dot.create_ker();
    block8_udot.create_ker();
    const auto block4x16_dot_ker = block4x16_dot.ker();
    const auto block4x16_udot_ker = block4x16_udot.ker();
    const auto block8_dot_ker = block8_dot.ker();
    const auto block8_udot_ker = block8_udot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                size_t w = 0;
                for (; w + 4 <= W; w += 4) {
                    const size_t src_off = ((n * H + h) * W + w) * IC;
                    const uint8_t* src_ptr0 = src.data() + src_off;
                    const int8_t* src_ptr0_s8 = reinterpret_cast<const int8_t*>(src_ptr0);
                    const size_t dst_off = ((n * H + h) * W + w) * OC;
                    int32_t* dst_ptr = dst.data() + dst_off;
                    for (size_t oc = 0; oc + 16 <= OC; oc += 16) {
                        const int8_t* wei_ptr = wei_packed_i4.data() + (oc / 16) * stride16_i4;
                        if (args.src_signed) {
                            block4x16_dot_ker(src_ptr0_s8,
                                              wei_ptr,
                                              dst_ptr + oc,
                                              IC,
                                              IC,
                                              OC * sizeof(int32_t),
                                              0);
                        } else {
                            block4x16_udot_ker(src_ptr0, wei_ptr, dst_ptr + oc, IC, IC, OC * sizeof(int32_t), 0);
                            if (executor_like_u8) {
                                add_1x1_comp_block(dst_ptr + oc, 4, OC, wei_comp.data() + oc, 16);
                            }
                        }
                    }
                }
                for (; w < W; ++w) {
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
                            if (executor_like_u8) {
                                add_1x1_comp_block(dst_ptr + oc, 1, OC, wei_comp.data() + oc, 8);
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

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(H) * static_cast<double>(W) *
                       static_cast<double>(IC) * static_cast<double>(OC);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_1x1_block4x16_dot_packed_lhs_interleaved4(const Args& args) {
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

    const size_t oc_blocks16 = OC / 16;
    const size_t oc_blocks8 = OC / 8;
    const size_t stride8 = packed_block_stride(IC, 8);
    const size_t stride16_i4 = packed_block_stride_dot_interleaved(IC, 16);
    std::vector<int8_t> wei_packed(oc_blocks8 * stride8);
    std::vector<int8_t> wei_packed_i4(oc_blocks16 * stride16_i4);
    for (size_t ocb = 0; ocb < oc_blocks8; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 8 * IC;
        int8_t* dst_block = wei_packed.data() + ocb * stride8;
        pack_dot_block(src_block, IC, 8, dst_block);
    }
    for (size_t ocb = 0; ocb < oc_blocks16; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 16 * IC;
        int8_t* dst_block_i4 = wei_packed_i4.data() + ocb * stride16_i4;
        pack_dot_block_interleaved4(src_block, IC, 16, dst_block_i4);
    }

    const size_t lhs_row_stride = IC * 4;
    std::vector<uint8_t> packed_lhs(4 * lhs_row_stride);

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_dot_packed_lhs_strided_interleaved4 block4x16_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_udot_packed_lhs_strided_interleaved4 block4x16_udot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_dot_packed block8_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_udot_packed block8_udot;
    block4x16_dot.create_ker();
    block4x16_udot.create_ker();
    block8_dot.create_ker();
    block8_udot.create_ker();
    const auto block4x16_dot_ker = block4x16_dot.ker();
    const auto block4x16_udot_ker = block4x16_udot.ker();
    const auto block8_dot_ker = block8_dot.ker();
    const auto block8_udot_ker = block8_udot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                size_t w = 0;
                for (; w + 4 <= W; w += 4) {
                    const size_t src_off = ((n * H + h) * W + w) * IC;
                    const uint8_t* src_ptr0 = src.data() + src_off;
                    const uint8_t* src_ptr1 = src_ptr0 + IC;
                    const uint8_t* src_ptr2 = src_ptr1 + IC;
                    const uint8_t* src_ptr3 = src_ptr2 + IC;
                    pack_lhs_4x16_dot_interleaved4(src_ptr0,
                                                   src_ptr1,
                                                   src_ptr2,
                                                   src_ptr3,
                                                   IC,
                                                   packed_lhs.data(),
                                                   lhs_row_stride);
                    const size_t dst_off = ((n * H + h) * W + w) * OC;
                    int32_t* dst_ptr = dst.data() + dst_off;
                    for (size_t oc = 0; oc + 16 <= OC; oc += 16) {
                        const int8_t* wei_ptr = wei_packed_i4.data() + (oc / 16) * stride16_i4;
                        if (args.src_signed) {
                            block4x16_dot_ker(reinterpret_cast<const int8_t*>(packed_lhs.data()),
                                              wei_ptr,
                                              dst_ptr + oc,
                                              IC,
                                              lhs_row_stride,
                                              OC * sizeof(int32_t),
                                              0);
                        } else {
                            block4x16_udot_ker(packed_lhs.data(),
                                               wei_ptr,
                                               dst_ptr + oc,
                                               IC,
                                               lhs_row_stride,
                                               OC * sizeof(int32_t),
                                               0);
                        }
                    }
                }
                for (; w < W; ++w) {
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

TimerResult bench_our_conv_kxk_block4x4_dot(const Args& args) {
    if (!has_asimd_dotprod()) {
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

    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
    dot.create_ker();
    const auto dot_ker = dot.ker();
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_dot block4x4_dot;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_udot block4x4_udot;
    block4x4_dot.create_ker();
    block4x4_udot.create_ker();
    const auto block4x4_dot_ker = block4x4_dot.ker();
    const auto block4x4_udot_ker = block4x4_udot.ker();

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
                size_t w = 0;
                for (; w + 4 <= OW; w += 4) {
                    for (size_t oc = 0; oc + 4 <= OC; oc += 4) {
                        int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC + oc;
                        for (size_t kh = 0; kh < KH; ++kh) {
                            const size_t src_h_off = (h + kh) * stride_h;
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const size_t src_w_off = (w + kw) * stride_w;
                                const uint8_t* src_ptrs_u8[4] = {
                                    src_n + src_h_off + src_w_off + 0 * stride_w,
                                    src_n + src_h_off + src_w_off + 1 * stride_w,
                                    src_n + src_h_off + src_w_off + 2 * stride_w,
                                    src_n + src_h_off + src_w_off + 3 * stride_w,
                                };
                                const int8_t* wei_ptr = packed_wei.data() + ((kh * KW + kw) * OC + oc) * IC;
                                const size_t accum = (kh == 0 && kw == 0) ? 0 : 1;
                                if (args.src_signed) {
                                    const int8_t* src_ptrs_s8[4] = {
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[0]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[1]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[2]),
                                        reinterpret_cast<const int8_t*>(src_ptrs_u8[3]),
                                    };
                                    block4x4_dot_ker(src_ptrs_s8, wei_ptr, dst_ptr, IC, IC, OC * sizeof(int32_t),
                                                     accum);
                                } else {
                                    block4x4_udot_ker(src_ptrs_u8, wei_ptr, dst_ptr, IC, IC, OC * sizeof(int32_t),
                                                      accum);
                                }
                            }
                        }
                    }
                    for (size_t oc = (OC / 4) * 4; oc < OC; ++oc) {
                        for (size_t w_tail = 0; w_tail < 4; ++w_tail) {
                            int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + (w + w_tail)) * OC + oc;
                            bool wrote = false;
                            for (size_t kh = 0; kh < KH; ++kh) {
                                for (size_t kw = 0; kw < KW; ++kw) {
                                    const uint8_t* src_ptr =
                                        src_n + (h + kh) * stride_h + (w + w_tail + kw) * stride_w;
                                    const int8_t* wei_ptr =
                                        packed_wei.data() + ((kh * KW + kw) * OC + oc) * IC;
                                    dot_ker(src_ptr, wei_ptr, dst_ptr, IC, wrote ? 1 : 0);
                                    wrote = true;
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
    const size_t OH = conv_out_dim(H, KH, args.stride, args.pad, args.dilation);
    const size_t OW = conv_out_dim(W, KW, args.stride, args.pad, args.dilation);
    if (OH == 0 || OW == 0) {
        return {-1.0, -1.0};
    }
    const ptrdiff_t pad = static_cast<ptrdiff_t>(args.pad);
    const ptrdiff_t dilation_h = static_cast<ptrdiff_t>(args.dilation);
    const ptrdiff_t dilation_w = static_cast<ptrdiff_t>(args.dilation * IC);
    const size_t conv_stride = args.stride;
    const size_t stride_w = IC;
    const size_t stride_h = W * IC;
    const size_t ic_padded = round_up(IC, 8);

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC * KH * KW);
    std::vector<int32_t> dst(N * OH * OW * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);
    std::vector<uint8_t> zero_src(IC, 0);

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
                            const ptrdiff_t ih = static_cast<ptrdiff_t>(h * conv_stride) - pad +
                                                 static_cast<ptrdiff_t>(kh) * dilation_h;
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const size_t khkw = kh * KW + kw;
                                const uint8_t* src_ptrs_u8[4] = {
                                    zero_src.data(),
                                    zero_src.data(),
                                    zero_src.data(),
                                    zero_src.data(),
                                };
                                if (ih >= 0 && ih < static_cast<ptrdiff_t>(H)) {
                                    const uint8_t* row_base = src_n + static_cast<size_t>(ih) * stride_h;
                                    const ptrdiff_t iw_base = static_cast<ptrdiff_t>(w * conv_stride) - pad +
                                                              static_cast<ptrdiff_t>(kw) * dilation_w / static_cast<ptrdiff_t>(IC);
                                    for (size_t lane = 0; lane < 4; ++lane) {
                                        const ptrdiff_t iw =
                                            iw_base + static_cast<ptrdiff_t>(lane * conv_stride);
                                        if (iw >= 0 && iw < static_cast<ptrdiff_t>(W)) {
                                            src_ptrs_u8[lane] = row_base + static_cast<size_t>(iw) * stride_w;
                                        }
                                    }
                                }
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
                                const ptrdiff_t ih = static_cast<ptrdiff_t>(h * conv_stride) - pad +
                                                     static_cast<ptrdiff_t>(kh) * dilation_h;
                                const ptrdiff_t iw = static_cast<ptrdiff_t>(w * conv_stride) - pad +
                                                     static_cast<ptrdiff_t>(kw) * (dilation_w / static_cast<ptrdiff_t>(IC));
                                if (ih < 0 || ih >= static_cast<ptrdiff_t>(H) || iw < 0 ||
                                    iw >= static_cast<ptrdiff_t>(W)) {
                                    continue;
                                }
                                const uint8_t* src_ptr =
                                    src_n + static_cast<size_t>(ih) * stride_h + static_cast<size_t>(iw) * stride_w;
                                const int8_t* wei_ptr = wei_col.data() + ((kh * KW + kw) * OC + oc) * IC;
                                dot_ker(src_ptr, wei_ptr, dst_ptr + oc, IC, wrote ? 1 : 0);
                                wrote = true;
                            }
                        }
                        if (!wrote) {
                            dst_ptr[oc] = 0;
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

TimerResult bench_our_conv_kxk_mmla_packed_fused(const Args& args, size_t oc_block, bool use_m8x8) {
    if (!has_i8mm() || (oc_block != 4 && oc_block != 8 && oc_block != 12 && oc_block != 16)) {
        return {-1.0, -1.0};
    }
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t KH = args.KH;
    const size_t KW = args.KW;
    if ((IC % 8) != 0) {
        return {-1.0, -1.0};
    }
    if (oc_block == 12) {
        if (!((OC % 12) == 0 || (OC % 12) == 4 || (OC % 12) == 8)) {
            return {-1.0, -1.0};
        }
    } else if (OC % oc_block != 0) {
        return {-1.0, -1.0};
    }
    const size_t OH = conv_out_dim(H, KH, args.stride, args.pad, args.dilation);
    const size_t OW = conv_out_dim(W, KW, args.stride, args.pad, args.dilation);
    if (OH == 0 || OW == 0) {
        return {-1.0, -1.0};
    }
    const size_t conv_stride = args.stride;
    const size_t dilation = args.dilation;
    const ptrdiff_t pad = static_cast<ptrdiff_t>(args.pad);
    const size_t stride_w = IC;
    const size_t stride_h = W * IC;
    const ptrdiff_t dilation_w = static_cast<ptrdiff_t>(dilation * stride_w);
    const ptrdiff_t dilation_h = static_cast<ptrdiff_t>(dilation);
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

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_smmla_packed ker4_s8(true);
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_smmla_packed ker8_s8(true);
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_smmla_packed ker16_s8(true);
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x8_smmla_packed ker8x8_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x12_smmla_packed ker8x12_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_usmmla_packed ker4_u8(true);
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_usmmla_packed ker8_u8(true);
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_usmmla_packed ker16_u8(true);
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x8_usmmla_packed ker8x8_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x12_usmmla_packed ker8x12_u8;
    ker4_s8.create_ker();
    ker8_s8.create_ker();
    ker16_s8.create_ker();
    ker8x8_s8.create_ker();
    ker8x12_s8.create_ker();
    ker4_u8.create_ker();
    ker8_u8.create_ker();
    ker16_u8.create_ker();
    ker8x8_u8.create_ker();
    ker8x12_u8.create_ker();

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

    const size_t oc_blocks = (oc_block == 12) ? (OC / 12) : (OC / oc_block);
    const size_t stride_pack = packed_block_stride_mmla(k_total, oc_block);
    std::vector<int8_t> packed_wei_storage(oc_blocks * stride_pack + 63, 0);
    int8_t* packed_wei = align_ptr(packed_wei_storage.data(), 64);
    std::vector<int8_t> packed_wei12_8;
    std::vector<int8_t> packed_wei12_4;
    size_t stride_pack12_8 = 0;
    size_t stride_pack12_4 = 0;
    std::vector<int8_t> packed_wei16_8;
    size_t stride_pack16_8 = 0;
    size_t stride_pack16_pair = 0;
    for (size_t ocb = 0; ocb < oc_blocks; ++ocb) {
        const int8_t* src_block = fused_col.data() + ocb * oc_block * k_total;
        int8_t* dst_block = packed_wei + ocb * stride_pack;
        pack_mmla_block(src_block, k_total, k_total_padded, oc_block, dst_block);
    }
    if (oc_block == 12) {
        stride_pack12_8 = packed_block_stride_mmla(k_total, 8);
        stride_pack12_4 = packed_block_stride_mmla(k_total, 4);
        packed_wei12_8.assign(oc_blocks * stride_pack12_8, 0);
        packed_wei12_4.assign(oc_blocks * stride_pack12_4, 0);
        for (size_t ocb = 0; ocb < oc_blocks; ++ocb) {
            const int8_t* src_block = fused_col.data() + ocb * 12 * k_total;
            int8_t* dst_block8 = packed_wei12_8.data() + ocb * stride_pack12_8;
            int8_t* dst_block4 = packed_wei12_4.data() + ocb * stride_pack12_4;
            pack_mmla_block(src_block, k_total, k_total_padded, 8, dst_block8);
            pack_mmla_block(src_block + 8 * k_total, k_total, k_total_padded, 4, dst_block4);
        }
    } else if (oc_block == 16) {
        stride_pack16_8 = packed_block_stride_mmla(k_total, 8);
        stride_pack16_pair = stride_pack16_8 * 2;
        packed_wei16_8.assign((OC / 16) * stride_pack16_pair, 0);
        for (size_t ocb = 0; ocb < (OC / 16); ++ocb) {
            const int8_t* src_block = fused_col.data() + ocb * 16 * k_total;
            int8_t* dst_block = packed_wei16_8.data() + ocb * stride_pack16_pair;
            pack_mmla_block(src_block, k_total, k_total_padded, 8, dst_block);
            pack_mmla_block(src_block + 8 * k_total, k_total, k_total_padded, 8, dst_block + stride_pack16_8);
        }
    }
    const bool use_tail4 = (oc_block == 12) && (OC % 12 == 4);
    const bool use_tail8 = (oc_block == 12) && (OC % 12 == 8);
    size_t stride_pack_tail4 = 0;
    std::vector<int8_t> packed_wei_tail4;
    size_t stride_pack_tail8 = 0;
    std::vector<int8_t> packed_wei_tail8;
    if (use_tail4) {
        stride_pack_tail4 = packed_block_stride_mmla(k_total, 4);
        packed_wei_tail4.assign(stride_pack_tail4, 0);
        const size_t oc_tail = oc_blocks * 12;
        const int8_t* src_block = fused_col.data() + oc_tail * k_total;
        pack_mmla_block(src_block, k_total, k_total_padded, 4, packed_wei_tail4.data());
    }
    if (use_tail8) {
        stride_pack_tail8 = packed_block_stride_mmla(k_total, 8);
        packed_wei_tail8.assign(stride_pack_tail8, 0);
        const size_t oc_tail = oc_blocks * 12;
        const int8_t* src_block = fused_col.data() + oc_tail * k_total;
        pack_mmla_block(src_block, k_total, k_total_padded, 8, packed_wei_tail8.data());
    }

    std::vector<uint8_t> packed_src_u8_storage;
    std::vector<int8_t> packed_src_s8_storage;
    uint8_t* packed_src_u8 = nullptr;
    int8_t* packed_src_s8 = nullptr;
    const size_t packed_pairs_elems = 16 * k_total_padded;
    if (use_s8) {
        packed_src_s8_storage.resize(packed_pairs_elems + 63);
        packed_src_s8 = align_ptr(packed_src_s8_storage.data(), 64);
    } else {
        packed_src_u8_storage.resize(packed_pairs_elems + 63);
        packed_src_u8 = align_ptr(packed_src_u8_storage.data(), 64);
    }
    auto pack_ic_blocks_2x8 = [&](const uint8_t* src0, const uint8_t* src1, uint8_t* dst, size_t ic_blocks) {
        const auto* src0_64 = reinterpret_cast<const uint64_t*>(src0);
        const auto* src1_64 = reinterpret_cast<const uint64_t*>(src1);
        auto* dst64 = reinterpret_cast<uint64_t*>(dst);
        for (size_t icb = 0; icb < ic_blocks; ++icb) {
            dst64[0] = src0_64[icb];
            dst64[1] = src1_64[icb];
            dst64 += 2;
        }
    };

    auto zero_ic_blocks_2x8 = [&](uint8_t* dst, size_t ic_blocks) {
        std::memset(dst, 0, ic_blocks * 16);
    };

    auto pack_ic_blocks_2x8_src0 = [&](const uint8_t* src0, uint8_t* dst, size_t ic_blocks) {
        const auto* src0_64 = reinterpret_cast<const uint64_t*>(src0);
        auto* dst64 = reinterpret_cast<uint64_t*>(dst);
        for (size_t icb = 0; icb < ic_blocks; ++icb) {
            dst64[0] = src0_64[icb];
            dst64[1] = 0;
            dst64 += 2;
        }
    };

    auto pack_ic_blocks_2x8_src1 = [&](const uint8_t* src1, uint8_t* dst, size_t ic_blocks) {
        const auto* src1_64 = reinterpret_cast<const uint64_t*>(src1);
        auto* dst64 = reinterpret_cast<uint64_t*>(dst);
        for (size_t icb = 0; icb < ic_blocks; ++icb) {
            dst64[0] = 0;
            dst64[1] = src1_64[icb];
            dst64 += 2;
        }
    };

    auto pack_interleaved = [&](const uint8_t* src_n, size_t h, size_t w_base, size_t m_block, bool use_s8) {
        const size_t pair_count = m_block / 2;
        const size_t ic_blocks = IC / 8;
        const ptrdiff_t ih_base = static_cast<ptrdiff_t>(h * conv_stride) - pad;
        const ptrdiff_t iw_base = static_cast<ptrdiff_t>(w_base * conv_stride) - pad;
        uint8_t* packed_base = use_s8 ? reinterpret_cast<uint8_t*>(packed_src_s8) : packed_src_u8;
        uint8_t* pair_ptrs[8] = {
            packed_base,
            packed_base + 2 * k_total_padded,
            packed_base + 4 * k_total_padded,
            packed_base + 6 * k_total_padded,
            packed_base + 8 * k_total_padded,
            packed_base + 10 * k_total_padded,
            packed_base + 12 * k_total_padded,
            packed_base + 14 * k_total_padded,
        };
        if (k_total_padded != k_total) {
            for (size_t p = 0; p < pair_count; ++p) {
                std::memset(pair_ptrs[p], 0, 2 * k_total_padded);
            }
        }

        const size_t pack_step = ic_blocks * 16;
        if (conv_stride == 1 && dilation == 1 && pair_count <= 4) {
            for (size_t kh = 0; kh < KH; ++kh) {
                const size_t ih = static_cast<size_t>(ih_base + static_cast<ptrdiff_t>(kh) * dilation_h);
                const uint8_t* row_base = src_n + ih * stride_h + static_cast<size_t>(iw_base) * stride_w;
                const size_t kh_off = kh * KW * pack_step;
                uint8_t* dst0 = pair_ptrs[0] + kh_off;
                uint8_t* dst1 = pair_ptrs[1] + kh_off;
                uint8_t* dst2 = pair_count > 2 ? pair_ptrs[2] + kh_off : nullptr;
                uint8_t* dst3 = pair_count > 3 ? pair_ptrs[3] + kh_off : nullptr;
                const uint8_t* src0 = row_base;
                const uint8_t* src1 = src0 + stride_w;
                const uint8_t* src2 = src1 + stride_w;
                const uint8_t* src3 = src2 + stride_w;
                const uint8_t* src4 = src3 + stride_w;
                const uint8_t* src5 = src4 + stride_w;
                const uint8_t* src6 = src5 + stride_w;
                const uint8_t* src7 = src6 + stride_w;
                for (size_t kw = 0; kw < KW; ++kw) {
                    pack_ic_blocks_2x8(src0, src1, dst0, ic_blocks);
                    if (dst1) {
                        pack_ic_blocks_2x8(src2, src3, dst1, ic_blocks);
                    }
                    if (dst2) {
                        pack_ic_blocks_2x8(src4, src5, dst2, ic_blocks);
                    }
                    if (dst3) {
                        pack_ic_blocks_2x8(src6, src7, dst3, ic_blocks);
                    }
                    dst0 += pack_step;
                    src0 += stride_w;
                    src1 += stride_w;
                    if (dst1) {
                        dst1 += pack_step;
                        src2 += stride_w;
                        src3 += stride_w;
                    }
                    if (dst2) {
                        dst2 += pack_step;
                        src4 += stride_w;
                        src5 += stride_w;
                    }
                    if (dst3) {
                        dst3 += pack_step;
                        src6 += stride_w;
                        src7 += stride_w;
                    }
                }
            }
            return;
        }

        for (size_t kh = 0; kh < KH; ++kh) {
            const size_t ih = static_cast<size_t>(ih_base + static_cast<ptrdiff_t>(kh) * dilation_h);
            const size_t src_h_off = ih * stride_h;
            const size_t kh_off = kh * KW * pack_step;
            for (size_t p = 0; p < pair_count; ++p) {
                const ptrdiff_t iw0 = iw_base + static_cast<ptrdiff_t>(2 * p * conv_stride);
                const ptrdiff_t iw1 = iw0 + static_cast<ptrdiff_t>(conv_stride);
                const uint8_t* src0 = src_n + src_h_off + static_cast<size_t>(iw0) * stride_w;
                const uint8_t* src1 = src_n + src_h_off + static_cast<size_t>(iw1) * stride_w;
                uint8_t* dst = pair_ptrs[p] + kh_off;
                for (size_t kw = 0; kw < KW; ++kw) {
                    pack_ic_blocks_2x8(src0, src1, dst, ic_blocks);
                    dst += pack_step;
                    src0 += dilation_w;
                    src1 += dilation_w;
                }
            }
        }
    };

    // Border-capable packing: fill out-of-bounds reads with zeros so we can still run the fused kernels.
    auto pack_interleaved_border = [&](const uint8_t* src_n, size_t h, size_t w_base, size_t m_block, bool use_s8) {
        const size_t pair_count = m_block / 2;
        const size_t ic_blocks = IC / 8;
        const ptrdiff_t ih_base = static_cast<ptrdiff_t>(h * conv_stride) - pad;
        const ptrdiff_t iw_base = static_cast<ptrdiff_t>(w_base * conv_stride) - pad;
        ptrdiff_t iw0_pair[8] = {};
        ptrdiff_t iw1_pair[8] = {};
        bool full_pair[8] = {};
        for (size_t p = 0; p < pair_count; ++p) {
            const ptrdiff_t iw0 = iw_base + static_cast<ptrdiff_t>(2 * p * conv_stride);
            const ptrdiff_t iw1 = iw0 + static_cast<ptrdiff_t>(conv_stride);
            iw0_pair[p] = iw0;
            iw1_pair[p] = iw1;
            const bool full0 =
                iw0 >= 0 && (iw0 + static_cast<ptrdiff_t>((KW - 1) * dilation)) < static_cast<ptrdiff_t>(W);
            const bool full1 =
                iw1 >= 0 && (iw1 + static_cast<ptrdiff_t>((KW - 1) * dilation)) < static_cast<ptrdiff_t>(W);
            full_pair[p] = full0 && full1;
        }

        if (use_s8) {
            int8_t* pair_ptrs[8] = {packed_src_s8,
                                    packed_src_s8 + 2 * k_total_padded,
                                    packed_src_s8 + 4 * k_total_padded,
                                    packed_src_s8 + 6 * k_total_padded,
                                    packed_src_s8 + 8 * k_total_padded,
                                    packed_src_s8 + 10 * k_total_padded,
                                    packed_src_s8 + 12 * k_total_padded,
                                    packed_src_s8 + 14 * k_total_padded};
            if (k_total_padded != k_total) {
                for (size_t p = 0; p < pair_count; ++p) {
                    std::memset(pair_ptrs[p], 0, 2 * k_total_padded);
                }
            }
            for (size_t kh = 0; kh < KH; ++kh) {
                const ptrdiff_t ih = ih_base + static_cast<ptrdiff_t>(kh) * dilation_h;
                const uint8_t* row_base_u8 = nullptr;
                if (ih >= 0 && ih < static_cast<ptrdiff_t>(H)) {
                    row_base_u8 = src_n + static_cast<size_t>(ih) * stride_h;
                }
                const size_t kh_off = kh * KW * ic_blocks * 16;
                if (!row_base_u8) {
                    for (size_t p = 0; p < pair_count; ++p) {
                        std::memset(pair_ptrs[p] + kh_off, 0, KW * ic_blocks * 16);
                    }
                    continue;
                }
                for (size_t p = 0; p < pair_count; ++p) {
                    const ptrdiff_t iw0 = iw0_pair[p];
                    const ptrdiff_t iw1 = iw1_pair[p];
                    int8_t* dst = pair_ptrs[p] + kh_off;
                    if (full_pair[p]) {
                        const int8_t* src0 =
                            reinterpret_cast<const int8_t*>(row_base_u8 + static_cast<size_t>(iw0) * stride_w);
                        const int8_t* src1 =
                            reinterpret_cast<const int8_t*>(row_base_u8 + static_cast<size_t>(iw1) * stride_w);
                        for (size_t kw = 0; kw < KW; ++kw) {
                            pack_ic_blocks_2x8(reinterpret_cast<const uint8_t*>(src0),
                                               reinterpret_cast<const uint8_t*>(src1),
                                               reinterpret_cast<uint8_t*>(dst),
                                               ic_blocks);
                            dst += ic_blocks * 16;
                            src0 += dilation_w;
                            src1 += dilation_w;
                        }
                        continue;
                    }

                    for (size_t kw = 0; kw < KW; ++kw) {
                        const ptrdiff_t iw0_kw = iw0 + static_cast<ptrdiff_t>(kw * dilation);
                        const ptrdiff_t iw1_kw = iw1 + static_cast<ptrdiff_t>(kw * dilation);
                        const bool valid0 = iw0_kw >= 0 && iw0_kw < static_cast<ptrdiff_t>(W);
                        const bool valid1 = iw1_kw >= 0 && iw1_kw < static_cast<ptrdiff_t>(W);
                        if (valid0 && valid1) {
                            const auto* src0 = reinterpret_cast<const uint8_t*>(
                                row_base_u8 + static_cast<size_t>(iw0_kw) * stride_w);
                            const auto* src1 = reinterpret_cast<const uint8_t*>(
                                row_base_u8 + static_cast<size_t>(iw1_kw) * stride_w);
                            pack_ic_blocks_2x8(src0, src1, reinterpret_cast<uint8_t*>(dst), ic_blocks);
                        } else if (valid0) {
                            const auto* src0 = reinterpret_cast<const uint8_t*>(
                                row_base_u8 + static_cast<size_t>(iw0_kw) * stride_w);
                            pack_ic_blocks_2x8_src0(src0, reinterpret_cast<uint8_t*>(dst), ic_blocks);
                        } else if (valid1) {
                            const auto* src1 = reinterpret_cast<const uint8_t*>(
                                row_base_u8 + static_cast<size_t>(iw1_kw) * stride_w);
                            pack_ic_blocks_2x8_src1(src1, reinterpret_cast<uint8_t*>(dst), ic_blocks);
                        } else {
                            zero_ic_blocks_2x8(reinterpret_cast<uint8_t*>(dst), ic_blocks);
                        }
                        dst += ic_blocks * 16;
                    }
                }
            }
        } else {
            uint8_t* pair_ptrs[8] = {packed_src_u8,
                                     packed_src_u8 + 2 * k_total_padded,
                                     packed_src_u8 + 4 * k_total_padded,
                                     packed_src_u8 + 6 * k_total_padded,
                                     packed_src_u8 + 8 * k_total_padded,
                                     packed_src_u8 + 10 * k_total_padded,
                                     packed_src_u8 + 12 * k_total_padded,
                                     packed_src_u8 + 14 * k_total_padded};
            if (k_total_padded != k_total) {
                for (size_t p = 0; p < pair_count; ++p) {
                    std::memset(pair_ptrs[p], 0, 2 * k_total_padded);
                }
            }
            for (size_t kh = 0; kh < KH; ++kh) {
                const ptrdiff_t ih = ih_base + static_cast<ptrdiff_t>(kh) * dilation_h;
                const uint8_t* row_base = nullptr;
                if (ih >= 0 && ih < static_cast<ptrdiff_t>(H)) {
                    row_base = src_n + static_cast<size_t>(ih) * stride_h;
                }
                const size_t kh_off = kh * KW * ic_blocks * 16;
                if (!row_base) {
                    for (size_t p = 0; p < pair_count; ++p) {
                        std::memset(pair_ptrs[p] + kh_off, 0, KW * ic_blocks * 16);
                    }
                    continue;
                }
                for (size_t p = 0; p < pair_count; ++p) {
                    const ptrdiff_t iw0 = iw0_pair[p];
                    const ptrdiff_t iw1 = iw1_pair[p];
                    uint8_t* dst = pair_ptrs[p] + kh_off;
                    if (full_pair[p]) {
                        const uint8_t* src0 = row_base + static_cast<size_t>(iw0) * stride_w;
                        const uint8_t* src1 = row_base + static_cast<size_t>(iw1) * stride_w;
                        for (size_t kw = 0; kw < KW; ++kw) {
                            pack_ic_blocks_2x8(src0, src1, dst, ic_blocks);
                            dst += ic_blocks * 16;
                            src0 += dilation_w;
                            src1 += dilation_w;
                        }
                        continue;
                    }

                    for (size_t kw = 0; kw < KW; ++kw) {
                        const ptrdiff_t iw0_kw = iw0 + static_cast<ptrdiff_t>(kw * dilation);
                        const ptrdiff_t iw1_kw = iw1 + static_cast<ptrdiff_t>(kw * dilation);
                        const bool valid0 = iw0_kw >= 0 && iw0_kw < static_cast<ptrdiff_t>(W);
                        const bool valid1 = iw1_kw >= 0 && iw1_kw < static_cast<ptrdiff_t>(W);
                        if (valid0 && valid1) {
                            pack_ic_blocks_2x8(row_base + static_cast<size_t>(iw0_kw) * stride_w,
                                               row_base + static_cast<size_t>(iw1_kw) * stride_w,
                                               dst,
                                               ic_blocks);
                        } else if (valid0) {
                            pack_ic_blocks_2x8_src0(row_base + static_cast<size_t>(iw0_kw) * stride_w, dst, ic_blocks);
                        } else if (valid1) {
                            pack_ic_blocks_2x8_src1(row_base + static_cast<size_t>(iw1_kw) * stride_w, dst, ic_blocks);
                        } else {
                            zero_ic_blocks_2x8(dst, ic_blocks);
                        }
                        dst += ic_blocks * 16;
                    }
                }
            }
        }
    };

    // Fast border pack for rows where all KH samples are in-bounds: hoist pair/full-window checks out of KH loop.
    auto pack_interleaved_border_hfull = [&](const uint8_t* src_n, size_t h, size_t w_base, size_t m_block, bool use_s8) {
        const size_t pair_count = m_block / 2;
        const size_t ic_blocks = IC / 8;
        const ptrdiff_t ih_base = static_cast<ptrdiff_t>(h * conv_stride) - pad;
        const ptrdiff_t iw_base = static_cast<ptrdiff_t>(w_base * conv_stride) - pad;
        ptrdiff_t iw0_pair[8] = {};
        ptrdiff_t iw1_pair[8] = {};
        bool full_pair[8] = {};
        for (size_t p = 0; p < pair_count; ++p) {
            const ptrdiff_t iw0 = iw_base + static_cast<ptrdiff_t>(2 * p * conv_stride);
            const ptrdiff_t iw1 = iw0 + static_cast<ptrdiff_t>(conv_stride);
            iw0_pair[p] = iw0;
            iw1_pair[p] = iw1;
            const bool full0 = iw0 >= 0 && (iw0 + static_cast<ptrdiff_t>((KW - 1) * dilation)) < static_cast<ptrdiff_t>(W);
            const bool full1 = iw1 >= 0 && (iw1 + static_cast<ptrdiff_t>((KW - 1) * dilation)) < static_cast<ptrdiff_t>(W);
            full_pair[p] = full0 && full1;
        }
        const bool use_special_3x3_d2 = (KH == 3) && (KW == 3) && (dilation == 2);
        const bool use_special_5x5_p1 = (KH == 5) && (KW == 5) && (dilation == 1);
        const bool left_partial_only = pair_count >= 2 && !full_pair[0] &&
                                       std::all_of(full_pair + 1, full_pair + pair_count, [](bool v) { return v; });
        const bool right_partial_only = pair_count >= 2 && !full_pair[pair_count - 1] &&
                                        std::all_of(full_pair, full_pair + pair_count - 1, [](bool v) { return v; });
        auto pack_partial_pair_s8 = [&](const uint8_t* row_base_u8, ptrdiff_t iw0, ptrdiff_t iw1, int8_t*& dst) {
            auto pack_generic_kw = [&](ptrdiff_t kw_idx) {
                const ptrdiff_t iw0_kw = iw0 + kw_idx * static_cast<ptrdiff_t>(dilation);
                const ptrdiff_t iw1_kw = iw1 + kw_idx * static_cast<ptrdiff_t>(dilation);
                const bool valid0 = iw0_kw >= 0 && iw0_kw < static_cast<ptrdiff_t>(W);
                const bool valid1 = iw1_kw >= 0 && iw1_kw < static_cast<ptrdiff_t>(W);
                if (valid0 && valid1) {
                    const auto* src0 =
                        reinterpret_cast<const uint8_t*>(row_base_u8 + static_cast<size_t>(iw0_kw) * stride_w);
                    const auto* src1 =
                        reinterpret_cast<const uint8_t*>(row_base_u8 + static_cast<size_t>(iw1_kw) * stride_w);
                    pack_ic_blocks_2x8(src0, src1, reinterpret_cast<uint8_t*>(dst), ic_blocks);
                } else if (valid0) {
                    const auto* src0 =
                        reinterpret_cast<const uint8_t*>(row_base_u8 + static_cast<size_t>(iw0_kw) * stride_w);
                    pack_ic_blocks_2x8_src0(src0, reinterpret_cast<uint8_t*>(dst), ic_blocks);
                } else if (valid1) {
                    const auto* src1 =
                        reinterpret_cast<const uint8_t*>(row_base_u8 + static_cast<size_t>(iw1_kw) * stride_w);
                    pack_ic_blocks_2x8_src1(src1, reinterpret_cast<uint8_t*>(dst), ic_blocks);
                } else {
                    zero_ic_blocks_2x8(reinterpret_cast<uint8_t*>(dst), ic_blocks);
                }
                dst += ic_blocks * 16;
            };

            if (dilation == 1) {
                const ptrdiff_t kw0_beg = std::max<ptrdiff_t>(0, -iw0);
                const ptrdiff_t kw0_end =
                    std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW), static_cast<ptrdiff_t>(W) - iw0);
                const ptrdiff_t kw1_beg = std::max<ptrdiff_t>(0, -iw1);
                const ptrdiff_t kw1_end =
                    std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW), static_cast<ptrdiff_t>(W) - iw1);
                const ptrdiff_t mid_beg = std::max(kw0_beg, kw1_beg);
                const ptrdiff_t mid_end = std::min(kw0_end, kw1_end);

                ptrdiff_t kw = 0;
                for (; kw < mid_beg; ++kw) {
                    pack_generic_kw(kw);
                }
                if (mid_end > mid_beg) {
                    const int8_t* src0 = reinterpret_cast<const int8_t*>(
                        row_base_u8 + static_cast<size_t>(iw0 + mid_beg) * stride_w);
                    const int8_t* src1 = reinterpret_cast<const int8_t*>(
                        row_base_u8 + static_cast<size_t>(iw1 + mid_beg) * stride_w);
                    for (ptrdiff_t k = mid_beg; k < mid_end; ++k) {
                        pack_ic_blocks_2x8(reinterpret_cast<const uint8_t*>(src0),
                                           reinterpret_cast<const uint8_t*>(src1),
                                           reinterpret_cast<uint8_t*>(dst),
                                           ic_blocks);
                        dst += ic_blocks * 16;
                        src0 += stride_w;
                        src1 += stride_w;
                    }
                    kw = mid_end;
                }
                for (; kw < static_cast<ptrdiff_t>(KW); ++kw) {
                    pack_generic_kw(kw);
                }
                return;
            }

            for (size_t kw = 0; kw < KW; ++kw) {
                pack_generic_kw(static_cast<ptrdiff_t>(kw));
            }
        };

        auto pack_partial_pair_u8 = [&](const uint8_t* row_base, ptrdiff_t iw0, ptrdiff_t iw1, uint8_t*& dst) {
            auto pack_generic_kw = [&](ptrdiff_t kw_idx) {
                const ptrdiff_t iw0_kw = iw0 + kw_idx * static_cast<ptrdiff_t>(dilation);
                const ptrdiff_t iw1_kw = iw1 + kw_idx * static_cast<ptrdiff_t>(dilation);
                const bool valid0 = iw0_kw >= 0 && iw0_kw < static_cast<ptrdiff_t>(W);
                const bool valid1 = iw1_kw >= 0 && iw1_kw < static_cast<ptrdiff_t>(W);
                if (valid0 && valid1) {
                    pack_ic_blocks_2x8(row_base + static_cast<size_t>(iw0_kw) * stride_w,
                                       row_base + static_cast<size_t>(iw1_kw) * stride_w,
                                       dst,
                                       ic_blocks);
                } else if (valid0) {
                    pack_ic_blocks_2x8_src0(row_base + static_cast<size_t>(iw0_kw) * stride_w, dst, ic_blocks);
                } else if (valid1) {
                    pack_ic_blocks_2x8_src1(row_base + static_cast<size_t>(iw1_kw) * stride_w, dst, ic_blocks);
                } else {
                    zero_ic_blocks_2x8(dst, ic_blocks);
                }
                dst += ic_blocks * 16;
            };

            if (dilation == 1) {
                const ptrdiff_t kw0_beg = std::max<ptrdiff_t>(0, -iw0);
                const ptrdiff_t kw0_end =
                    std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW), static_cast<ptrdiff_t>(W) - iw0);
                const ptrdiff_t kw1_beg = std::max<ptrdiff_t>(0, -iw1);
                const ptrdiff_t kw1_end =
                    std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW), static_cast<ptrdiff_t>(W) - iw1);
                const ptrdiff_t mid_beg = std::max(kw0_beg, kw1_beg);
                const ptrdiff_t mid_end = std::min(kw0_end, kw1_end);

                ptrdiff_t kw = 0;
                for (; kw < mid_beg; ++kw) {
                    pack_generic_kw(kw);
                }
                if (mid_end > mid_beg) {
                    const uint8_t* src0 = row_base + static_cast<size_t>(iw0 + mid_beg) * stride_w;
                    const uint8_t* src1 = row_base + static_cast<size_t>(iw1 + mid_beg) * stride_w;
                    for (ptrdiff_t k = mid_beg; k < mid_end; ++k) {
                        pack_ic_blocks_2x8(src0, src1, dst, ic_blocks);
                        dst += ic_blocks * 16;
                        src0 += stride_w;
                        src1 += stride_w;
                    }
                    kw = mid_end;
                }
                for (; kw < static_cast<ptrdiff_t>(KW); ++kw) {
                    pack_generic_kw(kw);
                }
                return;
            }

            for (size_t kw = 0; kw < KW; ++kw) {
                pack_generic_kw(static_cast<ptrdiff_t>(kw));
            }
        };

        if (use_s8) {
            int8_t* pair_ptrs[8] = {packed_src_s8,
                                    packed_src_s8 + 2 * k_total_padded,
                                    packed_src_s8 + 4 * k_total_padded,
                                    packed_src_s8 + 6 * k_total_padded,
                                    packed_src_s8 + 8 * k_total_padded,
                                    packed_src_s8 + 10 * k_total_padded,
                                    packed_src_s8 + 12 * k_total_padded,
                                    packed_src_s8 + 14 * k_total_padded};
            if (k_total_padded != k_total) {
                for (size_t p = 0; p < pair_count; ++p) {
                    std::memset(pair_ptrs[p], 0, 2 * k_total_padded);
                }
            }
            for (size_t kh = 0; kh < KH; ++kh) {
                const ptrdiff_t ih = ih_base + static_cast<ptrdiff_t>(kh) * dilation_h;
                const uint8_t* row_base_u8 = src_n + static_cast<size_t>(ih) * stride_h;
                const size_t kh_off = kh * KW * ic_blocks * 16;
                auto pack_full_pair_s8 = [&](size_t p) {
                    int8_t* dst = pair_ptrs[p] + kh_off;
                    const int8_t* src0 =
                        reinterpret_cast<const int8_t*>(row_base_u8 + static_cast<size_t>(iw0_pair[p]) * stride_w);
                    const int8_t* src1 =
                        reinterpret_cast<const int8_t*>(row_base_u8 + static_cast<size_t>(iw1_pair[p]) * stride_w);
                    for (size_t kw = 0; kw < KW; ++kw) {
                        pack_ic_blocks_2x8(reinterpret_cast<const uint8_t*>(src0),
                                           reinterpret_cast<const uint8_t*>(src1),
                                           reinterpret_cast<uint8_t*>(dst),
                                           ic_blocks);
                        dst += ic_blocks * 16;
                        src0 += dilation_w;
                        src1 += dilation_w;
                    }
                };
                if (left_partial_only && (use_special_3x3_d2 || use_special_5x5_p1)) {
                    int8_t* dst = pair_ptrs[0] + kh_off;
                    if (use_special_3x3_d2) {
                        zero_ic_blocks_2x8(reinterpret_cast<uint8_t*>(dst), ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(row_base_u8,
                                           row_base_u8 + stride_w,
                                           reinterpret_cast<uint8_t*>(dst),
                                           ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(row_base_u8 + 2 * stride_w,
                                           row_base_u8 + 3 * stride_w,
                                           reinterpret_cast<uint8_t*>(dst),
                                           ic_blocks);
                    } else {
                        zero_ic_blocks_2x8(reinterpret_cast<uint8_t*>(dst), ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8_src1(row_base_u8 + stride_w, reinterpret_cast<uint8_t*>(dst), ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(row_base_u8,
                                           row_base_u8 + stride_w,
                                           reinterpret_cast<uint8_t*>(dst),
                                           ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(row_base_u8 + stride_w,
                                           row_base_u8 + 2 * stride_w,
                                           reinterpret_cast<uint8_t*>(dst),
                                           ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(row_base_u8 + 2 * stride_w,
                                           row_base_u8 + 3 * stride_w,
                                           reinterpret_cast<uint8_t*>(dst),
                                           ic_blocks);
                    }
                    for (size_t p = 1; p < pair_count; ++p) {
                        pack_full_pair_s8(p);
                    }
                    continue;
                }
                if (right_partial_only && (use_special_3x3_d2 || use_special_5x5_p1)) {
                    for (size_t p = 0; p + 1 < pair_count; ++p) {
                        pack_full_pair_s8(p);
                    }
                    int8_t* dst = pair_ptrs[pair_count - 1] + kh_off;
                    const uint8_t* src0 = row_base_u8 + static_cast<size_t>(iw0_pair[pair_count - 1]) * stride_w;
                    const uint8_t* src1 = row_base_u8 + static_cast<size_t>(iw1_pair[pair_count - 1]) * stride_w;
                    if (use_special_3x3_d2) {
                        pack_ic_blocks_2x8(src0, src1, reinterpret_cast<uint8_t*>(dst), ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(src0 + 2 * stride_w,
                                           src1 + 2 * stride_w,
                                           reinterpret_cast<uint8_t*>(dst),
                                           ic_blocks);
                        dst += ic_blocks * 16;
                        zero_ic_blocks_2x8(reinterpret_cast<uint8_t*>(dst), ic_blocks);
                    } else {
                        pack_ic_blocks_2x8(src0, src1, reinterpret_cast<uint8_t*>(dst), ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(src0 + stride_w,
                                           src1 + stride_w,
                                           reinterpret_cast<uint8_t*>(dst),
                                           ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(src0 + 2 * stride_w,
                                           src1 + 2 * stride_w,
                                           reinterpret_cast<uint8_t*>(dst),
                                           ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8_src0(src0 + 3 * stride_w, reinterpret_cast<uint8_t*>(dst), ic_blocks);
                        dst += ic_blocks * 16;
                        zero_ic_blocks_2x8(reinterpret_cast<uint8_t*>(dst), ic_blocks);
                    }
                    continue;
                }
                for (size_t p = 0; p < pair_count; ++p) {
                    int8_t* dst = pair_ptrs[p] + kh_off;
                    const ptrdiff_t iw0 = iw0_pair[p];
                    const ptrdiff_t iw1 = iw1_pair[p];
                    if (full_pair[p]) {
                        const int8_t* src0 =
                            reinterpret_cast<const int8_t*>(row_base_u8 + static_cast<size_t>(iw0) * stride_w);
                        const int8_t* src1 =
                            reinterpret_cast<const int8_t*>(row_base_u8 + static_cast<size_t>(iw1) * stride_w);
                        for (size_t kw = 0; kw < KW; ++kw) {
                            pack_ic_blocks_2x8(reinterpret_cast<const uint8_t*>(src0),
                                               reinterpret_cast<const uint8_t*>(src1),
                                               reinterpret_cast<uint8_t*>(dst),
                                               ic_blocks);
                            dst += ic_blocks * 16;
                            src0 += dilation_w;
                            src1 += dilation_w;
                        }
                        continue;
                    }
                    pack_partial_pair_s8(row_base_u8, iw0, iw1, dst);
                }
            }
        } else {
            uint8_t* pair_ptrs[8] = {packed_src_u8,
                                     packed_src_u8 + 2 * k_total_padded,
                                     packed_src_u8 + 4 * k_total_padded,
                                     packed_src_u8 + 6 * k_total_padded,
                                     packed_src_u8 + 8 * k_total_padded,
                                     packed_src_u8 + 10 * k_total_padded,
                                     packed_src_u8 + 12 * k_total_padded,
                                     packed_src_u8 + 14 * k_total_padded};
            if (k_total_padded != k_total) {
                for (size_t p = 0; p < pair_count; ++p) {
                    std::memset(pair_ptrs[p], 0, 2 * k_total_padded);
                }
            }
            for (size_t kh = 0; kh < KH; ++kh) {
                const ptrdiff_t ih = ih_base + static_cast<ptrdiff_t>(kh) * dilation_h;
                const uint8_t* row_base = src_n + static_cast<size_t>(ih) * stride_h;
                const size_t kh_off = kh * KW * ic_blocks * 16;
                auto pack_full_pair_u8 = [&](size_t p) {
                    uint8_t* dst = pair_ptrs[p] + kh_off;
                    const uint8_t* src0 = row_base + static_cast<size_t>(iw0_pair[p]) * stride_w;
                    const uint8_t* src1 = row_base + static_cast<size_t>(iw1_pair[p]) * stride_w;
                    for (size_t kw = 0; kw < KW; ++kw) {
                        pack_ic_blocks_2x8(src0, src1, dst, ic_blocks);
                        dst += ic_blocks * 16;
                        src0 += dilation_w;
                        src1 += dilation_w;
                    }
                };
                if (left_partial_only && (use_special_3x3_d2 || use_special_5x5_p1)) {
                    uint8_t* dst = pair_ptrs[0] + kh_off;
                    if (use_special_3x3_d2) {
                        zero_ic_blocks_2x8(dst, ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(row_base, row_base + stride_w, dst, ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(row_base + 2 * stride_w, row_base + 3 * stride_w, dst, ic_blocks);
                    } else {
                        zero_ic_blocks_2x8(dst, ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8_src1(row_base + stride_w, dst, ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(row_base, row_base + stride_w, dst, ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(row_base + stride_w, row_base + 2 * stride_w, dst, ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(row_base + 2 * stride_w, row_base + 3 * stride_w, dst, ic_blocks);
                    }
                    for (size_t p = 1; p < pair_count; ++p) {
                        pack_full_pair_u8(p);
                    }
                    continue;
                }
                if (right_partial_only && (use_special_3x3_d2 || use_special_5x5_p1)) {
                    for (size_t p = 0; p + 1 < pair_count; ++p) {
                        pack_full_pair_u8(p);
                    }
                    uint8_t* dst = pair_ptrs[pair_count - 1] + kh_off;
                    const uint8_t* src0 = row_base + static_cast<size_t>(iw0_pair[pair_count - 1]) * stride_w;
                    const uint8_t* src1 = row_base + static_cast<size_t>(iw1_pair[pair_count - 1]) * stride_w;
                    if (use_special_3x3_d2) {
                        pack_ic_blocks_2x8(src0, src1, dst, ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(src0 + 2 * stride_w, src1 + 2 * stride_w, dst, ic_blocks);
                        dst += ic_blocks * 16;
                        zero_ic_blocks_2x8(dst, ic_blocks);
                    } else {
                        pack_ic_blocks_2x8(src0, src1, dst, ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(src0 + stride_w, src1 + stride_w, dst, ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8(src0 + 2 * stride_w, src1 + 2 * stride_w, dst, ic_blocks);
                        dst += ic_blocks * 16;
                        pack_ic_blocks_2x8_src0(src0 + 3 * stride_w, dst, ic_blocks);
                        dst += ic_blocks * 16;
                        zero_ic_blocks_2x8(dst, ic_blocks);
                    }
                    continue;
                }
                for (size_t p = 0; p < pair_count; ++p) {
                    uint8_t* dst = pair_ptrs[p] + kh_off;
                    const ptrdiff_t iw0 = iw0_pair[p];
                    const ptrdiff_t iw1 = iw1_pair[p];
                    if (full_pair[p]) {
                        const uint8_t* src0 = row_base + static_cast<size_t>(iw0) * stride_w;
                        const uint8_t* src1 = row_base + static_cast<size_t>(iw1) * stride_w;
                        for (size_t kw = 0; kw < KW; ++kw) {
                            pack_ic_blocks_2x8(src0, src1, dst, ic_blocks);
                            dst += ic_blocks * 16;
                            src0 += dilation_w;
                            src1 += dilation_w;
                        }
                        continue;
                    }
                    pack_partial_pair_u8(row_base, iw0, iw1, dst);
                }
            }
        }
    };

    if (args.pack_only) {
        auto run_pack_only = [&]() {
            for (size_t n = 0; n < N; ++n) {
                const uint8_t* src_n = src.data() + n * H * W * IC;
                for (size_t h = 0; h < OH; ++h) {
                    const ptrdiff_t ih_base = static_cast<ptrdiff_t>(h * conv_stride) - pad;
                    const bool full_h =
                        ih_base >= 0 &&
                        (ih_base + static_cast<ptrdiff_t>((KH - 1) * dilation + 1)) <= static_cast<ptrdiff_t>(H);
                    size_t w = 0;
                    for (; w + 8 <= OW; w += 8) {
                        const ptrdiff_t iw_base = static_cast<ptrdiff_t>(w * conv_stride) - pad;
                        const bool full_w =
                            iw_base >= 0 &&
                            (iw_base + static_cast<ptrdiff_t>((8 - 1) * conv_stride + (KW - 1) * dilation)) <
                                static_cast<ptrdiff_t>(W);
                        if (full_h && full_w) {
                            pack_interleaved(src_n, h, w, 8, use_s8);
                        } else if (full_h) {
                            pack_interleaved_border_hfull(src_n, h, w, 8, use_s8);
                        } else {
                            pack_interleaved_border(src_n, h, w, 8, use_s8);
                        }
                    }
                    for (; w + 4 <= OW; w += 4) {
                        const ptrdiff_t iw_base = static_cast<ptrdiff_t>(w * conv_stride) - pad;
                        const bool full_w =
                            iw_base >= 0 &&
                            (iw_base + static_cast<ptrdiff_t>((4 - 1) * conv_stride + (KW - 1) * dilation)) <
                                static_cast<ptrdiff_t>(W);
                        if (full_h && full_w) {
                            pack_interleaved(src_n, h, w, 4, use_s8);
                        } else if (full_h) {
                            pack_interleaved_border_hfull(src_n, h, w, 4, use_s8);
                        } else {
                            pack_interleaved_border(src_n, h, w, 4, use_s8);
                        }
                    }
                }
            }
        };

        for (size_t i = 0; i < args.warmup; ++i) {
            run_pack_only();
        }
        return time_loop(args.iters, 0.0, run_pack_only);
    }

    const int8_t* packed_s8_base = packed_src_s8;
    const uint8_t* packed_u8_base = packed_src_u8;
    const int8_t* packed_s8_full[8] = {
        packed_s8_base,
        packed_s8_base ? packed_s8_base + 2 * k_total_padded : nullptr,
        packed_s8_base ? packed_s8_base + 4 * k_total_padded : nullptr,
        packed_s8_base ? packed_s8_base + 6 * k_total_padded : nullptr,
        packed_s8_base ? packed_s8_base + 8 * k_total_padded : nullptr,
        packed_s8_base ? packed_s8_base + 10 * k_total_padded : nullptr,
        packed_s8_base ? packed_s8_base + 12 * k_total_padded : nullptr,
        packed_s8_base ? packed_s8_base + 14 * k_total_padded : nullptr,
    };
    const uint8_t* packed_u8_full[8] = {
        packed_u8_base,
        packed_u8_base ? packed_u8_base + 2 * k_total_padded : nullptr,
        packed_u8_base ? packed_u8_base + 4 * k_total_padded : nullptr,
        packed_u8_base ? packed_u8_base + 6 * k_total_padded : nullptr,
        packed_u8_base ? packed_u8_base + 8 * k_total_padded : nullptr,
        packed_u8_base ? packed_u8_base + 10 * k_total_padded : nullptr,
        packed_u8_base ? packed_u8_base + 12 * k_total_padded : nullptr,
        packed_u8_base ? packed_u8_base + 14 * k_total_padded : nullptr,
    };
    const int8_t* packed_s8_pair0[4] = {
        packed_s8_base,
        packed_s8_base ? packed_s8_base + 2 * k_total_padded : nullptr,
        nullptr,
        nullptr,
    };
    const uint8_t* packed_u8_pair0[4] = {
        packed_u8_base,
        packed_u8_base ? packed_u8_base + 2 * k_total_padded : nullptr,
        nullptr,
        nullptr,
    };

    auto run = [&]() {
        auto run_block4 = [&](size_t n, size_t h, size_t w, const uint8_t* src_n, bool full_h) {
            const size_t m_block = 4;
            const ptrdiff_t iw0_base = static_cast<ptrdiff_t>(w * conv_stride) - pad;
            const bool full_w =
                iw0_base >= 0 &&
                (iw0_base + static_cast<ptrdiff_t>((m_block - 1) * conv_stride + (KW - 1) * dilation)) <
                    static_cast<ptrdiff_t>(W);
            if (full_h && full_w) {
                pack_interleaved(src_n, h, w, m_block, use_s8);
            } else if (full_h) {
                pack_interleaved_border_hfull(src_n, h, w, m_block, use_s8);
            } else {
                pack_interleaved_border(src_n, h, w, m_block, use_s8);
            }
            if (oc_block == 12) {
                for (size_t oc = 0; oc + 12 <= OC; oc += 12) {
                    const size_t ocb = oc / 12;
                    const int8_t* wei_ptr8 = packed_wei12_8.data() + ocb * stride_pack12_8;
                    const int8_t* wei_ptr4 = packed_wei12_4.data() + ocb * stride_pack12_4;
                    int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC + oc;
                    if (use_s8) {
                        ker8_s8.ker()(packed_s8_pair0, wei_ptr8, dst_ptr, k_total_padded, 0, OC * sizeof(int32_t), 0);
                        ker4_s8.ker()(packed_s8_pair0,
                                      wei_ptr4,
                                      dst_ptr + 8,
                                      k_total_padded,
                                      0,
                                      OC * sizeof(int32_t),
                                      0);
                    } else {
                        ker8_u8.ker()(packed_u8_pair0, wei_ptr8, dst_ptr, k_total_padded, 0, OC * sizeof(int32_t), 0);
                        ker4_u8.ker()(packed_u8_pair0,
                                      wei_ptr4,
                                      dst_ptr + 8,
                                      k_total_padded,
                                      0,
                                      OC * sizeof(int32_t),
                                      0);
                    }
                }
                if (use_tail8) {
                    const size_t oc_tail = oc_blocks * 12;
                    const int8_t* wei_ptr = packed_wei_tail8.data();
                    int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC + oc_tail;
                    if (use_s8) {
                        ker8_s8.ker()(packed_s8_pair0, wei_ptr, dst_ptr, k_total_padded, 0, OC * sizeof(int32_t), 0);
                    } else {
                        ker8_u8.ker()(packed_u8_pair0, wei_ptr, dst_ptr, k_total_padded, 0, OC * sizeof(int32_t), 0);
                    }
                } else if (use_tail4) {
                    const size_t oc_tail = oc_blocks * 12;
                    const int8_t* wei_ptr = packed_wei_tail4.data();
                    int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC + oc_tail;
                    if (use_s8) {
                        ker4_s8.ker()(packed_s8_pair0, wei_ptr, dst_ptr, k_total_padded, 0, OC * sizeof(int32_t), 0);
                    } else {
                        ker4_u8.ker()(packed_u8_pair0, wei_ptr, dst_ptr, k_total_padded, 0, OC * sizeof(int32_t), 0);
                    }
                }
            } else {
                for (size_t oc = 0; oc < OC; oc += oc_block) {
                    int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC + oc;
                    const int8_t* wei_ptr = packed_wei + (oc / oc_block) * stride_pack;
                    if (use_s8) {
                        if (oc_block == 16) {
                            ker16_s8.ker()(packed_s8_pair0, wei_ptr, dst_ptr, k_total_padded, 0, OC * sizeof(int32_t), 0);
                        } else if (oc_block == 8) {
                            ker8_s8.ker()(packed_s8_pair0, wei_ptr, dst_ptr, k_total_padded, 0, OC * sizeof(int32_t), 0);
                        } else {
                            ker4_s8.ker()(packed_s8_pair0, wei_ptr, dst_ptr, k_total_padded, 0, OC * sizeof(int32_t), 0);
                        }
                    } else {
                        if (oc_block == 16) {
                            ker16_u8.ker()(packed_u8_pair0, wei_ptr, dst_ptr, k_total_padded, 0, OC * sizeof(int32_t), 0);
                        } else if (oc_block == 8) {
                            ker8_u8.ker()(packed_u8_pair0, wei_ptr, dst_ptr, k_total_padded, 0, OC * sizeof(int32_t), 0);
                        } else {
                            ker4_u8.ker()(packed_u8_pair0, wei_ptr, dst_ptr, k_total_padded, 0, OC * sizeof(int32_t), 0);
                        }
                    }
                }
            }
        };

        for (size_t n = 0; n < N; ++n) {
            const uint8_t* src_n = src.data() + n * H * W * IC;
            for (size_t h = 0; h < OH; ++h) {
                const ptrdiff_t ih0_base = static_cast<ptrdiff_t>(h * conv_stride) - pad;
                const bool full_h =
                    ih0_base >= 0 &&
                    (ih0_base + static_cast<ptrdiff_t>((KH - 1) * dilation + 1)) <= static_cast<ptrdiff_t>(H);
                size_t w = 0;
                for (; w + 8 <= OW; w += 8) {
                    const size_t m_block = 8;
                    const ptrdiff_t iw0_base = static_cast<ptrdiff_t>(w * conv_stride) - pad;
                    const bool full_w =
                        iw0_base >= 0 &&
                        (iw0_base + static_cast<ptrdiff_t>((m_block - 1) * conv_stride + (KW - 1) * dilation)) <
                            static_cast<ptrdiff_t>(W);
                    if (full_h && full_w) {
                        pack_interleaved(src_n, h, w, m_block, use_s8);
                    } else if (full_h) {
                        pack_interleaved_border_hfull(src_n, h, w, m_block, use_s8);
                    } else {
                        pack_interleaved_border(src_n, h, w, m_block, use_s8);
                    }
                    if (use_m8x8 && oc_block == 12) {
                        size_t oc = 0;
                        for (; oc + 12 <= OC; oc += 12) {
                            const int8_t* wei_ptr = packed_wei + (oc / 12) * stride_pack;
                            int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC + oc;
                            if (use_s8) {
                                ker8x12_s8.ker()(packed_s8_full,
                                                 wei_ptr,
                                                 dst_ptr,
                                                 k_total_padded,
                                                 0,
                                                 OC * sizeof(int32_t),
                                                 0);
                            } else {
                                ker8x12_u8.ker()(packed_u8_full,
                                                 wei_ptr,
                                                 dst_ptr,
                                                 k_total_padded,
                                                 0,
                                                 OC * sizeof(int32_t),
                                                 0);
                            }
                        }
                        if (use_tail8) {
                            const size_t oc_tail = oc_blocks * 12;
                            const int8_t* wei_ptr = packed_wei_tail8.data();
                            int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC + oc_tail;
                            if (use_s8) {
                                ker8x8_s8.ker()(packed_s8_full,
                                                wei_ptr,
                                                dst_ptr,
                                                k_total_padded,
                                                0,
                                                OC * sizeof(int32_t),
                                                0);
                            } else {
                                ker8x8_u8.ker()(packed_u8_full,
                                                wei_ptr,
                                                dst_ptr,
                                                k_total_padded,
                                                0,
                                                OC * sizeof(int32_t),
                                                0);
                            }
                        } else if (use_tail4) {
                            const size_t oc_tail = oc_blocks * 12;
                            const int8_t* wei_ptr = packed_wei_tail4.data();
                            for (size_t blk = 0; blk < m_block; blk += 4) {
                                int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + (w + blk)) * OC + oc_tail;
                                const size_t pair0_off = (blk / 4) * 4 * k_total_padded;
                                const int8_t* src_ptrs_s8[4] = {
                                    packed_s8_base ? packed_s8_base + pair0_off : nullptr,
                                    packed_s8_base ? packed_s8_base + pair0_off + 2 * k_total_padded : nullptr,
                                    nullptr,
                                    nullptr,
                                };
                                const uint8_t* src_ptrs_u8[4] = {
                                    packed_u8_base ? packed_u8_base + pair0_off : nullptr,
                                    packed_u8_base ? packed_u8_base + pair0_off + 2 * k_total_padded : nullptr,
                                    nullptr,
                                    nullptr,
                                };
                                if (use_s8) {
                                    ker4_s8.ker()(src_ptrs_s8,
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
                    } else {
                        for (size_t oc = 0; oc < OC; oc += oc_block) {
                            const int8_t* wei_ptr = packed_wei + (oc / oc_block) * stride_pack;
                            if (oc_block == 16 && m_block == 8) {
                                const size_t ocb = oc / 16;
                                const int8_t* wei_ptr0 = packed_wei16_8.data() + ocb * stride_pack16_pair;
                                const int8_t* wei_ptr1 = wei_ptr0 + stride_pack16_8;
                                int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC + oc;
                                if (use_s8) {
                                    ker8x8_s8.ker()(packed_s8_full,
                                                    wei_ptr0,
                                                    dst_ptr,
                                                    k_total_padded,
                                                    0,
                                                    OC * sizeof(int32_t),
                                                    0);
                                    ker8x8_s8.ker()(packed_s8_full,
                                                    wei_ptr1,
                                                    dst_ptr + 8,
                                                    k_total_padded,
                                                    0,
                                                    OC * sizeof(int32_t),
                                                    0);
                                } else {
                                    ker8x8_u8.ker()(packed_u8_full,
                                                    wei_ptr0,
                                                    dst_ptr,
                                                    k_total_padded,
                                                    0,
                                                    OC * sizeof(int32_t),
                                                    0);
                                    ker8x8_u8.ker()(packed_u8_full,
                                                    wei_ptr1,
                                                    dst_ptr + 8,
                                                    k_total_padded,
                                                    0,
                                                    OC * sizeof(int32_t),
                                                    0);
                                }
                            } else if (use_m8x8 && oc_block == 8) {
                                int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC + oc;
                                if (use_s8) {
                                    ker8x8_s8.ker()(packed_s8_full,
                                                    wei_ptr,
                                                    dst_ptr,
                                                    k_total_padded,
                                                    0,
                                                    OC * sizeof(int32_t),
                                                    0);
                                } else {
                                    ker8x8_u8.ker()(packed_u8_full,
                                                    wei_ptr,
                                                    dst_ptr,
                                                    k_total_padded,
                                                    0,
                                                    OC * sizeof(int32_t),
                                                    0);
                                }
                            } else {
                                for (size_t blk = 0; blk < m_block; blk += 4) {
                                    int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + (w + blk)) * OC + oc;
                                    const size_t pair0_off = (blk / 4) * 4 * k_total_padded;
                                    const int8_t* src_ptrs_s8[4] = {
                                        packed_s8_base ? packed_s8_base + pair0_off : nullptr,
                                        packed_s8_base ? packed_s8_base + pair0_off + 2 * k_total_padded : nullptr,
                                        nullptr,
                                        nullptr,
                                    };
                                    const uint8_t* src_ptrs_u8[4] = {
                                        packed_u8_base ? packed_u8_base + pair0_off : nullptr,
                                        packed_u8_base ? packed_u8_base + pair0_off + 2 * k_total_padded : nullptr,
                                        nullptr,
                                        nullptr,
                                    };
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
                        }
                    }
                }
                for (; w + 4 <= OW; w += 4) {
                    run_block4(n, h, w, src_n, full_h);
                }
                for (; w < OW; ++w) {
                    const ptrdiff_t iw_base = static_cast<ptrdiff_t>(w * conv_stride) - pad;
                    int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC;
                    for (size_t oc = 0; oc < OC; ++oc) {
                        dst_ptr[oc] = 0;
                        const int8_t* wei_oc = fused_col.data() + oc * k_total;
                        for (size_t kh = 0; kh < KH; ++kh) {
                            const ptrdiff_t ih = ih0_base + static_cast<ptrdiff_t>(kh) * dilation_h;
                            if (ih < 0 || ih >= static_cast<ptrdiff_t>(H)) {
                                continue;
                            }
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const ptrdiff_t iw = iw_base + static_cast<ptrdiff_t>(kw * dilation);
                                if (iw < 0 || iw >= static_cast<ptrdiff_t>(W)) {
                                    continue;
                                }
                                const uint8_t* src_ptr =
                                    src_n + static_cast<size_t>(ih) * stride_h + static_cast<size_t>(iw) * stride_w;
                                const int8_t* wei_ptr = wei_oc + (kh * KW + kw) * IC;
                                dot_ker(src_ptr, wei_ptr, dst_ptr + oc, IC, 1);
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

TimerResult bench_our_conv_kxk_brgemm_mbnb(const Args& args, size_t mb, size_t nb) {
    using dnnl::impl::cpu::aarch64::brgemm_batch_element_t;

    if (!has_sve() || args.stride != 1) {
        return {-1.0, -1.0};
    }
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t KH = args.KH;
    const size_t KW = args.KW;
    const size_t OH = conv_out_dim(H, KH, args.stride, args.pad, args.dilation);
    const size_t OW = conv_out_dim(W, KW, args.stride, args.pad, args.dilation);
    if (OH == 0 || OW == 0) {
        return {-1.0, -1.0};
    }
    const ptrdiff_t pad = static_cast<ptrdiff_t>(args.pad);
    const ptrdiff_t dilation_h = static_cast<ptrdiff_t>(args.dilation);
    const ptrdiff_t dilation_w = static_cast<ptrdiff_t>(args.dilation);

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
        ov::intel_cpu::aarch64::BrgemmInt8Kernel brg(mb, nb, IC, IC, OC, OC, args.src_signed);
        ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
        dot.create_ker();
        const auto dot_ker = dot.ker();
        const bool use_onednn_brgemm = brg.uses_brgemm();

        std::vector<brgemm_batch_element_t> batch(KH * KW);
        std::vector<const int8_t*> batch_bases(KH * KW);
        const int8_t* packed_wei_base = use_onednn_brgemm ? packed_wei_brgemm.data() : packed_wei_dot.data();
        const size_t packed_wei_stride = use_onednn_brgemm ? (IC * OC) : (OC * IC);

        auto run = [&]() {
            const size_t ow_full = (OW / mb) * mb;
            const size_t oc_full = (OC / nb) * nb;
            for (size_t n = 0; n < N; ++n) {
                const uint8_t* src_n = src.data() + n * H * W * IC;
                for (size_t h = 0; h < OH; ++h) {
                    const ptrdiff_t ih_base = static_cast<ptrdiff_t>(h) - pad;
                    const bool full_h = ih_base >= 0 &&
                                        (ih_base + static_cast<ptrdiff_t>((KH - 1) * args.dilation)) <
                                            static_cast<ptrdiff_t>(H);
                    size_t w = 0;
                    for (; w < ow_full; w += mb) {
                        const ptrdiff_t iw_base = static_cast<ptrdiff_t>(w) - pad;
                        const bool full_w = iw_base >= 0 &&
                                            (iw_base + static_cast<ptrdiff_t>((mb - 1) + (KW - 1) * args.dilation)) <
                                                static_cast<ptrdiff_t>(W);
                        if (!(full_h && full_w)) {
                            break;
                        }
                        int32_t* dst_block = dst.data() + ((n * OH + h) * OW + w) * OC;
                        size_t idx = 0;
                        for (size_t kh = 0; kh < KH; ++kh) {
                            const ptrdiff_t ih = ih_base + static_cast<ptrdiff_t>(kh) * dilation_h;
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const ptrdiff_t iw = iw_base + static_cast<ptrdiff_t>(kw) * dilation_w;
                                const uint8_t* src_ptr =
                                    src_n + (static_cast<size_t>(ih) * W + static_cast<size_t>(iw)) * IC;
                                batch[idx].ptr.A = src_ptr;
                                batch_bases[idx] = packed_wei_base + (kh * KW + kw) * packed_wei_stride;
                                ++idx;
                            }
                        }
                        for (size_t oc = 0; oc < oc_full; oc += nb) {
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
                                    const ptrdiff_t ih = static_cast<ptrdiff_t>(h) - pad +
                                                         static_cast<ptrdiff_t>(kh) * dilation_h;
                                    const ptrdiff_t iw = static_cast<ptrdiff_t>(w) - pad +
                                                         static_cast<ptrdiff_t>(kw) * dilation_w;
                                    if (ih < 0 || ih >= static_cast<ptrdiff_t>(H) || iw < 0 ||
                                        iw >= static_cast<ptrdiff_t>(W)) {
                                        continue;
                                    }
                                    const uint8_t* src_ptr =
                                        src_n + (static_cast<size_t>(ih) * W + static_cast<size_t>(iw)) * IC;
                                    const int8_t* wei_ptr =
                                        packed_wei_dot.data() + ((kh * KW + kw) * OC + oc) * IC;
                                    dot_ker(src_ptr, wei_ptr, dst_ptr + oc, IC, wrote ? 1 : 0);
                                    wrote = true;
                                }
                            }
                            if (!wrote) {
                                dst_ptr[oc] = 0;
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

TimerResult bench_our_conv_kxk_brgemm_offs_mbnb(const Args& args, size_t mb, size_t nb) {
    using dnnl::impl::cpu::aarch64::brgemm_batch_element_t;
    using dnnl::impl::cpu::aarch64::brgemm_offs;

    if (!has_sve() || args.stride != 1) {
        return {-1.0, -1.0};
    }
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t KH = args.KH;
    const size_t KW = args.KW;
    const size_t OH = conv_out_dim(H, KH, args.stride, args.pad, args.dilation);
    const size_t OW = conv_out_dim(W, KW, args.stride, args.pad, args.dilation);
    if (OH == 0 || OW == 0) {
        return {-1.0, -1.0};
    }
    const ptrdiff_t pad = static_cast<ptrdiff_t>(args.pad);
    const ptrdiff_t dilation_h = static_cast<ptrdiff_t>(args.dilation);
    const ptrdiff_t dilation_w = static_cast<ptrdiff_t>(args.dilation);

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

    std::vector<brgemm_batch_element_t> static_offsets(KH * KW);
    for (size_t kh = 0; kh < KH; ++kh) {
        for (size_t kw = 0; kw < KW; ++kw) {
            auto& e = static_offsets[kh * KW + kw];
            e.offset.A = static_cast<ptrdiff_t>((kh * static_cast<size_t>(dilation_h)) * W * IC +
                                                (kw * static_cast<size_t>(dilation_w)) * IC);
            e.offset.B = static_cast<ptrdiff_t>((kh * KW + kw) * IC * OC);
        }
    }

    try {
        ov::intel_cpu::aarch64::BrgemmInt8Kernel brg(mb,
                                                     nb,
                                                     IC,
                                                     IC,
                                                     OC,
                                                     OC,
                                                     args.src_signed,
                                                     brgemm_offs,
                                                     nullptr,
                                                     static_cast<int>(static_offsets.size()));
        if (!brg.uses_brgemm()) {
            return {-1.0, -1.0};
        }

        ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
        dot.create_ker();
        const auto dot_ker = dot.ker();

        auto run = [&]() {
            const size_t ow_full = (OW / mb) * mb;
            const size_t oc_full = (OC / nb) * nb;
            for (size_t n = 0; n < N; ++n) {
                const uint8_t* src_n = src.data() + n * H * W * IC;
                for (size_t h = 0; h < OH; ++h) {
                    const ptrdiff_t ih_base = static_cast<ptrdiff_t>(h) - pad;
                    const bool full_h = ih_base >= 0 &&
                                        (ih_base + static_cast<ptrdiff_t>((KH - 1) * args.dilation)) <
                                            static_cast<ptrdiff_t>(H);
                    size_t w = 0;
                    for (; w < ow_full; w += mb) {
                        const ptrdiff_t iw_base = static_cast<ptrdiff_t>(w) - pad;
                        const bool full_w = iw_base >= 0 &&
                                            (iw_base + static_cast<ptrdiff_t>((mb - 1) + (KW - 1) * args.dilation)) <
                                                static_cast<ptrdiff_t>(W);
                        if (!(full_h && full_w)) {
                            break;
                        }
                        const uint8_t* src_ptr =
                            src_n + (static_cast<size_t>(ih_base) * W + static_cast<size_t>(iw_base)) * IC;
                        int32_t* dst_block = dst.data() + ((n * OH + h) * OW + w) * OC;
                        for (size_t oc = 0; oc < oc_full; oc += nb) {
                            brg.execute_batch_offsets(src_ptr,
                                                      packed_wei_brgemm.data() + oc,
                                                      static_offsets.data(),
                                                      static_cast<int>(static_offsets.size()),
                                                      dst_block + oc);
                        }
                        for (size_t oc = oc_full; oc < OC; ++oc) {
                            int32_t* dst_ptr = dst_block + oc;
                            bool wrote = false;
                            for (size_t kh = 0; kh < KH; ++kh) {
                                for (size_t kw = 0; kw < KW; ++kw) {
                                    const uint8_t* src_tail =
                                        src_ptr + static_cast<size_t>(static_offsets[kh * KW + kw].offset.A);
                                    const int8_t* wei_ptr =
                                        packed_wei_dot.data() + ((kh * KW + kw) * OC + oc) * IC;
                                    dot_ker(src_tail, wei_ptr, dst_ptr, IC, wrote ? 1 : 0);
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
                                    const ptrdiff_t ih = static_cast<ptrdiff_t>(h) - pad +
                                                         static_cast<ptrdiff_t>(kh) * dilation_h;
                                    const ptrdiff_t iw = static_cast<ptrdiff_t>(w) - pad +
                                                         static_cast<ptrdiff_t>(kw) * dilation_w;
                                    if (ih < 0 || ih >= static_cast<ptrdiff_t>(H) || iw < 0 ||
                                        iw >= static_cast<ptrdiff_t>(W)) {
                                        continue;
                                    }
                                    const uint8_t* src_tail =
                                        src_n + (static_cast<size_t>(ih) * W + static_cast<size_t>(iw)) * IC;
                                    const int8_t* wei_ptr =
                                        packed_wei_dot.data() + ((kh * KW + kw) * OC + oc) * IC;
                                    dot_ker(src_tail, wei_ptr, dst_ptr + oc, IC, wrote ? 1 : 0);
                                    wrote = true;
                                }
                            }
                            if (!wrote) {
                                dst_ptr[oc] = 0;
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

TimerResult bench_our_conv_kxk_brgemm_im2col(const Args& args, size_t mb) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t KH = args.KH;
    const size_t KW = args.KW;
    const size_t OH = H - KH + 1;
    const size_t OW = W - KW + 1;
    const size_t K = IC * KH * KW;
    const size_t M = N * OH * OW;
    if (M == 0 || K == 0 || OC == 0) {
        return {-1.0, -1.0};
    }

    std::vector<uint8_t> src_u8;
    std::vector<int8_t> src_s8;
    std::vector<int8_t> wei(OC * IC * KH * KW);
    std::vector<int32_t> dst(M * OC);
    std::mt19937 gen(42);
    if (args.src_signed) {
        src_s8.resize(N * H * W * IC);
        fill_random(src_s8, -10, 10, gen);
    } else {
        src_u8.resize(N * H * W * IC);
        fill_random(src_u8, 0, 10, gen);
    }
    fill_random(wei, -10, 10, gen);

    std::vector<int8_t> packed_wei(K * OC);
    for (size_t oc = 0; oc < OC; ++oc) {
        size_t k = 0;
        for (size_t kh = 0; kh < KH; ++kh) {
            for (size_t kw = 0; kw < KW; ++kw) {
                for (size_t ic = 0; ic < IC; ++ic) {
                    packed_wei[k * OC + oc] = wei[((oc * IC + ic) * KH + kh) * KW + kw];
                    ++k;
                }
            }
        }
    }

    const size_t mb_eff = std::min(mb, M);
    const size_t M_full = (M / mb_eff) * mb_eff;
    const size_t M_tail = M - M_full;

    ov::intel_cpu::aarch64::BrgemmInt8Kernel brg(mb_eff, OC, K, K, OC, OC, args.src_signed);
    ov::intel_cpu::aarch64::BrgemmInt8Kernel brg_tail(
        M_tail == 0 ? 1 : M_tail, OC, K, K, OC, OC, args.src_signed);

    std::vector<uint8_t> a_u8(mb_eff * K);
    std::vector<int8_t> a_s8(mb_eff * K);
    std::vector<uint8_t> a_tail_u8(M_tail * K);
    std::vector<int8_t> a_tail_s8(M_tail * K);

    auto pack_row = [&](size_t row_idx, uint8_t* dst_row_u8, int8_t* dst_row_s8) {
        const size_t oh = (row_idx / OW) % OH;
        const size_t ow = row_idx % OW;
        const size_t n = row_idx / (OH * OW);
        const uint8_t* src_n_u8 = args.src_signed ? nullptr : (src_u8.data() + n * H * W * IC);
        const int8_t* src_n_s8 = args.src_signed ? (src_s8.data() + n * H * W * IC) : nullptr;
        size_t col = 0;
        for (size_t kh = 0; kh < KH; ++kh) {
            const size_t src_h = (oh + kh) * W * IC;
            for (size_t kw = 0; kw < KW; ++kw) {
                if (args.src_signed) {
                    const int8_t* src_ptr = src_n_s8 + src_h + (ow + kw) * IC;
                    std::memcpy(dst_row_s8 + col, src_ptr, IC);
                    col += IC;
                } else {
                    const uint8_t* src_ptr = src_n_u8 + src_h + (ow + kw) * IC;
                    std::memcpy(dst_row_u8 + col, src_ptr, IC);
                    col += IC;
                }
            }
        }
    };

    auto run = [&]() {
        for (size_t m = 0; m < M_full; m += mb_eff) {
            if (args.src_signed) {
                for (size_t r = 0; r < mb_eff; ++r) {
                    pack_row(m + r, nullptr, a_s8.data() + r * K);
                }
                brg.execute(a_s8.data(), packed_wei.data(), dst.data() + m * OC);
            } else {
                for (size_t r = 0; r < mb_eff; ++r) {
                    pack_row(m + r, a_u8.data() + r * K, nullptr);
                }
                brg.execute(a_u8.data(), packed_wei.data(), dst.data() + m * OC);
            }
        }
        if (M_tail) {
            const size_t base = M_full;
            if (args.src_signed) {
                for (size_t r = 0; r < M_tail; ++r) {
                    pack_row(base + r, nullptr, a_tail_s8.data() + r * K);
                }
                brg_tail.execute(a_tail_s8.data(), packed_wei.data(), dst.data() + base * OC);
            } else {
                for (size_t r = 0; r < M_tail; ++r) {
                    pack_row(base + r, a_tail_u8.data() + r * K, nullptr);
                }
                brg_tail.execute(a_tail_u8.data(), packed_wei.data(), dst.data() + base * OC);
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(OC) * static_cast<double>(K);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_kxk_mmla_im2col(const Args& args) {
    if (!has_i8mm()) {
        return {-1.0, -1.0};
    }
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t KH = args.KH;
    const size_t KW = args.KW;
    const size_t OH = conv_out_dim(H, KH, args.stride, args.pad, args.dilation);
    const size_t OW = conv_out_dim(W, KW, args.stride, args.pad, args.dilation);
    const size_t K = IC * KH * KW;
    const size_t M = N * OH * OW;
    if (M == 0 || K == 0 || OC == 0 || (K % 8) != 0) {
        return {-1.0, -1.0};
    }

    std::vector<uint8_t> src_u8;
    std::vector<int8_t> src_s8;
    std::vector<int8_t> wei(OC * IC * KH * KW);
    std::vector<int32_t> dst(M * OC);
    std::mt19937 gen(42);
    if (args.src_signed) {
        src_s8.resize(N * H * W * IC);
        fill_random(src_s8, -10, 10, gen);
    } else {
        src_u8.resize(N * H * W * IC);
        fill_random(src_u8, 0, 10, gen);
    }
    fill_random(wei, -10, 10, gen);

    std::vector<int8_t> B(OC * K);
    for (size_t oc = 0; oc < OC; ++oc) {
        size_t k = 0;
        for (size_t kh = 0; kh < KH; ++kh) {
            for (size_t kw = 0; kw < KW; ++kw) {
                for (size_t ic = 0; ic < IC; ++ic) {
                    B[oc * K + k] = wei[((oc * IC + ic) * KH + kh) * KW + kw];
                    ++k;
                }
            }
        }
    }

    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
    dot.create_ker();
    const auto dot_ker = dot.ker();

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x8_usmmla_packed ker8x8_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x12_usmmla_packed ker8x12_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_usmmla_packed ker4x16_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_usmmla_packed ker4x8_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_usmmla_packed ker4x4_u8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x8_smmla_packed ker8x8_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x12_smmla_packed ker8x12_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_smmla_packed ker4x16_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_smmla_packed ker4x8_s8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_smmla_packed ker4x4_s8;
    if (args.src_signed) {
        ker8x8_s8.create_ker();
        ker8x12_s8.create_ker();
        ker4x16_s8.create_ker();
        ker4x8_s8.create_ker();
        ker4x4_s8.create_ker();
    } else {
        ker8x8_u8.create_ker();
        ker8x12_u8.create_ker();
        ker4x16_u8.create_ker();
        ker4x8_u8.create_ker();
        ker4x4_u8.create_ker();
    }

    const size_t Kp = round_up(K, 8);
    const size_t n12_blocks = OC / 12;
    const size_t n12_main = n12_blocks * 12;
    const size_t rem12 = OC - n12_main;
    const bool use_block12 = (n12_blocks > 0);
    const size_t stride8 = packed_block_stride_mmla(K, 8);
    const size_t stride12 = packed_block_stride_mmla(K, 12);
    const size_t stride16 = packed_block_stride_mmla(K, 16);
    const size_t stride4 = packed_block_stride_mmla(K, 4);
    if (stride8 == 0 || stride4 == 0) {
        return {-1.0, -1.0};
    }

    std::vector<int8_t> B_packed8_main;
    std::vector<int8_t> B_packed12;
    std::vector<int8_t> B_packed8_tail12;
    std::vector<int8_t> B_packed4_tail12;
    std::vector<int8_t> B_packed16_tail;
    std::vector<int8_t> B_packed8_tail;
    std::vector<int8_t> B_packed4_tail;

    if (use_block12) {
        B_packed12.resize(n12_blocks * stride12);
        for (size_t nb = 0; nb < n12_blocks; ++nb) {
            const int8_t* src_block = B.data() + nb * 12 * K;
            int8_t* dst_block = B_packed12.data() + nb * stride12;
            pack_mmla_block(src_block, K, Kp, 12, dst_block);
        }
    } else {
        const size_t n_blocks8 = OC / 8;
        B_packed8_main.resize(n_blocks8 * stride8);
        for (size_t nb = 0; nb < n_blocks8; ++nb) {
            const int8_t* src_block = B.data() + nb * 8 * K;
            int8_t* dst_block = B_packed8_main.data() + nb * stride8;
            pack_mmla_block(src_block, K, Kp, 8, dst_block);
        }
    }
    if (rem12 >= 8) {
        B_packed8_tail12.resize(stride8);
        const int8_t* src_block = B.data() + n12_main * K;
        pack_mmla_block(src_block, K, Kp, 8, B_packed8_tail12.data());
    }
    if (rem12 >= 4) {
        const size_t tail4_off = n12_main + ((rem12 >= 8) ? 8 : 0);
        B_packed4_tail12.resize(stride4);
        const int8_t* src_block = B.data() + tail4_off * K;
        pack_mmla_block(src_block, K, Kp, 4, B_packed4_tail12.data());
    }

    const size_t n_blocks16 = OC / 16;
    const size_t n_blocks8_tail = (OC % 16) / 8;
    const size_t n_blocks4_tail = (OC % 8) / 4;
    if (n_blocks16 > 0) {
        B_packed16_tail.resize(n_blocks16 * stride16);
        for (size_t nb = 0; nb < n_blocks16; ++nb) {
            const int8_t* src_block = B.data() + nb * 16 * K;
            int8_t* dst_block = B_packed16_tail.data() + nb * stride16;
            pack_mmla_block(src_block, K, Kp, 16, dst_block);
        }
    }
    if (n_blocks8_tail > 0) {
        B_packed8_tail.resize(n_blocks8_tail * stride8);
        for (size_t nb = 0; nb < n_blocks8_tail; ++nb) {
            const int8_t* src_block = B.data() + (n_blocks16 * 16 + nb * 8) * K;
            int8_t* dst_block = B_packed8_tail.data() + nb * stride8;
            pack_mmla_block(src_block, K, Kp, 8, dst_block);
        }
    }
    if (n_blocks4_tail > 0) {
        B_packed4_tail.resize(n_blocks4_tail * stride4);
        for (size_t nb = 0; nb < n_blocks4_tail; ++nb) {
            const int8_t* src_block =
                B.data() + (n_blocks16 * 16 + n_blocks8_tail * 8 + nb * 4) * K;
            int8_t* dst_block = B_packed4_tail.data() + nb * stride4;
            pack_mmla_block(src_block, K, Kp, 4, dst_block);
        }
    }

    std::vector<uint8_t> A_u8;
    std::vector<int8_t> A_s8;
    if (args.src_signed) {
        A_s8.resize(M * K);
    } else {
        A_u8.resize(M * K);
    }

    auto pack_row_u8 = [&](size_t row_idx, uint8_t* dst_row_u8) {
        const size_t oh = (row_idx / OW) % OH;
        const size_t ow = row_idx % OW;
        const size_t n = row_idx / (OH * OW);
        const uint8_t* src_n_u8 = src_u8.data() + n * H * W * IC;
        const ptrdiff_t ih0 = static_cast<ptrdiff_t>(oh * args.stride) - static_cast<ptrdiff_t>(args.pad);
        const ptrdiff_t iw0 = static_cast<ptrdiff_t>(ow * args.stride) - static_cast<ptrdiff_t>(args.pad);
        const bool fast_contig = (args.dilation == 1) && ih0 >= 0 && iw0 >= 0 &&
                                 (ih0 + static_cast<ptrdiff_t>(KH)) <= static_cast<ptrdiff_t>(H) &&
                                 (iw0 + static_cast<ptrdiff_t>(KW)) <= static_cast<ptrdiff_t>(W);

        size_t col = 0;
        if (fast_contig) {
            const size_t row_bytes = KW * IC;
            const size_t iw = static_cast<size_t>(iw0);
            const size_t ih_base = static_cast<size_t>(ih0);
            for (size_t kh = 0; kh < KH; ++kh) {
                const uint8_t* src_ptr = src_n_u8 + ((ih_base + kh) * W + iw) * IC;
                std::memcpy(dst_row_u8 + col, src_ptr, row_bytes);
                col += row_bytes;
            }
            return;
        }
        if (args.dilation == 1) {
            for (size_t kh = 0; kh < KH; ++kh) {
                uint8_t* out_row = dst_row_u8 + col;
                const ptrdiff_t ih = ih0 + static_cast<ptrdiff_t>(kh);
                if (ih < 0 || ih >= static_cast<ptrdiff_t>(H)) {
                    std::memset(out_row, 0, KW * IC);
                    col += KW * IC;
                    continue;
                }
                const ptrdiff_t iw = iw0;
                ptrdiff_t left = (iw < 0) ? -iw : 0;
                ptrdiff_t right = 0;
                const ptrdiff_t iw_end = iw + static_cast<ptrdiff_t>(KW);
                if (iw_end > static_cast<ptrdiff_t>(W)) {
                    right = iw_end - static_cast<ptrdiff_t>(W);
                }
                if (left > static_cast<ptrdiff_t>(KW)) {
                    left = static_cast<ptrdiff_t>(KW);
                }
                if (right > static_cast<ptrdiff_t>(KW)) {
                    right = static_cast<ptrdiff_t>(KW);
                }
                const size_t valid = KW - static_cast<size_t>(left) - static_cast<size_t>(right);
                if (left) {
                    std::memset(out_row, 0, static_cast<size_t>(left) * IC);
                }
                if (valid) {
                    const size_t iw_src = static_cast<size_t>(std::max<ptrdiff_t>(0, iw));
                    const uint8_t* src_ptr =
                        src_n_u8 + ((static_cast<size_t>(ih) * W + iw_src) * IC);
                    std::memcpy(out_row + static_cast<size_t>(left) * IC, src_ptr, valid * IC);
                }
                if (right) {
                    std::memset(out_row + (static_cast<size_t>(left) + valid) * IC, 0, static_cast<size_t>(right) * IC);
                }
                col += KW * IC;
            }
            return;
        }

        for (size_t kh = 0; kh < KH; ++kh) {
            const ptrdiff_t ih = ih0 + static_cast<ptrdiff_t>(kh * args.dilation);
            for (size_t kw = 0; kw < KW; ++kw) {
                const ptrdiff_t iw = iw0 + static_cast<ptrdiff_t>(kw * args.dilation);
                uint8_t* out = dst_row_u8 + col;
                if (ih >= 0 && iw >= 0 && ih < static_cast<ptrdiff_t>(H) && iw < static_cast<ptrdiff_t>(W)) {
                    const uint8_t* src_ptr = src_n_u8 + ((static_cast<size_t>(ih) * W + static_cast<size_t>(iw)) * IC);
                    std::memcpy(out, src_ptr, IC);
                } else {
                    std::memset(out, 0, IC);
                }
                col += IC;
            }
        }
    };

    auto pack_row_s8 = [&](size_t row_idx, int8_t* dst_row_s8) {
        const size_t oh = (row_idx / OW) % OH;
        const size_t ow = row_idx % OW;
        const size_t n = row_idx / (OH * OW);
        const int8_t* src_n_s8 = src_s8.data() + n * H * W * IC;
        const ptrdiff_t ih0 = static_cast<ptrdiff_t>(oh * args.stride) - static_cast<ptrdiff_t>(args.pad);
        const ptrdiff_t iw0 = static_cast<ptrdiff_t>(ow * args.stride) - static_cast<ptrdiff_t>(args.pad);
        const bool fast_contig = (args.dilation == 1) && ih0 >= 0 && iw0 >= 0 &&
                                 (ih0 + static_cast<ptrdiff_t>(KH)) <= static_cast<ptrdiff_t>(H) &&
                                 (iw0 + static_cast<ptrdiff_t>(KW)) <= static_cast<ptrdiff_t>(W);

        size_t col = 0;
        if (fast_contig) {
            const size_t row_bytes = KW * IC;
            const size_t iw = static_cast<size_t>(iw0);
            const size_t ih_base = static_cast<size_t>(ih0);
            for (size_t kh = 0; kh < KH; ++kh) {
                const int8_t* src_ptr = src_n_s8 + ((ih_base + kh) * W + iw) * IC;
                std::memcpy(dst_row_s8 + col, src_ptr, row_bytes);
                col += row_bytes;
            }
            return;
        }
        if (args.dilation == 1) {
            for (size_t kh = 0; kh < KH; ++kh) {
                int8_t* out_row = dst_row_s8 + col;
                const ptrdiff_t ih = ih0 + static_cast<ptrdiff_t>(kh);
                if (ih < 0 || ih >= static_cast<ptrdiff_t>(H)) {
                    std::memset(out_row, 0, KW * IC);
                    col += KW * IC;
                    continue;
                }
                const ptrdiff_t iw = iw0;
                ptrdiff_t left = (iw < 0) ? -iw : 0;
                ptrdiff_t right = 0;
                const ptrdiff_t iw_end = iw + static_cast<ptrdiff_t>(KW);
                if (iw_end > static_cast<ptrdiff_t>(W)) {
                    right = iw_end - static_cast<ptrdiff_t>(W);
                }
                if (left > static_cast<ptrdiff_t>(KW)) {
                    left = static_cast<ptrdiff_t>(KW);
                }
                if (right > static_cast<ptrdiff_t>(KW)) {
                    right = static_cast<ptrdiff_t>(KW);
                }
                const size_t valid = KW - static_cast<size_t>(left) - static_cast<size_t>(right);
                if (left) {
                    std::memset(out_row, 0, static_cast<size_t>(left) * IC);
                }
                if (valid) {
                    const size_t iw_src = static_cast<size_t>(std::max<ptrdiff_t>(0, iw));
                    const int8_t* src_ptr =
                        src_n_s8 + ((static_cast<size_t>(ih) * W + iw_src) * IC);
                    std::memcpy(out_row + static_cast<size_t>(left) * IC, src_ptr, valid * IC);
                }
                if (right) {
                    std::memset(out_row + (static_cast<size_t>(left) + valid) * IC, 0, static_cast<size_t>(right) * IC);
                }
                col += KW * IC;
            }
            return;
        }

        for (size_t kh = 0; kh < KH; ++kh) {
            const ptrdiff_t ih = ih0 + static_cast<ptrdiff_t>(kh * args.dilation);
            for (size_t kw = 0; kw < KW; ++kw) {
                const ptrdiff_t iw = iw0 + static_cast<ptrdiff_t>(kw * args.dilation);
                int8_t* out = dst_row_s8 + col;
                if (ih >= 0 && iw >= 0 && ih < static_cast<ptrdiff_t>(H) && iw < static_cast<ptrdiff_t>(W)) {
                    const int8_t* src_ptr =
                        src_n_s8 + ((static_cast<size_t>(ih) * W + static_cast<size_t>(iw)) * IC);
                    std::memcpy(out, src_ptr, IC);
                } else {
                    std::memset(out, 0, IC);
                }
                col += IC;
            }
        }
    };

    auto run = [&]() {
        if (args.src_signed) {
            for (size_t m = 0; m < M; ++m) {
                pack_row_s8(m, A_s8.data() + m * K);
            }
            size_t m = 0;
            for (; m + 8 <= M; m += 8) {
                const int8_t* a_ptrs_s8[8] = {
                    A_s8.data() + (m + 0) * K,
                    A_s8.data() + (m + 1) * K,
                    A_s8.data() + (m + 2) * K,
                    A_s8.data() + (m + 3) * K,
                    A_s8.data() + (m + 4) * K,
                    A_s8.data() + (m + 5) * K,
                    A_s8.data() + (m + 6) * K,
                    A_s8.data() + (m + 7) * K,
                };
                const int8_t* a_ptrs_s8_lo[4] = {a_ptrs_s8[0], a_ptrs_s8[1], a_ptrs_s8[2], a_ptrs_s8[3]};
                const int8_t* a_ptrs_s8_hi[4] = {a_ptrs_s8[4], a_ptrs_s8[5], a_ptrs_s8[6], a_ptrs_s8[7]};

                if (use_block12) {
                    for (size_t n = 0; n + 12 <= n12_main; n += 12) {
                        const int8_t* b_ptr = B_packed12.data() + (n / 12) * stride12;
                        int32_t* c_ptr = dst.data() + m * OC + n;
                        ker8x12_s8.ker()(a_ptrs_s8, b_ptr, c_ptr, K, 0, OC * sizeof(int32_t), 0);
                    }
                    if (rem12 >= 8) {
                        const int8_t* b_ptr = B_packed8_tail12.data();
                        int32_t* c_ptr = dst.data() + m * OC + n12_main;
                        ker8x8_s8.ker()(a_ptrs_s8, b_ptr, c_ptr, K, 0, OC * sizeof(int32_t), 0);
                    }
                    if (rem12 >= 4) {
                        const size_t tail4_off = n12_main + ((rem12 >= 8) ? 8 : 0);
                        const int8_t* b_ptr = B_packed4_tail12.data();
                        int32_t* c_ptr0 = dst.data() + m * OC + tail4_off;
                        int32_t* c_ptr1 = dst.data() + (m + 4) * OC + tail4_off;
                        ker4x4_s8.ker()(a_ptrs_s8_lo, b_ptr, c_ptr0, K, 0, OC * sizeof(int32_t), 0);
                        ker4x4_s8.ker()(a_ptrs_s8_hi, b_ptr, c_ptr1, K, 0, OC * sizeof(int32_t), 0);
                    }
                } else {
                    for (size_t n = 0; n + 8 <= OC; n += 8) {
                        const int8_t* b_ptr = B_packed8_main.data() + (n / 8) * stride8;
                        int32_t* c_ptr = dst.data() + m * OC + n;
                        ker8x8_s8.ker()(a_ptrs_s8, b_ptr, c_ptr, K, 0, OC * sizeof(int32_t), 0);
                    }
                    if ((OC % 8) >= 4) {
                        const size_t tail4_off = (OC / 8) * 8;
                        const int8_t* b_ptr = B_packed4_tail12.data();
                        int32_t* c_ptr0 = dst.data() + m * OC + tail4_off;
                        int32_t* c_ptr1 = dst.data() + (m + 4) * OC + tail4_off;
                        ker4x4_s8.ker()(a_ptrs_s8_lo, b_ptr, c_ptr0, K, 0, OC * sizeof(int32_t), 0);
                        ker4x4_s8.ker()(a_ptrs_s8_hi, b_ptr, c_ptr1, K, 0, OC * sizeof(int32_t), 0);
                    }
                }
            }

            for (; m + 4 <= M; m += 4) {
                const int8_t* a_ptrs_s8[4] = {
                    A_s8.data() + (m + 0) * K,
                    A_s8.data() + (m + 1) * K,
                    A_s8.data() + (m + 2) * K,
                    A_s8.data() + (m + 3) * K,
                };

                size_t n = 0;
                for (; n + 16 <= OC; n += 16) {
                    const int8_t* b_ptr = B_packed16_tail.data() + (n / 16) * stride16;
                    int32_t* c_ptr = dst.data() + m * OC + n;
                    ker4x16_s8.ker()(a_ptrs_s8, b_ptr, c_ptr, K, 0, OC * sizeof(int32_t), 0);
                }
                for (; n + 8 <= OC; n += 8) {
                    const size_t idx = (n - n_blocks16 * 16) / 8;
                    const int8_t* b_ptr = B_packed8_tail.data() + idx * stride8;
                    int32_t* c_ptr = dst.data() + m * OC + n;
                    ker4x8_s8.ker()(a_ptrs_s8, b_ptr, c_ptr, K, 0, OC * sizeof(int32_t), 0);
                }
                for (; n + 4 <= OC; n += 4) {
                    const size_t tail4_idx = (n - n_blocks16 * 16 - n_blocks8_tail * 8) / 4;
                    const int8_t* b_ptr = B_packed4_tail.data() + tail4_idx * stride4;
                    int32_t* c_ptr = dst.data() + m * OC + n;
                    ker4x4_s8.ker()(a_ptrs_s8, b_ptr, c_ptr, K, 0, OC * sizeof(int32_t), 0);
                }
            }

            for (; m < M; ++m) {
                const uint8_t* a_ptr_s8 = reinterpret_cast<const uint8_t*>(A_s8.data() + m * K);
                for (size_t n = 0; n < OC; ++n) {
                    int32_t* c_ptr = dst.data() + m * OC + n;
                    const int8_t* b_ptr = B.data() + n * K;
                    dot_ker(a_ptr_s8, b_ptr, c_ptr, K, 0);
                }
            }
        } else {
            for (size_t m = 0; m < M; ++m) {
                pack_row_u8(m, A_u8.data() + m * K);
            }
            size_t m = 0;
            for (; m + 8 <= M; m += 8) {
                const uint8_t* a_ptrs_u8[8] = {
                    A_u8.data() + (m + 0) * K,
                    A_u8.data() + (m + 1) * K,
                    A_u8.data() + (m + 2) * K,
                    A_u8.data() + (m + 3) * K,
                    A_u8.data() + (m + 4) * K,
                    A_u8.data() + (m + 5) * K,
                    A_u8.data() + (m + 6) * K,
                    A_u8.data() + (m + 7) * K,
                };
                const uint8_t* a_ptrs_u8_lo[4] = {a_ptrs_u8[0], a_ptrs_u8[1], a_ptrs_u8[2], a_ptrs_u8[3]};
                const uint8_t* a_ptrs_u8_hi[4] = {a_ptrs_u8[4], a_ptrs_u8[5], a_ptrs_u8[6], a_ptrs_u8[7]};

                if (use_block12) {
                    for (size_t n = 0; n + 12 <= n12_main; n += 12) {
                        const int8_t* b_ptr = B_packed12.data() + (n / 12) * stride12;
                        int32_t* c_ptr = dst.data() + m * OC + n;
                        ker8x12_u8.ker()(a_ptrs_u8, b_ptr, c_ptr, K, 0, OC * sizeof(int32_t), 0);
                    }
                    if (rem12 >= 8) {
                        const int8_t* b_ptr = B_packed8_tail12.data();
                        int32_t* c_ptr = dst.data() + m * OC + n12_main;
                        ker8x8_u8.ker()(a_ptrs_u8, b_ptr, c_ptr, K, 0, OC * sizeof(int32_t), 0);
                    }
                    if (rem12 >= 4) {
                        const size_t tail4_off = n12_main + ((rem12 >= 8) ? 8 : 0);
                        const int8_t* b_ptr = B_packed4_tail12.data();
                        int32_t* c_ptr0 = dst.data() + m * OC + tail4_off;
                        int32_t* c_ptr1 = dst.data() + (m + 4) * OC + tail4_off;
                        ker4x4_u8.ker()(a_ptrs_u8_lo, b_ptr, c_ptr0, K, 0, OC * sizeof(int32_t), 0);
                        ker4x4_u8.ker()(a_ptrs_u8_hi, b_ptr, c_ptr1, K, 0, OC * sizeof(int32_t), 0);
                    }
                } else {
                    for (size_t n = 0; n + 8 <= OC; n += 8) {
                        const int8_t* b_ptr = B_packed8_main.data() + (n / 8) * stride8;
                        int32_t* c_ptr = dst.data() + m * OC + n;
                        ker8x8_u8.ker()(a_ptrs_u8, b_ptr, c_ptr, K, 0, OC * sizeof(int32_t), 0);
                    }
                    if ((OC % 8) >= 4) {
                        const size_t tail4_off = (OC / 8) * 8;
                        const int8_t* b_ptr = B_packed4_tail12.data();
                        int32_t* c_ptr0 = dst.data() + m * OC + tail4_off;
                        int32_t* c_ptr1 = dst.data() + (m + 4) * OC + tail4_off;
                        ker4x4_u8.ker()(a_ptrs_u8_lo, b_ptr, c_ptr0, K, 0, OC * sizeof(int32_t), 0);
                        ker4x4_u8.ker()(a_ptrs_u8_hi, b_ptr, c_ptr1, K, 0, OC * sizeof(int32_t), 0);
                    }
                }
            }

            for (; m + 4 <= M; m += 4) {
                const uint8_t* a_ptrs_u8[4] = {
                    A_u8.data() + (m + 0) * K,
                    A_u8.data() + (m + 1) * K,
                    A_u8.data() + (m + 2) * K,
                    A_u8.data() + (m + 3) * K,
                };

                size_t n = 0;
                for (; n + 16 <= OC; n += 16) {
                    const int8_t* b_ptr = B_packed16_tail.data() + (n / 16) * stride16;
                    int32_t* c_ptr = dst.data() + m * OC + n;
                    ker4x16_u8.ker()(a_ptrs_u8, b_ptr, c_ptr, K, 0, OC * sizeof(int32_t), 0);
                }
                for (; n + 8 <= OC; n += 8) {
                    const size_t idx = (n - n_blocks16 * 16) / 8;
                    const int8_t* b_ptr = B_packed8_tail.data() + idx * stride8;
                    int32_t* c_ptr = dst.data() + m * OC + n;
                    ker4x8_u8.ker()(a_ptrs_u8, b_ptr, c_ptr, K, 0, OC * sizeof(int32_t), 0);
                }
                for (; n + 4 <= OC; n += 4) {
                    const size_t tail4_idx = (n - n_blocks16 * 16 - n_blocks8_tail * 8) / 4;
                    const int8_t* b_ptr = B_packed4_tail.data() + tail4_idx * stride4;
                    int32_t* c_ptr = dst.data() + m * OC + n;
                    ker4x4_u8.ker()(a_ptrs_u8, b_ptr, c_ptr, K, 0, OC * sizeof(int32_t), 0);
                }
            }

            for (; m < M; ++m) {
                const uint8_t* a_ptr_u8 = A_u8.data() + m * K;
                for (size_t n = 0; n < OC; ++n) {
                    int32_t* c_ptr = dst.data() + m * OC + n;
                    const int8_t* b_ptr = B.data() + n * K;
                    dot_ker(a_ptr_u8, b_ptr, c_ptr, K, 0);
                }
            }
        }
    };

    for (size_t i = 0; i < args.warmup; ++i) {
        run();
    }

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(OC) * static_cast<double>(K);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_our_conv_kxk_brgemm(const Args& args) {
    return bench_our_conv_kxk_brgemm_mbnb(args, kBrgemmMB, kBrgemmNB);
}

TimerResult bench_our_conv_kxk_brgemm_fused_mbnb(const Args& args, size_t mb, size_t nb) {
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
    const size_t k_total = IC * KH * KW;
    if (OW < mb || (OC % nb) != 0) {
        return {-1.0, -1.0};
    }

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC * KH * KW);
    std::vector<int32_t> dst(N * OH * OW * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    std::vector<int8_t> packed_wei(k_total * OC);
    for (size_t oc = 0; oc < OC; ++oc) {
        size_t k = 0;
        for (size_t kh = 0; kh < KH; ++kh) {
            for (size_t kw = 0; kw < KW; ++kw) {
                for (size_t ic = 0; ic < IC; ++ic) {
                    packed_wei[k * OC + oc] = wei[((oc * IC + ic) * KH + kh) * KW + kw];
                    ++k;
                }
            }
        }
    }

    try {
        ov::intel_cpu::aarch64::BrgemmInt8Kernel brg(mb, nb, k_total, k_total, OC, OC, args.src_signed);
        if (!brg.uses_brgemm()) {
            return {-1.0, -1.0};
        }
        ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
        dot.create_ker();
        const auto dot_ker = dot.ker();

        std::vector<uint8_t> packed_src(mb * k_total);

        auto run = [&]() {
            const size_t ow_full = (OW / mb) * mb;
            for (size_t n = 0; n < N; ++n) {
                const uint8_t* src_n = src.data() + n * H * W * IC;
                for (size_t h = 0; h < OH; ++h) {
                    size_t w = 0;
                    for (; w < ow_full; w += mb) {
                        for (size_t m = 0; m < mb; ++m) {
                            uint8_t* dst_row = packed_src.data() + m * k_total;
                            const uint8_t* src_base = src_n + (h * W + (w + m)) * IC;
                            for (size_t kh = 0; kh < KH; ++kh) {
                                const uint8_t* src_ptr = src_base + kh * W * IC;
                                uint8_t* dst_ptr = dst_row + (kh * KW) * IC;
                                std::memcpy(dst_ptr, src_ptr, KW * IC);
                            }
                        }
                        int32_t* dst_ptr = dst.data() + ((n * OH + h) * OW + w) * OC;
                        for (size_t oc = 0; oc < OC; oc += nb) {
                            brg.execute(packed_src.data(), packed_wei.data() + oc, dst_ptr + oc);
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
                                        packed_wei.data() + ((kh * KW + kw) * IC) * OC + oc;
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
    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
    dot.create_ker();
    const auto dot_ker = dot.ker();

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
                for (; w < W; ++w) {
                    const uint8_t* src_ptr = src.data() + ((n * H + h) * W + w) * IC;
                    int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC;
                    for (size_t oc = 0; oc < OC; ++oc) {
                        dot_ker(src_ptr, wei.data() + oc * IC, dst_ptr + oc, IC, 0);
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
    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
    dot.create_ker();
    const auto dot_ker = dot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                size_t w = 0;
                for (; w + 4 <= W; w += 4) {
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
                for (; w < W; ++w) {
                    const uint8_t* src_ptr = src.data() + ((n * H + h) * W + w) * IC;
                    int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC;
                    for (size_t oc = 0; oc < OC; ++oc) {
                        dot_ker(src_ptr, wei.data() + oc * IC, dst_ptr + oc, IC, 0);
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
    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
    dot.create_ker();
    const auto dot_ker = dot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                size_t w = 0;
                for (; w + 4 <= W; w += 4) {
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
                for (; w < W; ++w) {
                    const uint8_t* src_ptr = src.data() + ((n * H + h) * W + w) * IC;
                    int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC;
                    for (size_t oc = 0; oc < OC; ++oc) {
                        dot_ker(src_ptr, wei.data() + oc * IC, dst_ptr + oc, IC, 0);
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
    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
    dot.create_ker();
    const auto dot_ker = dot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                size_t w = 0;
                for (; w + 4 <= W; w += 4) {
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
                for (; w < W; ++w) {
                    const uint8_t* src_ptr = src.data() + ((n * H + h) * W + w) * IC;
                    int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC;
                    for (size_t oc = 0; oc < OC; ++oc) {
                        dot_ker(src_ptr, wei.data() + oc * IC, dst_ptr + oc, IC, 0);
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

TimerResult bench_our_conv_1x1_block8x8_mmla_packed(const Args& args, bool executor_like_u8 = false) {
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
    const auto wei_comp =
        executor_like_u8 && !args.src_signed ? build_wei_comp_1x1(wei, OC, IC) : std::vector<int32_t>{};
    const auto bias_comp =
        executor_like_u8 && !args.src_signed ? build_bias_comp_1x1(wei, OC, IC) : std::vector<int32_t>{};

    const size_t oc_blocks = OC / 8;
    const size_t ic_padded = round_up(IC, 8);
    const size_t stride8 = packed_block_stride_mmla(IC, 8);
    std::vector<int8_t> wei_packed(oc_blocks * stride8);
    for (size_t ocb = 0; ocb < oc_blocks; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 8 * IC;
        int8_t* dst_block = wei_packed.data() + ocb * stride8;
        pack_mmla_block(src_block, IC, ic_padded, 8, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x8_smmla_packed block8x8_smmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x8_usmmla_packed block8x8_usmmla;
    block8x8_smmla.create_ker();
    block8x8_usmmla.create_ker();
    const auto block8x8_smmla_ker = block8x8_smmla.ker();
    const auto block8x8_usmmla_ker = block8x8_usmmla.ker();
    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
    dot.create_ker();
    const auto dot_ker = dot.ker();

    std::vector<uint8_t> packed_src_u8;
    std::vector<int8_t> packed_src_s8;
    const size_t packed_pair_stride = IC * 2;
    if (args.src_signed) {
        packed_src_s8.resize(packed_pair_stride * 4);
    } else {
        packed_src_u8.resize(packed_pair_stride * 4);
    }

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                size_t w = 0;
                for (; w + 8 <= W; w += 8) {
                    const uint8_t* src_base = src.data() + ((n * H + h) * W + w) * IC;
                    if (args.src_signed) {
                        int8_t* pair_ptrs[4] = {packed_src_s8.data(),
                                                packed_src_s8.data() + packed_pair_stride,
                                                packed_src_s8.data() + 2 * packed_pair_stride,
                                                packed_src_s8.data() + 3 * packed_pair_stride};
                        for (size_t p = 0; p < 4; ++p) {
                            const int8_t* row0 = reinterpret_cast<const int8_t*>(src_base + (2 * p) * IC);
                            const int8_t* row1 = reinterpret_cast<const int8_t*>(src_base + (2 * p + 1) * IC);
                            int8_t* dst_pair = pair_ptrs[p];
                            for (size_t icb = 0; icb < IC; icb += 8) {
                                std::memcpy(dst_pair, row0 + icb, 8);
                                std::memcpy(dst_pair + 8, row1 + icb, 8);
                                dst_pair += 16;
                            }
                        }
                        const int8_t* src_ptrs_s8[4] = {pair_ptrs[0], pair_ptrs[1], pair_ptrs[2], pair_ptrs[3]};
                        for (size_t oc = 0; oc + 8 <= OC; oc += 8) {
                            const int8_t* wei_ptr = wei_packed.data() + (oc / 8) * stride8;
                            int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC + oc;
                            block8x8_smmla_ker(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), 0);
                        }
                    } else {
                        uint8_t* pair_ptrs[4] = {packed_src_u8.data(),
                                                 packed_src_u8.data() + packed_pair_stride,
                                                 packed_src_u8.data() + 2 * packed_pair_stride,
                                                 packed_src_u8.data() + 3 * packed_pair_stride};
                        for (size_t p = 0; p < 4; ++p) {
                            const uint8_t* row0 = src_base + (2 * p) * IC;
                            const uint8_t* row1 = src_base + (2 * p + 1) * IC;
                            uint8_t* dst_pair = pair_ptrs[p];
                            for (size_t icb = 0; icb < IC; icb += 8) {
                                std::memcpy(dst_pair, row0 + icb, 8);
                                std::memcpy(dst_pair + 8, row1 + icb, 8);
                                dst_pair += 16;
                            }
                        }
                        const uint8_t* src_ptrs_u8[4] = {pair_ptrs[0], pair_ptrs[1], pair_ptrs[2], pair_ptrs[3]};
                        for (size_t oc = 0; oc + 8 <= OC; oc += 8) {
                            const int8_t* wei_ptr = wei_packed.data() + (oc / 8) * stride8;
                            int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC + oc;
                            const int32_t* bias_block = executor_like_u8 ? bias_comp.data() + oc : nullptr;
                            block8x8_usmmla_ker(
                                src_ptrs_u8, wei_ptr, dst_ptr, IC, bias_block, OC * sizeof(int32_t), 0);
                            if (executor_like_u8 && !bias_block) {
                                add_1x1_comp_block(dst_ptr, 8, OC, wei_comp.data() + oc, 8);
                            }
                        }
                    }
                }
                for (; w < W; ++w) {
                    const uint8_t* src_ptr = src.data() + ((n * H + h) * W + w) * IC;
                    int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC;
                    for (size_t oc = 0; oc < OC; ++oc) {
                        dot_ker(src_ptr, wei.data() + oc * IC, dst_ptr + oc, IC, 0);
                        if (executor_like_u8 && !args.src_signed) {
                            dst_ptr[oc] += wei_comp[oc];
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

TimerResult bench_our_conv_1x1_block8x12_mmla_packed(const Args& args, bool executor_like_u8 = false) {
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
    const auto wei_comp =
        executor_like_u8 && !args.src_signed ? build_wei_comp_1x1(wei, OC, IC) : std::vector<int32_t>{};
    const auto bias_comp =
        executor_like_u8 && !args.src_signed ? build_bias_comp_1x1(wei, OC, IC) : std::vector<int32_t>{};

    const size_t oc_blocks = OC / 12;
    const size_t ic_padded = round_up(IC, 8);
    const size_t stride12 = packed_block_stride_mmla(IC, 12);
    std::vector<int8_t> wei_packed(oc_blocks * stride12);
    for (size_t ocb = 0; ocb < oc_blocks; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 12 * IC;
        int8_t* dst_block = wei_packed.data() + ocb * stride12;
        pack_mmla_block(src_block, IC, ic_padded, 12, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x12_smmla_packed block8x12_smmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_8x12_usmmla_packed block8x12_usmmla;
    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
    block8x12_smmla.create_ker();
    block8x12_usmmla.create_ker();
    dot.create_ker();
    const auto block8x12_smmla_ker = block8x12_smmla.ker();
    const auto block8x12_usmmla_ker = block8x12_usmmla.ker();
    const auto dot_ker = dot.ker();

    std::vector<uint8_t> packed_src_u8;
    std::vector<int8_t> packed_src_s8;
    const size_t packed_pair_stride = IC * 2;
    if (args.src_signed) {
        packed_src_s8.resize(packed_pair_stride * 4);
    } else {
        packed_src_u8.resize(packed_pair_stride * 4);
    }

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                size_t w = 0;
                for (; w + 8 <= W; w += 8) {
                    const uint8_t* src_base = src.data() + ((n * H + h) * W + w) * IC;
                    if (args.src_signed) {
                        int8_t* pair_ptrs[4] = {packed_src_s8.data(),
                                                packed_src_s8.data() + packed_pair_stride,
                                                packed_src_s8.data() + 2 * packed_pair_stride,
                                                packed_src_s8.data() + 3 * packed_pair_stride};
                        for (size_t p = 0; p < 4; ++p) {
                            const int8_t* row0 = reinterpret_cast<const int8_t*>(src_base + (2 * p) * IC);
                            const int8_t* row1 = reinterpret_cast<const int8_t*>(src_base + (2 * p + 1) * IC);
                            int8_t* dst_pair = pair_ptrs[p];
                            for (size_t icb = 0; icb < IC; icb += 8) {
                                std::memcpy(dst_pair, row0 + icb, 8);
                                std::memcpy(dst_pair + 8, row1 + icb, 8);
                                dst_pair += 16;
                            }
                        }
                        const int8_t* src_ptrs_s8[4] = {pair_ptrs[0], pair_ptrs[1], pair_ptrs[2], pair_ptrs[3]};
                        for (size_t oc = 0; oc + 12 <= OC; oc += 12) {
                            const int8_t* wei_ptr = wei_packed.data() + (oc / 12) * stride12;
                            int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC + oc;
                            block8x12_smmla_ker(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, OC * sizeof(int32_t), 0);
                        }
                    } else {
                        uint8_t* pair_ptrs[4] = {packed_src_u8.data(),
                                                 packed_src_u8.data() + packed_pair_stride,
                                                 packed_src_u8.data() + 2 * packed_pair_stride,
                                                 packed_src_u8.data() + 3 * packed_pair_stride};
                        for (size_t p = 0; p < 4; ++p) {
                            const uint8_t* row0 = src_base + (2 * p) * IC;
                            const uint8_t* row1 = src_base + (2 * p + 1) * IC;
                            uint8_t* dst_pair = pair_ptrs[p];
                            for (size_t icb = 0; icb < IC; icb += 8) {
                                std::memcpy(dst_pair, row0 + icb, 8);
                                std::memcpy(dst_pair + 8, row1 + icb, 8);
                                dst_pair += 16;
                            }
                        }
                        const uint8_t* src_ptrs_u8[4] = {pair_ptrs[0], pair_ptrs[1], pair_ptrs[2], pair_ptrs[3]};
                        for (size_t oc = 0; oc + 12 <= OC; oc += 12) {
                            const int8_t* wei_ptr = wei_packed.data() + (oc / 12) * stride12;
                            int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC + oc;
                            const int32_t* bias_block = executor_like_u8 ? bias_comp.data() + oc : nullptr;
                            block8x12_usmmla_ker(
                                src_ptrs_u8, wei_ptr, dst_ptr, IC, bias_block, OC * sizeof(int32_t), 0);
                            if (executor_like_u8 && !bias_block) {
                                add_1x1_comp_block(dst_ptr, 8, OC, wei_comp.data() + oc, 12);
                            }
                        }
                    }
                }
                for (; w < W; ++w) {
                    const uint8_t* src_ptr = src.data() + ((n * H + h) * W + w) * IC;
                    int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC;
                    for (size_t oc = 0; oc < OC; ++oc) {
                        dot_ker(src_ptr, wei.data() + oc * IC, dst_ptr + oc, IC, 0);
                        if (executor_like_u8 && !args.src_signed) {
                            dst_ptr[oc] += wei_comp[oc];
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
    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
    dot.create_ker();
    const auto dot_ker = dot.ker();

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                size_t w = 0;
                for (; w + 4 <= W; w += 4) {
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
                for (; w < W; ++w) {
                    const uint8_t* src_ptr = src.data() + ((n * H + h) * W + w) * IC;
                    int32_t* dst_ptr = dst.data() + ((n * H + h) * W + w) * OC;
                    for (size_t oc = 0; oc < OC; ++oc) {
                        dot_ker(src_ptr, wei.data() + oc * IC, dst_ptr + oc, IC, 0);
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

TimerResult bench_our_conv_1x1_gemm_mmla(const Args& args) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t M = H * W;

    std::vector<uint8_t> src(N * M * IC);
    std::vector<int8_t> wei(OC * IC);
    std::vector<int32_t> dst(N * M * OC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    const size_t ic_padded = round_up(IC, 8);
    const size_t stride16 = packed_block_stride_mmla(IC, 16);
    const size_t stride8 = packed_block_stride_mmla(IC, 8);
    const size_t stride4 = packed_block_stride_mmla(IC, 4);
    const size_t oc_blocks16 = OC / 16;
    const size_t oc_blocks8 = OC / 8;
    const size_t oc_blocks4 = OC / 4;
    std::vector<int8_t> wei_packed16(oc_blocks16 * stride16);
    std::vector<int8_t> wei_packed8(oc_blocks8 * stride8);
    std::vector<int8_t> wei_packed4(oc_blocks4 * stride4);
    for (size_t ocb = 0; ocb < oc_blocks16; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 16 * IC;
        int8_t* dst_block = wei_packed16.data() + ocb * stride16;
        pack_mmla_block(src_block, IC, ic_padded, 16, dst_block);
    }
    for (size_t ocb = 0; ocb < oc_blocks8; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 8 * IC;
        int8_t* dst_block = wei_packed8.data() + ocb * stride8;
        pack_mmla_block(src_block, IC, ic_padded, 8, dst_block);
    }
    for (size_t ocb = 0; ocb < oc_blocks4; ++ocb) {
        const int8_t* src_block = wei.data() + ocb * 4 * IC;
        int8_t* dst_block = wei_packed4.data() + ocb * stride4;
        pack_mmla_block(src_block, IC, ic_padded, 4, dst_block);
    }

    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_smmla_packed block4x16_smmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_usmmla_packed block4x16_usmmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_smmla_packed block4x8_smmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_usmmla_packed block4x8_usmmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_smmla_packed block4x4_smmla;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_usmmla_packed block4x4_usmmla;
    block4x16_smmla.create_ker();
    block4x16_usmmla.create_ker();
    block4x8_smmla.create_ker();
    block4x8_usmmla.create_ker();
    block4x4_smmla.create_ker();
    block4x4_usmmla.create_ker();
    const auto block4x16_smmla_ker = block4x16_smmla.ker();
    const auto block4x16_usmmla_ker = block4x16_usmmla.ker();
    const auto block4x8_smmla_ker = block4x8_smmla.ker();
    const auto block4x8_usmmla_ker = block4x8_usmmla.ker();
    const auto block4x4_smmla_ker = block4x4_smmla.ker();
    const auto block4x4_usmmla_ker = block4x4_usmmla.ker();
    ov::intel_cpu::aarch64::jit_int8_dot_kernel dot(args.src_signed);
    dot.create_ker();
    const auto dot_ker = dot.ker();

    const size_t dst_stride = OC;
    const size_t dst_stride_bytes = OC * sizeof(int32_t);

    auto run = [&]() {
        for (size_t n = 0; n < N; ++n) {
            const uint8_t* src_base = src.data() + n * M * IC;
            int32_t* dst_base = dst.data() + n * M * OC;
            size_t m = 0;
            for (; m + 4 <= M; m += 4) {
                const uint8_t* src_ptr0 = src_base + (m + 0) * IC;
                const uint8_t* src_ptr1 = src_base + (m + 1) * IC;
                const uint8_t* src_ptr2 = src_base + (m + 2) * IC;
                const uint8_t* src_ptr3 = src_base + (m + 3) * IC;
                const int8_t* src_ptrs_s8[4] = {
                    reinterpret_cast<const int8_t*>(src_ptr0),
                    reinterpret_cast<const int8_t*>(src_ptr1),
                    reinterpret_cast<const int8_t*>(src_ptr2),
                    reinterpret_cast<const int8_t*>(src_ptr3),
                };
                const uint8_t* src_ptrs_u8[4] = {src_ptr0, src_ptr1, src_ptr2, src_ptr3};
                size_t oc = 0;
                for (; oc + 16 <= OC; oc += 16) {
                    const int8_t* wei_ptr = wei_packed16.data() + (oc / 16) * stride16;
                    int32_t* dst_ptr = dst_base + m * dst_stride + oc;
                    if (args.src_signed) {
                        block4x16_smmla_ker(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, dst_stride_bytes, 0);
                    } else {
                        block4x16_usmmla_ker(src_ptrs_u8, wei_ptr, dst_ptr, IC, 0, dst_stride_bytes, 0);
                    }
                }
                for (; oc + 8 <= OC; oc += 8) {
                    const int8_t* wei_ptr = wei_packed8.data() + (oc / 8) * stride8;
                    int32_t* dst_ptr = dst_base + m * dst_stride + oc;
                    if (args.src_signed) {
                        block4x8_smmla_ker(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, dst_stride_bytes, 0);
                    } else {
                        block4x8_usmmla_ker(src_ptrs_u8, wei_ptr, dst_ptr, IC, 0, dst_stride_bytes, 0);
                    }
                }
                for (; oc + 4 <= OC; oc += 4) {
                    const int8_t* wei_ptr = wei_packed4.data() + (oc / 4) * stride4;
                    int32_t* dst_ptr = dst_base + m * dst_stride + oc;
                    if (args.src_signed) {
                        block4x4_smmla_ker(src_ptrs_s8, wei_ptr, dst_ptr, IC, 0, dst_stride_bytes, 0);
                    } else {
                        block4x4_usmmla_ker(src_ptrs_u8, wei_ptr, dst_ptr, IC, 0, dst_stride_bytes, 0);
                    }
                }
                for (; oc < OC; ++oc) {
                    const int8_t* wei_ptr = wei.data() + oc * IC;
                    int32_t* dst_ptr = dst_base + m * dst_stride + oc;
                    dot_ker(src_ptr0, wei_ptr, dst_ptr, IC, 0);
                    dot_ker(src_ptr1, wei_ptr, dst_ptr + dst_stride, IC, 0);
                    dot_ker(src_ptr2, wei_ptr, dst_ptr + 2 * dst_stride, IC, 0);
                    dot_ker(src_ptr3, wei_ptr, dst_ptr + 3 * dst_stride, IC, 0);
                }
            }
            for (; m < M; ++m) {
                const uint8_t* src_ptr = src_base + m * IC;
                int32_t* dst_ptr = dst_base + m * dst_stride;
                for (size_t oc = 0; oc < OC; ++oc) {
                    dot_ker(src_ptr, wei.data() + oc * IC, dst_ptr + oc, IC, 0);
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

TimerResult bench_kleidiai_gemm_total(const Args& args, bool& supported) {
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

    const size_t lhs_packed_offset = ukernel->get_lhs_packed_offset(0, K);
    auto run = [&]() {
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

TimerResult bench_kleidiai_matmul(const Args& args,
                                  size_t M,
                                  size_t N,
                                  size_t K,
                                  const float* A,
                                  const int8_t* B,
                                  bool& supported) {
    supported = false;
    if (K % 8 != 0 || (!has_i8mm() && !has_asimd_dotprod())) {
        return {};
    }

    const auto* ukernel = has_i8mm() ? &kKleidiI8I8mm : &kKleidiI8Dotprod;
    supported = true;

    std::vector<float> C(M * N);
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
                                             B,
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
                                           A + m_blk * m_step * K,
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

TimerResult bench_kleidiai_conv_1x1(const Args& args, bool& supported) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t OH = conv_out_dim(H, args.KH, args.stride, args.pad, args.dilation);
    const size_t OW = conv_out_dim(W, args.KW, args.stride, args.pad, args.dilation);
    if (OH == 0 || OW == 0) {
        supported = false;
        return {-1.0, -1.0};
    }

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    const size_t M = N * OH * OW;
    const size_t K = IC;
    std::vector<float> A(M * K);
    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < OH; ++h) {
            for (size_t w = 0; w < OW; ++w) {
                const size_t row = (n * OH + h) * OW + w;
                float* A_row = A.data() + row * K;
                const ptrdiff_t ih = static_cast<ptrdiff_t>(h * args.stride) - static_cast<ptrdiff_t>(args.pad);
                const ptrdiff_t iw = static_cast<ptrdiff_t>(w * args.stride) - static_cast<ptrdiff_t>(args.pad);
                if (ih < 0 || iw < 0 || ih >= static_cast<ptrdiff_t>(H) || iw >= static_cast<ptrdiff_t>(W)) {
                    std::fill_n(A_row, IC, 0.0f);
                    continue;
                }
                const size_t src_base = ((n * H + static_cast<size_t>(ih)) * W + static_cast<size_t>(iw)) * IC;
                for (size_t ic = 0; ic < IC; ++ic) {
                    if (args.src_signed) {
                        A_row[ic] = static_cast<float>(static_cast<int8_t>(src[src_base + ic]));
                    } else {
                        A_row[ic] = static_cast<float>(src[src_base + ic]);
                    }
                }
            }
        }
    }

    std::vector<int8_t> B(OC * K);
    for (size_t oc = 0; oc < OC; ++oc) {
        std::copy_n(wei.data() + oc * IC, IC, B.data() + oc * K);
    }

    return bench_kleidiai_matmul(args, M, OC, K, A.data(), B.data(), supported);
}

TimerResult bench_kleidiai_conv_1x1_total(const Args& args, bool& supported) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t OH = conv_out_dim(H, args.KH, args.stride, args.pad, args.dilation);
    const size_t OW = conv_out_dim(W, args.KW, args.stride, args.pad, args.dilation);
    if (OH == 0 || OW == 0) {
        supported = false;
        return {-1.0, -1.0};
    }
    const size_t M = N * OH * OW;
    const size_t K = IC;
    supported = false;
    if (K % 8 != 0 || (!has_i8mm() && !has_asimd_dotprod())) {
        return {};
    }

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    const auto* ukernel = has_i8mm() ? &kKleidiI8I8mm : &kKleidiI8Dotprod;
    supported = true;

    std::vector<float> A(M * K);
    std::vector<int8_t> B(OC * K);
    for (size_t oc = 0; oc < OC; ++oc) {
        std::copy_n(wei.data() + oc * IC, IC, B.data() + oc * K);
    }

    std::vector<float> C(M * OC);
    const size_t mr = ukernel->get_mr();
    const size_t nr = ukernel->get_nr();
    const size_t kr = ukernel->get_kr();
    const size_t sr = ukernel->get_sr();
    const size_t m_step = 16;
    const size_t n_step = ukernel->get_n_step();
    const size_t lhs_stride = K * sizeof(float);
    const size_t dst_stride_row = OC * sizeof(float);
    const size_t dst_stride_col = sizeof(float);

    const size_t lhs_block_bytes = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m_step, K, mr, kr, sr);
    const size_t m_blocks = (M + m_step - 1) / m_step;
    std::vector<uint8_t> lhs_packed(lhs_block_bytes * m_blocks);

    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(OC, K, nr, kr, sr);
    std::vector<int8_t> rhs_packed(rhs_packed_size);
    std::vector<float> rhs_scales(OC, 1.0f);
    std::vector<float> rhs_bias(OC, 0.0f);
    kai_rhs_pack_qsi8cx_params params{};
    params.lhs_zero_point = 1;
    params.scale_multiplier = 1.0f;

    kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(1,
                                             OC,
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

    const size_t lhs_packed_offset = ukernel->get_lhs_packed_offset(0, K);
    auto materialize_lhs = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < OH; ++h) {
                for (size_t w = 0; w < OW; ++w) {
                    const size_t row = (n * OH + h) * OW + w;
                    float* A_row = A.data() + row * K;
                    const ptrdiff_t ih = static_cast<ptrdiff_t>(h * args.stride) - static_cast<ptrdiff_t>(args.pad);
                    const ptrdiff_t iw = static_cast<ptrdiff_t>(w * args.stride) - static_cast<ptrdiff_t>(args.pad);
                    if (ih < 0 || iw < 0 || ih >= static_cast<ptrdiff_t>(H) || iw >= static_cast<ptrdiff_t>(W)) {
                        std::fill_n(A_row, IC, 0.0f);
                        continue;
                    }
                    const size_t src_base = ((n * H + static_cast<size_t>(ih)) * W + static_cast<size_t>(iw)) * IC;
                    for (size_t ic = 0; ic < IC; ++ic) {
                        A_row[ic] = args.src_signed ? static_cast<float>(static_cast<int8_t>(src[src_base + ic]))
                                                    : static_cast<float>(src[src_base + ic]);
                    }
                }
            }
        }
    };

    auto run = [&]() {
        materialize_lhs();
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
        for (size_t m_blk = 0; m_blk < m_blocks; ++m_blk) {
            const size_t M_iter = std::min(M - m_blk * m_step, m_step);
            const auto* lhs_packed_block = lhs_packed.data() + m_blk * lhs_block_bytes;
            for (size_t n_idx = 0; n_idx < OC; n_idx += n_step) {
                const size_t N_iter = std::min(OC - n_idx, n_step);
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

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(OC) * static_cast<double>(K);
    return time_loop(args.iters, ops, run);
}

TimerResult bench_kleidiai_conv_kxk(const Args& args, bool& supported) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t KH = args.KH;
    const size_t KW = args.KW;
    const size_t OH = conv_out_dim(H, KH, args.stride, args.pad, args.dilation);
    const size_t OW = conv_out_dim(W, KW, args.stride, args.pad, args.dilation);
    if (OH == 0 || OW == 0) {
        supported = false;
        return {-1.0, -1.0};
    }

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC * KH * KW);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    const size_t M = N * OH * OW;
    const size_t K = IC * KH * KW;
    std::vector<float> A(M * K);
    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < OH; ++h) {
            for (size_t w = 0; w < OW; ++w) {
                const size_t row = (n * OH + h) * OW + w;
                float* A_row = A.data() + row * K;
                for (size_t kh = 0; kh < KH; ++kh) {
                    for (size_t kw = 0; kw < KW; ++kw) {
                        const size_t col_base = (kh * KW + kw) * IC;
                        const ptrdiff_t ih = static_cast<ptrdiff_t>(h * args.stride) -
                                             static_cast<ptrdiff_t>(args.pad) +
                                             static_cast<ptrdiff_t>(kh * args.dilation);
                        const ptrdiff_t iw = static_cast<ptrdiff_t>(w * args.stride) -
                                             static_cast<ptrdiff_t>(args.pad) +
                                             static_cast<ptrdiff_t>(kw * args.dilation);
                        if (ih < 0 || iw < 0 || ih >= static_cast<ptrdiff_t>(H) || iw >= static_cast<ptrdiff_t>(W)) {
                            std::fill_n(A_row + col_base, IC, 0.0f);
                            continue;
                        }
                        const size_t src_base =
                            ((n * H + static_cast<size_t>(ih)) * W + static_cast<size_t>(iw)) * IC;
                        for (size_t ic = 0; ic < IC; ++ic) {
                            if (args.src_signed) {
                                A_row[col_base + ic] = static_cast<float>(static_cast<int8_t>(src[src_base + ic]));
                            } else {
                                A_row[col_base + ic] = static_cast<float>(src[src_base + ic]);
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<int8_t> B(OC * K);
    for (size_t oc = 0; oc < OC; ++oc) {
        for (size_t kh = 0; kh < KH; ++kh) {
            for (size_t kw = 0; kw < KW; ++kw) {
                for (size_t ic = 0; ic < IC; ++ic) {
                    const size_t k_idx = (kh * KW + kw) * IC + ic;
                    B[oc * K + k_idx] = wei[((oc * IC + ic) * KH + kh) * KW + kw];
                }
            }
        }
    }

    return bench_kleidiai_matmul(args, M, OC, K, A.data(), B.data(), supported);
}

TimerResult bench_kleidiai_conv_kxk_total(const Args& args, bool& supported) {
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t KH = args.KH;
    const size_t KW = args.KW;
    const size_t OH = conv_out_dim(H, KH, args.stride, args.pad, args.dilation);
    const size_t OW = conv_out_dim(W, KW, args.stride, args.pad, args.dilation);
    if (OH == 0 || OW == 0) {
        supported = false;
        return {-1.0, -1.0};
    }
    const size_t M = N * OH * OW;
    const size_t K = IC * KH * KW;
    supported = false;
    if (K % 8 != 0 || (!has_i8mm() && !has_asimd_dotprod())) {
        return {};
    }

    std::vector<uint8_t> src(N * H * W * IC);
    std::vector<int8_t> wei(OC * IC * KH * KW);
    std::mt19937 gen(42);
    fill_random(src, args.src_signed ? -10 : 0, 10, gen);
    fill_random(wei, -10, 10, gen);

    const auto* ukernel = has_i8mm() ? &kKleidiI8I8mm : &kKleidiI8Dotprod;
    supported = true;

    std::vector<float> A(M * K);
    std::vector<int8_t> B(OC * K);
    for (size_t oc = 0; oc < OC; ++oc) {
        for (size_t kh = 0; kh < KH; ++kh) {
            for (size_t kw = 0; kw < KW; ++kw) {
                for (size_t ic = 0; ic < IC; ++ic) {
                    const size_t k_idx = (kh * KW + kw) * IC + ic;
                    B[oc * K + k_idx] = wei[((oc * IC + ic) * KH + kh) * KW + kw];
                }
            }
        }
    }

    std::vector<float> C(M * OC);
    const size_t mr = ukernel->get_mr();
    const size_t nr = ukernel->get_nr();
    const size_t kr = ukernel->get_kr();
    const size_t sr = ukernel->get_sr();
    const size_t m_step = 16;
    const size_t n_step = ukernel->get_n_step();
    const size_t lhs_stride = K * sizeof(float);
    const size_t dst_stride_row = OC * sizeof(float);
    const size_t dst_stride_col = sizeof(float);

    const size_t lhs_block_bytes = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m_step, K, mr, kr, sr);
    const size_t m_blocks = (M + m_step - 1) / m_step;
    std::vector<uint8_t> lhs_packed(lhs_block_bytes * m_blocks);

    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(OC, K, nr, kr, sr);
    std::vector<int8_t> rhs_packed(rhs_packed_size);
    std::vector<float> rhs_scales(OC, 1.0f);
    std::vector<float> rhs_bias(OC, 0.0f);
    kai_rhs_pack_qsi8cx_params params{};
    params.lhs_zero_point = 1;
    params.scale_multiplier = 1.0f;

    kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(1,
                                             OC,
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

    const size_t lhs_packed_offset = ukernel->get_lhs_packed_offset(0, K);
    auto materialize_lhs = [&]() {
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < OH; ++h) {
                for (size_t w = 0; w < OW; ++w) {
                    const size_t row = (n * OH + h) * OW + w;
                    float* A_row = A.data() + row * K;
                    for (size_t kh = 0; kh < KH; ++kh) {
                        for (size_t kw = 0; kw < KW; ++kw) {
                            const size_t col_base = (kh * KW + kw) * IC;
                            const ptrdiff_t ih = static_cast<ptrdiff_t>(h * args.stride) -
                                                 static_cast<ptrdiff_t>(args.pad) +
                                                 static_cast<ptrdiff_t>(kh * args.dilation);
                            const ptrdiff_t iw = static_cast<ptrdiff_t>(w * args.stride) -
                                                 static_cast<ptrdiff_t>(args.pad) +
                                                 static_cast<ptrdiff_t>(kw * args.dilation);
                            if (ih < 0 || iw < 0 || ih >= static_cast<ptrdiff_t>(H) || iw >= static_cast<ptrdiff_t>(W)) {
                                std::fill_n(A_row + col_base, IC, 0.0f);
                                continue;
                            }
                            const size_t src_base =
                                ((n * H + static_cast<size_t>(ih)) * W + static_cast<size_t>(iw)) * IC;
                            for (size_t ic = 0; ic < IC; ++ic) {
                                A_row[col_base + ic] = args.src_signed
                                                           ? static_cast<float>(static_cast<int8_t>(src[src_base + ic]))
                                                           : static_cast<float>(src[src_base + ic]);
                            }
                        }
                    }
                }
            }
        }
    };

    auto run = [&]() {
        materialize_lhs();
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
        for (size_t m_blk = 0; m_blk < m_blocks; ++m_blk) {
            const size_t M_iter = std::min(M - m_blk * m_step, m_step);
            const auto* lhs_packed_block = lhs_packed.data() + m_blk * lhs_block_bytes;
            for (size_t n_idx = 0; n_idx < OC; n_idx += n_step) {
                const size_t N_iter = std::min(OC - n_idx, n_step);
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

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(OC) * static_cast<double>(K);
    return time_loop(args.iters, ops, run);
}
#endif

#if defined(OV_CPU_WITH_ACL)
TimerResult bench_acl_gemm(const Args& args) {
    using namespace arm_compute;
    Scheduler::get().set_num_threads(1);

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
    Scheduler::get().set_num_threads(1);
    const size_t N = args.Nn;
    const size_t H = args.H;
    const size_t W = args.W;
    const size_t IC = args.IC;
    const size_t OC = args.OC;
    const size_t KH = args.KH;
    const size_t KW = args.KW;
    const size_t OH = conv_out_dim(H, KH, args.stride, args.pad, args.dilation);
    const size_t OW = conv_out_dim(W, KW, args.stride, args.pad, args.dilation);

    const DataType src_dt = args.src_signed ? DataType::QASYMM8_SIGNED : DataType::QASYMM8;
    const QuantizationInfo qinfo(1.0F, 0);

    Tensor src;
    Tensor wei;
    Tensor dst;
    const TensorInfo src_info(TensorShape(IC, W, H, N), 1, src_dt, DataLayout::NHWC);
    const TensorInfo wei_info(TensorShape(IC, args.KW, args.KH, OC), 1, DataType::QASYMM8_SIGNED, DataLayout::NHWC);
    TensorInfo dst_info;
    const PadStrideInfo ps_info(args.stride, args.stride, args.pad, args.pad);
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
    conv.configure(&src,
                   &wei,
                   nullptr,
                   &dst,
                   ps_info,
                   WeightsInfo(),
                   Size2D(static_cast<unsigned int>(args.dilation), static_cast<unsigned int>(args.dilation)));

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

    const double ops = 2.0 * static_cast<double>(N) * static_cast<double>(OH) * static_cast<double>(OW) *
                       static_cast<double>(OC) * static_cast<double>(IC) * static_cast<double>(KH) *
                       static_cast<double>(KW);
    return time_loop(args.iters, ops, [&]() { conv.run(); });
}
#endif

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const auto runtime_isa = ov::intel_cpu::getAarch64Int8Isa();
    std::ostringstream out;
    const auto flush_report = [&]() {
        std::cout << rewrite_benchmark_report(out.str(), args.src_signed);
    };
    out << "INT8 microbench (AArch64)\n";
    out << "runtime_isa=" << ov::intel_cpu::aarch64Int8IsaName(runtime_isa)
              << " dotprod=" << has_asimd_dotprod() << " i8mm=" << has_i8mm() << " sve=" << has_sve() << "\n";
    out << "src=" << (args.src_signed ? "s8" : "u8") << " weights=s8\n";

    if (args.run_gemm) {
        out << "\nGEMM M=" << args.M << " N=" << args.N << " K=" << args.K << "\n";
        if (args.use_our) {
            const auto dot = bench_our_gemm(args, false, false);
            const auto block4 = bench_our_gemm(args, true, false);
            const auto brgemm = bench_our_brgemm_gemm(args);
            out << "  our_dot     : " << dot.ms << " ms, " << dot.gops << " GOPS\n";
            out << "  our_block4  : " << block4.ms << " ms, " << block4.gops << " GOPS\n";
            if (brgemm.ms >= 0.0) {
                out << "  our_brgemm4x4 : " << brgemm.ms << " ms, " << brgemm.gops << " GOPS\n";
            } else {
                out << "  our_brgemm4x4 : unsupported (brgemm unavailable)\n";
            }
            if (has_asimd_dotprod()) {
                const auto block4_dot = bench_our_gemm(args, true, true);
                out << "  our_block4_dot : " << block4_dot.ms << " ms, " << block4_dot.gops << " GOPS\n";
                if (args.N % 8 == 0) {
                    const auto block8_dot = bench_our_gemm_block8_dot(args);
                    out << "  our_block8_dot : " << block8_dot.ms << " ms, " << block8_dot.gops << " GOPS\n";
                    const auto block8_dot_packed = bench_our_gemm_block8_dot_packed(args);
                    out << "  our_block8_dot_packed : " << block8_dot_packed.ms << " ms, "
                              << block8_dot_packed.gops << " GOPS\n";
                } else {
                    out << "  our_block8_dot : unsupported (N multiple of 8)\n";
                    out << "  our_block8_dot_packed : unsupported (N multiple of 8)\n";
                }
                if (args.M % 4 == 0 && args.N % 4 == 0) {
                    const auto block4x4_dot = bench_our_gemm_block4x4_dot(args);
                    out << "  our_block4x4_dot : " << block4x4_dot.ms << " ms, " << block4x4_dot.gops
                              << " GOPS\n";
                    const auto block4x4_dot_packed = bench_our_gemm_block4x4_dot_packed(args);
                    out << "  our_block4x4_dot_packed : " << block4x4_dot_packed.ms << " ms, "
                              << block4x4_dot_packed.gops << " GOPS\n";
                } else {
                    out << "  our_block4x4_dot : unsupported (M,N multiple of 4)\n";
                    out << "  our_block4x4_dot_packed : unsupported (M,N multiple of 4)\n";
                }
            }
            if (has_i8mm()) {
                if (args.M % 4 == 0 && args.N % 4 == 0 && args.K % 8 == 0) {
                    const auto block4x4_mmla = bench_our_gemm_block4x4_mmla_packed(args);
                    out << "  our_block4x4_mmla_packed : " << block4x4_mmla.ms << " ms, "
                              << block4x4_mmla.gops << " GOPS\n";
                } else {
                    out << "  our_block4x4_mmla_packed : unsupported (M,N multiple of 4, K multiple of 8)\n";
                }
                if (args.M % 4 == 0 && args.N % 8 == 0 && args.K % 8 == 0) {
                    const auto block4x8_mmla = bench_our_gemm_block4x8_mmla_packed(args);
                    out << "  our_block4x8_mmla_packed : " << block4x8_mmla.ms << " ms, "
                              << block4x8_mmla.gops << " GOPS\n";
                } else {
                    out << "  our_block4x8_mmla_packed : unsupported (M multiple of 4, N multiple of 8, K multiple of 8)\n";
                }
                if (args.M % 4 == 0 && args.N % 16 == 0 && args.K % 8 == 0) {
                    const auto block4x16_mmla = bench_our_gemm_block4x16_mmla_packed(args);
                    out << "  our_block4x16_mmla_packed : " << block4x16_mmla.ms << " ms, "
                              << block4x16_mmla.gops << " GOPS\n";
                } else {
                    out << "  our_block4x16_mmla_packed : unsupported (M multiple of 4, N multiple of 16, K multiple of 8)\n";
                }
                if (args.M % 8 == 0 && args.N % 8 == 0 && args.K % 8 == 0) {
                    const auto block8x8_mmla = bench_our_gemm_block8x8_mmla_packed(args);
                    out << "  our_block8x8_mmla_packed : " << block8x8_mmla.ms << " ms, "
                              << block8x8_mmla.gops << " GOPS\n";
                } else {
                    out << "  our_block8x8_mmla_packed : unsupported (M multiple of 8, N multiple of 8, K multiple of 8)\n";
                }
                if (args.M % 8 == 0 && args.N % 12 == 0 && args.K % 8 == 0) {
                    const auto block8x12_mmla = bench_our_gemm_block8x12_mmla_packed(args);
                    out << "  our_block8x12_mmla_packed : " << block8x12_mmla.ms << " ms, "
                              << block8x12_mmla.gops << " GOPS\n";
                } else {
                    out << "  our_block8x12_mmla_packed : unsupported (M multiple of 8, N multiple of 12, K multiple of 8)\n";
                }
                if (args.K % 8 == 0) {
                    const auto tuned = bench_our_gemm_mmla_tuned(args);
                    out << "  our_brgemm_mmla_tuned : " << tuned.ms << " ms, " << tuned.gops << " GOPS\n";
                } else {
                    out << "  our_brgemm_mmla_tuned : unsupported (K multiple of 8)\n";
                }
            }
        }
#if defined(OV_CPU_WITH_ACL)
        if (args.use_acl) {
            const auto acl = bench_acl_gemm(args);
            out << "  acl_gemm    : " << acl.ms << " ms, " << acl.gops << " GOPS\n";
        }
#else
        if (args.use_acl) {
            out << "  acl_gemm    : unavailable (built without ACL)\n";
        }
#endif
#if defined(OV_CPU_WITH_KLEIDIAI)
        if (args.use_kleidiai) {
            bool supported = false;
            const auto kleidiai_total = bench_kleidiai_gemm_total(args, supported);
            if (!supported) {
                out << "  kleidiai_gemm : unsupported (requires dotprod/i8mm, K multiple of 8)\n";
            } else {
                out << "  kleidiai_gemm_total : " << kleidiai_total.ms << " ms, " << kleidiai_total.gops
                          << " GOPS\n";
                const auto kleidiai = bench_kleidiai_gemm(args, supported);
                out << "  kleidiai_gemm : " << kleidiai.ms << " ms, " << kleidiai.gops << " GOPS\n";
            }
        }
#else
        if (args.use_kleidiai) {
            out << "  kleidiai_gemm : unavailable (built without KleidiAI)\n";
        }
#endif
    }

    if (args.run_conv) {
        const size_t OH = conv_out_dim(args.H, args.KH, args.stride, args.pad, args.dilation);
        const size_t OW = conv_out_dim(args.W, args.KW, args.stride, args.pad, args.dilation);
        out << "\nCONV N=" << args.Nn << " H=" << args.H << " W=" << args.W << " IC=" << args.IC
                  << " OC=" << args.OC << " KH=" << args.KH << " KW=" << args.KW << " stride=" << args.stride
                  << " pad=" << args.pad << " dilation=" << args.dilation << " groups=" << args.groups
                  << "\n";
        const bool our_groups_ok = (args.groups == 1);
        const bool our_default_geom = (args.stride == 1) && (args.pad == 0) && (args.dilation == 1);
        if (!our_groups_ok) {
            out << "  our_conv    : unsupported (groups!=1)\n";
        }
        if (args.KH == 1 && args.KW == 1) {
            if (args.use_our) {
                if (!our_groups_ok || !our_default_geom) {
                    out << "  our_conv1x1 : unsupported (requires stride=1, pad=0, dilation=1, groups=1)\n";
                } else {
                const auto dot = bench_our_conv_1x1(args, false, false);
                const auto block4 = bench_our_conv_1x1(args, true, false);
                const auto brgemm = bench_our_conv_1x1_brgemm(args);
                out << "  our_conv1x1 : " << dot.ms << " ms, " << dot.gops << " GOPS\n";
                out << "  our_conv1x1_block4 : " << block4.ms << " ms, " << block4.gops << " GOPS\n";
                if (brgemm.ms >= 0.0) {
                    out << "  our_conv1x1_brgemm4x4 : " << brgemm.ms << " ms, " << brgemm.gops
                              << " GOPS\n";
                } else {
                    out << "  our_conv1x1_brgemm4x4 : unsupported (brgemm unavailable)\n";
                }
                if (args.W >= 8) {
                    const auto brgemm8x8 = bench_our_conv_1x1_brgemm_mnb(args, 8, 8);
                    if (brgemm8x8.ms >= 0.0) {
                        out << "  our_conv1x1_brgemm8x8 : " << brgemm8x8.ms << " ms, " << brgemm8x8.gops
                                  << " GOPS\n";
                    } else {
                        out << "  our_conv1x1_brgemm8x8 : unsupported (brgemm unavailable)\n";
                    }
                    const auto brgemm8x16 = bench_our_conv_1x1_brgemm_mnb(args, 8, 16);
                    if (brgemm8x16.ms >= 0.0) {
                        out << "  our_conv1x1_brgemm8x16 : " << brgemm8x16.ms << " ms, " << brgemm8x16.gops
                                  << " GOPS\n";
                    } else {
                        out << "  our_conv1x1_brgemm8x16 : unsupported (brgemm unavailable)\n";
                    }
                    const auto brgemm8xoc = bench_our_conv_1x1_brgemm_mnb(args, 8, args.OC);
                    if (brgemm8xoc.ms >= 0.0) {
                        out << "  our_conv1x1_brgemm8xoc : " << brgemm8xoc.ms << " ms, " << brgemm8xoc.gops
                                  << " GOPS\n";
                    } else {
                        out << "  our_conv1x1_brgemm8xoc : unsupported (brgemm unavailable)\n";
                    }
                } else {
                    out << "  our_conv1x1_brgemm8x8 : unsupported (W>=8)\n";
                    out << "  our_conv1x1_brgemm8x16 : unsupported (W>=8)\n";
                    out << "  our_conv1x1_brgemm8xoc : unsupported (W>=8)\n";
                }
                if (has_asimd_dotprod()) {
                    const auto block4_dot = bench_our_conv_1x1(args, true, true);
                    out << "  our_conv1x1_block4_dot : " << block4_dot.ms << " ms, " << block4_dot.gops
                              << " GOPS\n";
                    if (args.OC % 8 == 0) {
                        const auto block8_dot = bench_our_conv_1x1_block8_dot(args);
                        out << "  our_conv1x1_block8_dot : " << block8_dot.ms << " ms, " << block8_dot.gops
                                  << " GOPS\n";
                        const auto block8_dot_packed = bench_our_conv_1x1_block8_dot_packed(args);
                        out << "  our_conv1x1_block8_dot_packed : " << block8_dot_packed.ms << " ms, "
                                  << block8_dot_packed.gops << " GOPS\n";
                        const auto block2x8_dot_packed = bench_our_conv_1x1_block2x8_dot_packed(args);
                        out << "  our_conv1x1_block2x8_dot_packed : " << block2x8_dot_packed.ms << " ms, "
                                  << block2x8_dot_packed.gops << " GOPS\n";
                    if (args.IC % 4 == 0) {
                        const auto block2x8_dot_packed_i4 = bench_our_conv_1x1_block2x8_dot_packed_interleaved4(args);
                        out << "  our_conv1x1_block2x8_dot_packed_i4 : " << block2x8_dot_packed_i4.ms << " ms, "
                                  << block2x8_dot_packed_i4.gops << " GOPS\n";
                        if (args.OC % 16 == 0) {
                            if (args.OC % 32 == 0) {
                                const auto block2x32_dot_packed_i4 =
                                    bench_our_conv_1x1_block2x32_dot_packed_interleaved4(args);
                                out << "  our_conv1x1_block2x32_dot_packed_i4 : "
                                          << block2x32_dot_packed_i4.ms << " ms, "
                                          << block2x32_dot_packed_i4.gops << " GOPS\n";
                                if (!args.src_signed) {
                                    const auto block2x32_dot_packed_i4_exec =
                                        bench_our_conv_1x1_block2x32_dot_packed_interleaved4(args, true);
                                    out << "  our_conv1x1_block2x32_dot_packed_i4_exec_u8 : "
                                              << block2x32_dot_packed_i4_exec.ms << " ms, "
                                              << block2x32_dot_packed_i4_exec.gops << " GOPS\n";
                                }
                            } else {
                                out << "  our_conv1x1_block2x32_dot_packed_i4 : unsupported (OC multiple of 32)\n";
                            }
                            const auto block2x16_dot_packed_i4 =
                                bench_our_conv_1x1_block2x16_dot_packed_interleaved4(args);
                            out << "  our_conv1x1_block2x16_dot_packed_i4 : " << block2x16_dot_packed_i4.ms
                                      << " ms, " << block2x16_dot_packed_i4.gops << " GOPS\n";
                            if (!args.src_signed) {
                                const auto block2x16_dot_packed_i4_exec =
                                    bench_our_conv_1x1_block2x16_dot_packed_interleaved4(args, true);
                                out << "  our_conv1x1_block2x16_dot_packed_i4_exec_u8 : "
                                          << block2x16_dot_packed_i4_exec.ms << " ms, "
                                          << block2x16_dot_packed_i4_exec.gops << " GOPS\n";
                            }
                            const auto block4x16_dot_packed_i4 =
                                bench_our_conv_1x1_block4x16_dot_packed_interleaved4(args);
                            out << "  our_conv1x1_block4x16_dot_packed_i4 : " << block4x16_dot_packed_i4.ms
                                      << " ms, " << block4x16_dot_packed_i4.gops << " GOPS\n";
                            if (!args.src_signed) {
                                const auto block4x16_dot_packed_i4_exec =
                                    bench_our_conv_1x1_block4x16_dot_packed_interleaved4(args, true);
                                out << "  our_conv1x1_block4x16_dot_packed_i4_exec_u8 : "
                                          << block4x16_dot_packed_i4_exec.ms << " ms, "
                                          << block4x16_dot_packed_i4_exec.gops << " GOPS\n";
                            }
                            const auto block4x16_dot_packed_i4_lhs =
                                bench_our_conv_1x1_block4x16_dot_packed_lhs_interleaved4(args);
                            out << "  our_conv1x1_block4x16_dot_packed_i4_lhs : "
                                      << block4x16_dot_packed_i4_lhs.ms << " ms, "
                                      << block4x16_dot_packed_i4_lhs.gops << " GOPS\n";
                        } else {
                            out << "  our_conv1x1_block2x32_dot_packed_i4 : unsupported (OC multiple of 32)\n";
                            out << "  our_conv1x1_block2x16_dot_packed_i4 : unsupported (OC multiple of 16)\n";
                            out << "  our_conv1x1_block4x16_dot_packed_i4 : unsupported (OC multiple of 16)\n";
                            out
                                << "  our_conv1x1_block4x16_dot_packed_i4_lhs : unsupported (OC multiple of 16)\n";
                        }
                    } else {
                        out << "  our_conv1x1_block2x8_dot_packed_i4 : unsupported (IC multiple of 4)\n";
                        out << "  our_conv1x1_block2x16_dot_packed_i4 : unsupported (IC multiple of 4)\n";
                        out << "  our_conv1x1_block2x32_dot_packed_i4 : unsupported (IC multiple of 4)\n";
                        out << "  our_conv1x1_block4x16_dot_packed_i4 : unsupported (IC multiple of 4)\n";
                        out << "  our_conv1x1_block4x16_dot_packed_i4_lhs : unsupported (IC multiple of 4)\n";
                    }
                } else {
                    out << "  our_conv1x1_block8_dot : unsupported (OC multiple of 8)\n";
                    out << "  our_conv1x1_block8_dot_packed : unsupported (OC multiple of 8)\n";
                    out << "  our_conv1x1_block2x8_dot_packed : unsupported (OC multiple of 8)\n";
                    out << "  our_conv1x1_block2x8_dot_packed_i4 : unsupported (OC multiple of 8)\n";
                    out << "  our_conv1x1_block2x16_dot_packed_i4 : unsupported (OC multiple of 16)\n";
                    out << "  our_conv1x1_block2x32_dot_packed_i4 : unsupported (OC multiple of 32)\n";
                }
                    if (args.OC % 4 == 0) {
                        const auto block4x4_dot = bench_our_conv_1x1_block4x4_dot(args);
                        out << "  our_conv1x1_block4x4_dot : " << block4x4_dot.ms << " ms, "
                                  << block4x4_dot.gops << " GOPS\n";
                        const auto block4x4_dot_packed = bench_our_conv_1x1_block4x4_dot_packed(args);
                        out << "  our_conv1x1_block4x4_dot_packed : " << block4x4_dot_packed.ms << " ms, "
                                  << block4x4_dot_packed.gops << " GOPS\n";
                    } else {
                        out << "  our_conv1x1_block4x4_dot : unsupported (OC multiple of 4)\n";
                        out << "  our_conv1x1_block4x4_dot_packed : unsupported (OC multiple of 4)\n";
                    }
                }
                if (has_i8mm()) {
                    if (args.OC % 4 == 0 && args.IC % 8 == 0) {
                        const auto block4x4_mmla = bench_our_conv_1x1_block4x4_mmla_packed(args);
                        out << "  our_conv1x1_block4x4_mmla_packed : " << block4x4_mmla.ms << " ms, "
                                  << block4x4_mmla.gops << " GOPS\n";
                    } else {
                        out << "  our_conv1x1_block4x4_mmla_packed : unsupported (OC multiple of 4, IC multiple of 8)\n";
                    }
                    if (args.OC % 8 == 0 && args.IC % 8 == 0) {
                        const auto block4x8_mmla = bench_our_conv_1x1_block4x8_mmla_packed(args);
                        out << "  our_conv1x1_block4x8_mmla_packed : " << block4x8_mmla.ms << " ms, "
                                  << block4x8_mmla.gops << " GOPS\n";
                    } else {
                        out << "  our_conv1x1_block4x8_mmla_packed : unsupported (OC multiple of 8, IC multiple of 8)\n";
                    }
                    if (args.OC % 16 == 0 && args.IC % 8 == 0) {
                        const auto block4x16_mmla = bench_our_conv_1x1_block4x16_mmla_packed(args);
                        out << "  our_conv1x1_block4x16_mmla_packed : " << block4x16_mmla.ms << " ms, "
                                  << block4x16_mmla.gops << " GOPS\n";
                    } else {
                        out << "  our_conv1x1_block4x16_mmla_packed : unsupported (OC multiple of 16, IC multiple of 8)\n";
                    }
                    if (args.IC % 8 == 0) {
                        const auto gemm_mmla = bench_our_conv_1x1_gemm_mmla(args);
                        out << "  our_conv1x1_gemm_mmla : " << gemm_mmla.ms << " ms, " << gemm_mmla.gops
                                  << " GOPS\n";
                    } else {
                        out << "  our_conv1x1_gemm_mmla : unsupported (IC multiple of 8)\n";
                    }
                    if (args.W >= 8 && args.OC % 8 == 0 && args.IC % 8 == 0) {
                        const auto block8x8_mmla = bench_our_conv_1x1_block8x8_mmla_packed(args);
                        out << "  our_conv1x1_block8x8_mmla_packed : " << block8x8_mmla.ms << " ms, "
                                  << block8x8_mmla.gops << " GOPS\n";
                        if (!args.src_signed) {
                            const auto block8x8_mmla_exec = bench_our_conv_1x1_block8x8_mmla_packed(args, true);
                            out << "  our_conv1x1_block8x8_mmla_packed_exec_u8 : "
                                      << block8x8_mmla_exec.ms << " ms, " << block8x8_mmla_exec.gops << " GOPS\n";
                        }
                    } else {
                        out << "  our_conv1x1_block8x8_mmla_packed : unsupported (W>=8, OC multiple of 8, IC multiple of 8)\n";
                    }
                    if (args.W >= 8 && args.OC % 12 == 0 && args.IC % 8 == 0) {
                        const auto block8x12_mmla = bench_our_conv_1x1_block8x12_mmla_packed(args);
                        out << "  our_conv1x1_block8x12_mmla_packed : " << block8x12_mmla.ms << " ms, "
                                  << block8x12_mmla.gops << " GOPS\n";
                        if (!args.src_signed) {
                            const auto block8x12_mmla_exec = bench_our_conv_1x1_block8x12_mmla_packed(args, true);
                            out << "  our_conv1x1_block8x12_mmla_packed_exec_u8 : "
                                      << block8x12_mmla_exec.ms << " ms, " << block8x12_mmla_exec.gops << " GOPS\n";
                        }
                    } else {
                        out
                            << "  our_conv1x1_block8x12_mmla_packed : unsupported (W>=8, OC multiple of 12, IC multiple of 8)\n";
                    }
                }
                }
            }
        } else if ((args.KH == args.KW) && (args.KH == 3 || args.KH == 5)) {
            if (args.use_our) {
                if (args.pack_only) {
                    if (!our_groups_ok) {
                        out << "  our_conv" << args.KH << "x" << args.KW
                                  << "_pack_only : unsupported (groups=1)\n";
                    } else {
                        const auto pack_only = bench_our_conv_kxk_mmla_packed_fused(args, 8, true);
                        if (pack_only.ms < 0.0) {
                            out << "  our_conv" << args.KH << "x" << args.KW
                                      << "_pack_only : unsupported (IC multiple of 8, supported OC blocking)\n";
                            flush_report();
                            return 0;
                        }
                        out << "  our_conv" << args.KH << "x" << args.KW << "_pack_only : "
                                  << pack_only.ms << " ms\n";
                    }
                    flush_report();
                    return 0;
                }
                if (!our_groups_ok) {
                    // fall through to ACL/KleidiAI
                } else if (!our_default_geom) {
                    const auto conv_kxk_brgemm8x8 = bench_our_conv_kxk_brgemm_mbnb(args, 8, 8);
                    const auto conv_kxk_brgemm8x8_offs = bench_our_conv_kxk_brgemm_offs_mbnb(args, 8, 8);
                    if (conv_kxk_brgemm8x8.ms >= 0.0) {
                        out << "  our_conv" << args.KH << "x" << args.KW << "_brgemm8x8_direct : "
                                  << conv_kxk_brgemm8x8.ms << " ms, " << conv_kxk_brgemm8x8.gops << " GOPS\n";
                    } else {
                        out << "  our_conv" << args.KH << "x" << args.KW
                                  << "_brgemm8x8_direct : unsupported (SVE, stride=1, OC multiple of 8)\n";
                    }
                    if (conv_kxk_brgemm8x8_offs.ms >= 0.0) {
                        out << "  our_conv" << args.KH << "x" << args.KW << "_brgemm8x8_offs_direct : "
                                  << conv_kxk_brgemm8x8_offs.ms << " ms, " << conv_kxk_brgemm8x8_offs.gops
                                  << " GOPS\n";
                    } else {
                        out << "  our_conv" << args.KH << "x" << args.KW
                                  << "_brgemm8x8_offs_direct : unsupported (SVE brgemm offs)\n";
                    }
                    const auto conv_kxk_brgemm8x12 = bench_our_conv_kxk_brgemm_mbnb(args, 8, 12);
                    if (conv_kxk_brgemm8x12.ms >= 0.0) {
                        out << "  our_conv" << args.KH << "x" << args.KW << "_brgemm8x12_direct : "
                                  << conv_kxk_brgemm8x12.ms << " ms, " << conv_kxk_brgemm8x12.gops << " GOPS\n";
                    } else {
                        out << "  our_conv" << args.KH << "x" << args.KW
                                  << "_brgemm8x12_direct : unsupported (SVE, stride=1, OC multiple of 12)\n";
                    }
                    const auto conv_kxk_mmla4 = bench_our_conv_kxk_mmla_packed(args, 4);
                    if (conv_kxk_mmla4.ms >= 0.0) {
                        out << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x4_direct : "
                                  << conv_kxk_mmla4.ms << " ms, " << conv_kxk_mmla4.gops << " GOPS\n";
                    } else {
                        out << "  our_conv" << args.KH << "x" << args.KW
                                  << "_mmla4x4_direct : unsupported (IC multiple of 8, OC multiple of 4, i8mm)\n";
                    }
                    const auto conv_kxk_mmla8 = bench_our_conv_kxk_mmla_packed(args, 8);
                    if (conv_kxk_mmla8.ms >= 0.0) {
                        out << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x8_direct : "
                                  << conv_kxk_mmla8.ms << " ms, " << conv_kxk_mmla8.gops << " GOPS\n";
                    } else {
                        out << "  our_conv" << args.KH << "x" << args.KW
                                  << "_mmla4x8_direct : unsupported (IC multiple of 8, OC multiple of 8, i8mm)\n";
                    }
                    const auto conv_kxk_mmla16 = bench_our_conv_kxk_mmla_packed(args, 16);
                    if (conv_kxk_mmla16.ms >= 0.0) {
                        out << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x16_direct : "
                                  << conv_kxk_mmla16.ms << " ms, " << conv_kxk_mmla16.gops << " GOPS\n";
                    } else {
                        out << "  our_conv" << args.KH << "x" << args.KW
                                  << "_mmla4x16_direct : unsupported (IC multiple of 8, OC multiple of 16, i8mm)\n";
                    }
                    const auto conv_kxk_mmla8x8_fused = bench_our_conv_kxk_mmla_packed_fused(args, 8, true);
                    if (conv_kxk_mmla8x8_fused.ms >= 0.0) {
                        out << "  our_conv" << args.KH << "x" << args.KW << "_mmla8x8_fused : "
                                  << conv_kxk_mmla8x8_fused.ms << " ms, " << conv_kxk_mmla8x8_fused.gops
                                  << " GOPS\n";
                    } else {
                        out << "  our_conv" << args.KH << "x" << args.KW
                                  << "_mmla8x8_fused : unsupported (dilation=1, IC multiple of 8, i8mm)\n";
                    }
                    const auto conv_kxk_mmla8x12_fused = bench_our_conv_kxk_mmla_packed_fused(args, 12, true);
                    if (conv_kxk_mmla8x12_fused.ms >= 0.0) {
                        out << "  our_conv" << args.KH << "x" << args.KW << "_mmla8x12_fused : "
                                  << conv_kxk_mmla8x12_fused.ms << " ms, " << conv_kxk_mmla8x12_fused.gops
                                  << " GOPS\n";
                    } else {
                        out << "  our_conv" << args.KH << "x" << args.KW
                                  << "_mmla8x12_fused : unsupported (dilation=1, IC multiple of 8, i8mm)\n";
                    }
                    const auto conv_kxk_mmla4x16_fused = bench_our_conv_kxk_mmla_packed_fused(args, 16, false);
                    if (conv_kxk_mmla4x16_fused.ms >= 0.0) {
                        out << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x16_fused : "
                                  << conv_kxk_mmla4x16_fused.ms << " ms, " << conv_kxk_mmla4x16_fused.gops
                                  << " GOPS\n";
                    } else {
                        out << "  our_conv" << args.KH << "x" << args.KW
                                  << "_mmla4x16_fused : unsupported (dilation=1, IC multiple of 8, i8mm)\n";
                    }
                    const auto conv_kxk_mmla_im2col = bench_our_conv_kxk_mmla_im2col(args);
                    if (conv_kxk_mmla_im2col.ms >= 0.0) {
                        out << "  our_conv" << args.KH << "x" << args.KW << "_mmla_im2col : "
                                  << conv_kxk_mmla_im2col.ms << " ms, " << conv_kxk_mmla_im2col.gops
                                  << " GOPS\n";
                    } else {
                        out << "  our_conv" << args.KH << "x" << args.KW
                                  << "_mmla_im2col : unsupported (K multiple of 8, i8mm)\n";
                    }
                } else {
                const auto conv_kxk = bench_our_conv_kxk(args);
                const auto conv_kxk_dot4x4 = bench_our_conv_kxk_block4x4_dot(args);
                const auto conv_kxk_mmla4 = bench_our_conv_kxk_mmla_packed(args, 4);
                const auto conv_kxk_mmla8 = bench_our_conv_kxk_mmla_packed(args, 8);
                const auto conv_kxk_mmla16 = bench_our_conv_kxk_mmla_packed(args, 16);
                const auto conv_kxk_mmla4_fused = bench_our_conv_kxk_mmla_packed_fused(args, 4, false);
                const auto conv_kxk_mmla8_fused = bench_our_conv_kxk_mmla_packed_fused(args, 8, false);
                const auto conv_kxk_mmla8x8_fused = bench_our_conv_kxk_mmla_packed_fused(args, 8, true);
                const auto conv_kxk_mmla8x12_fused = bench_our_conv_kxk_mmla_packed_fused(args, 12, true);
                const auto conv_kxk_mmla16_fused = bench_our_conv_kxk_mmla_packed_fused(args, 16, false);
                const auto conv_kxk_brgemm = bench_our_conv_kxk_brgemm(args);
                const auto conv_kxk_brgemm8x8 = bench_our_conv_kxk_brgemm_mbnb(args, 8, 8);
                const auto conv_kxk_brgemm8x8_offs = bench_our_conv_kxk_brgemm_offs_mbnb(args, 8, 8);
                const auto conv_kxk_brgemm8x12 = bench_our_conv_kxk_brgemm_mbnb(args, 8, 12);
                const auto conv_kxk_brgemm_fused8x16 = bench_our_conv_kxk_brgemm_fused_mbnb(args, 8, 16);
                const auto conv_kxk_brgemm_im2col = bench_our_conv_kxk_brgemm_im2col(args, 32);
                const auto conv_kxk_mmla_im2col = bench_our_conv_kxk_mmla_im2col(args);
                out << "  our_conv" << args.KH << "x" << args.KW << " : " << conv_kxk.ms << " ms, "
                          << conv_kxk.gops << " GOPS\n";
                if (conv_kxk_dot4x4.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_dot4x4 : " << conv_kxk_dot4x4.ms
                              << " ms, " << conv_kxk_dot4x4.gops << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_dot4x4 : unsupported (dotprod unavailable)\n";
                }
                if (conv_kxk_mmla4.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x4 : " << conv_kxk_mmla4.ms
                              << " ms, " << conv_kxk_mmla4.gops << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla4x4 : unsupported (OC multiple of 4, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_mmla8.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x8 : " << conv_kxk_mmla8.ms
                              << " ms, " << conv_kxk_mmla8.gops << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla4x8 : unsupported (OC multiple of 8, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_mmla16.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x16 : " << conv_kxk_mmla16.ms
                              << " ms, " << conv_kxk_mmla16.gops << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla4x16 : unsupported (OC multiple of 16, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_mmla4_fused.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x4_fused : "
                              << conv_kxk_mmla4_fused.ms << " ms, " << conv_kxk_mmla4_fused.gops << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla4x4_fused : unsupported (OC multiple of 4, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_mmla8_fused.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x8_fused : "
                              << conv_kxk_mmla8_fused.ms << " ms, " << conv_kxk_mmla8_fused.gops << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla4x8_fused : unsupported (OC multiple of 8, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_mmla8x8_fused.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_mmla8x8_fused : "
                              << conv_kxk_mmla8x8_fused.ms << " ms, " << conv_kxk_mmla8x8_fused.gops << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla8x8_fused : unsupported (OW multiple of 8, OC multiple of 8, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_mmla8x12_fused.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_mmla8x12_fused : "
                              << conv_kxk_mmla8x12_fused.ms << " ms, " << conv_kxk_mmla8x12_fused.gops << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla8x12_fused : unsupported (OW multiple of 8, OC multiple of 12, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_mmla16_fused.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_mmla4x16_fused : "
                              << conv_kxk_mmla16_fused.ms << " ms, " << conv_kxk_mmla16_fused.gops << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla4x16_fused : unsupported (OC multiple of 16, IC multiple of 8, i8mm)\n";
                }
                if (conv_kxk_brgemm.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_brgemm4x4 : " << conv_kxk_brgemm.ms
                              << " ms, " << conv_kxk_brgemm.gops << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_brgemm4x4 : unsupported (brgemm unavailable)\n";
                }
                if (conv_kxk_brgemm8x8.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_brgemm8x8 : " << conv_kxk_brgemm8x8.ms
                              << " ms, " << conv_kxk_brgemm8x8.gops << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_brgemm8x8 : unsupported (brgemm unavailable)\n";
                }
                if (conv_kxk_brgemm8x8_offs.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_brgemm8x8_offs : "
                              << conv_kxk_brgemm8x8_offs.ms << " ms, " << conv_kxk_brgemm8x8_offs.gops
                              << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_brgemm8x8_offs : unsupported (brgemm offs unavailable)\n";
                }
                if (conv_kxk_brgemm8x12.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_brgemm8x12 : " << conv_kxk_brgemm8x12.ms
                              << " ms, " << conv_kxk_brgemm8x12.gops << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_brgemm8x12 : unsupported (brgemm unavailable)\n";
                }
                if (conv_kxk_brgemm_fused8x16.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_brgemm_fused8x16 : "
                              << conv_kxk_brgemm_fused8x16.ms << " ms, " << conv_kxk_brgemm_fused8x16.gops
                              << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_brgemm_fused8x16 : unsupported (brgemm unavailable)\n";
                }
                if (conv_kxk_brgemm_im2col.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_brgemm_im2col : "
                              << conv_kxk_brgemm_im2col.ms << " ms, " << conv_kxk_brgemm_im2col.gops
                              << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_brgemm_im2col : unsupported (brgemm unavailable)\n";
                }
                if (conv_kxk_mmla_im2col.ms >= 0.0) {
                    out << "  our_conv" << args.KH << "x" << args.KW << "_mmla_im2col : "
                              << conv_kxk_mmla_im2col.ms << " ms, " << conv_kxk_mmla_im2col.gops
                              << " GOPS\n";
                } else {
                    out << "  our_conv" << args.KH << "x" << args.KW
                              << "_mmla_im2col : unsupported (K multiple of 8, i8mm)\n";
                }
                }
            }
        } else {
            out << "  our_conv    : unsupported (only 1x1, 3x3, 5x5)\n";
        }
#if defined(OV_CPU_WITH_ACL)
        if (args.use_acl) {
            try {
                if (OH == 0 || OW == 0) {
                    out << "  acl_conv    : unsupported (invalid output shape)\n";
                } else if (args.groups != 1) {
                    out << "  acl_conv    : unsupported (groups!=1)\n";
                } else {
                const auto acl = bench_acl_conv_1x1(args);
                out << "  acl_conv    : " << acl.ms << " ms, " << acl.gops << " GOPS\n";
                }
            } catch (const std::exception& ex) {
                out << "  acl_conv    : failed (" << ex.what() << ")\n";
            }
        }
#else
        if (args.use_acl) {
            out << "  acl_conv    : unavailable (built without ACL)\n";
        }
#endif
#if defined(OV_CPU_WITH_KLEIDIAI)
        if (args.use_kleidiai) {
            bool supported = false;
            if (OH == 0 || OW == 0) {
                out << "  kleidiai_conv : unsupported (invalid output shape)\n";
            } else if (args.groups != 1) {
                out << "  kleidiai_conv : unsupported (groups!=1)\n";
            } else if (args.KH == 1 && args.KW == 1) {
                const auto kleidiai_total = bench_kleidiai_conv_1x1_total(args, supported);
                if (!supported) {
                    out << "  kleidiai_conv1x1 : unsupported (requires dotprod/i8mm, K multiple of 8)\n";
                } else {
                    out << "  kleidiai_conv1x1_total : " << kleidiai_total.ms << " ms, " << kleidiai_total.gops
                              << " GOPS\n";
                    const auto kleidiai = bench_kleidiai_conv_1x1(args, supported);
                    out << "  kleidiai_conv1x1 : " << kleidiai.ms << " ms, " << kleidiai.gops << " GOPS\n";
                }
            } else if ((args.KH == args.KW) && (args.KH == 3 || args.KH == 5)) {
                const auto kleidiai_total = bench_kleidiai_conv_kxk_total(args, supported);
                if (!supported) {
                    out << "  kleidiai_conv" << args.KH << "x" << args.KW
                              << " : unsupported (requires dotprod/i8mm, K multiple of 8)\n";
                } else {
                    out << "  kleidiai_conv" << args.KH << "x" << args.KW << "_total : "
                              << kleidiai_total.ms << " ms, " << kleidiai_total.gops << " GOPS\n";
                    const auto kleidiai = bench_kleidiai_conv_kxk(args, supported);
                    out << "  kleidiai_conv" << args.KH << "x" << args.KW
                              << " : " << kleidiai.ms << " ms, " << kleidiai.gops << " GOPS\n";
                }
            } else {
                out << "  kleidiai_conv : unsupported (only 1x1, 3x3, 5x5)\n";
            }
        }
#else
        if (args.use_kleidiai) {
            out << "  kleidiai_conv : unavailable (built without KleidiAI)\n";
        }
#endif
    }

    flush_report();
    return 0;
}
