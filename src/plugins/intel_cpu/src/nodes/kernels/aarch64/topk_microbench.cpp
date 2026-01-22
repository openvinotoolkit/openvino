// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/visibility.hpp"

#if !defined(OPENVINO_ARCH_ARM64)
#include <iostream>
int main() {
    std::cerr << "topk_microbench: build is not for AArch64\n";
    return 0;
}
#else

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <cpu/aarch64/cpu_isa_traits.hpp>

#include "nodes/kernels/aarch64/jit_uni_topk_kernel.hpp"
#include "openvino/core/type/float16.hpp"

namespace {

struct Options {
    std::string dtype = "f32";
    std::string layout = "ncsp";
    std::string mode = "max";
    size_t O = 1;
    size_t A = 1024;
    size_t I = 1;
    size_t top_k = 10;
    size_t blk = 16;
    bool topk_innermost = false;
    bool sort_index = false;
    bool stable = false;
    size_t iters = 50;
    size_t warmup = 5;
    uint32_t seed = 777;
};

bool to_bool(const std::string& v) {
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

bool parse_kv(const std::string& arg, std::string& key, std::string& value) {
    const auto pos = arg.find('=');
    if (pos == std::string::npos) {
        return false;
    }
    key = arg.substr(2, pos - 2);
    value = arg.substr(pos + 1);
    return true;
}

void print_usage() {
    std::cout << "topk_microbench options (use --key=value):\n"
              << "  --dtype=f32|f16|i8|u8|i32\n"
              << "  --layout=ncsp|nspc|blocked\n"
              << "  --mode=max|min\n"
              << "  --O=<outer> --A=<axis_dim> --I=<inner>\n"
              << "  --top_k=<k>\n"
              << "  --blk=<block> (blocked only)\n"
              << "  --topk_innermost=0|1\n"
              << "  --sort_index=0|1\n"
              << "  --stable=0|1\n"
              << "  --iters=<n> --warmup=<n> --seed=<n>\n";
}

std::string isa_name() {
    using namespace dnnl::impl::cpu::aarch64;
    if (mayiuse(sve_512)) return "sve_512";
    if (mayiuse(sve_384)) return "sve_384";
    if (mayiuse(sve_256)) return "sve_256";
    if (mayiuse(sve_128)) return "sve_128";
    if (mayiuse(asimd)) return "asimd";
    return "unknown";
}

template <typename T>
void fill_data(std::vector<T>& data, uint32_t seed) {
    std::mt19937 gen(seed);
    if constexpr (std::is_same_v<T, ov::float16>) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : data) v = ov::float16(dist(gen));
    } else if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : data) v = static_cast<T>(dist(gen));
    } else if constexpr (std::is_signed_v<T>) {
        std::uniform_int_distribution<int> dist(-100, 100);
        for (auto& v : data) v = static_cast<T>(dist(gen));
    } else {
        std::uniform_int_distribution<int> dist(0, 200);
        for (auto& v : data) v = static_cast<T>(dist(gen));
    }
}

size_t div_up(size_t v, size_t d) {
    return (v + d - 1) / d;
}

struct BenchContext {
    ov::element::Type precision;
    size_t data_size = 0;
    size_t total_elems = 0;
    size_t out_elems = 0;
    std::vector<uint8_t> src;
    std::vector<uint8_t> dst;
    std::vector<uint8_t> idx;
};

template <typename T>
void init_buffers(BenchContext& ctx, const Options& opt) {
    const size_t elem_size = sizeof(T);
    ctx.data_size = elem_size;
    if (opt.layout == "blocked" && opt.topk_innermost) {
        const size_t IA = div_up(opt.A, opt.blk);
        const size_t OA = div_up(opt.top_k, opt.blk);
        ctx.total_elems = opt.O * IA * opt.I * opt.blk;
        ctx.out_elems = opt.O * OA * opt.I * opt.blk;
    } else {
        ctx.total_elems = opt.O * opt.A * opt.I;
        ctx.out_elems = opt.O * opt.top_k * opt.I;
    }

    ctx.src.resize(ctx.total_elems * elem_size);
    ctx.dst.resize(ctx.out_elems * elem_size);
    ctx.idx.resize(ctx.out_elems * sizeof(int32_t));

    std::vector<T> tmp(ctx.total_elems);
    fill_data(tmp, opt.seed);
    std::memcpy(ctx.src.data(), tmp.data(), ctx.src.size());
}

void run_bench(const Options& opt) {
    if (opt.layout != "blocked") {
        if (opt.topk_innermost) {
            std::cerr << "topk_innermost is only valid for blocked layout\n";
        }
    }

    BenchContext ctx{};
    if (opt.dtype == "f32") {
        ctx.precision = ov::element::f32;
        init_buffers<float>(ctx, opt);
    } else if (opt.dtype == "f16") {
        ctx.precision = ov::element::f16;
        init_buffers<ov::float16>(ctx, opt);
    } else if (opt.dtype == "i8") {
        ctx.precision = ov::element::i8;
        init_buffers<int8_t>(ctx, opt);
    } else if (opt.dtype == "u8") {
        ctx.precision = ov::element::u8;
        init_buffers<uint8_t>(ctx, opt);
    } else if (opt.dtype == "i32") {
        ctx.precision = ov::element::i32;
        init_buffers<int32_t>(ctx, opt);
    } else {
        std::cerr << "Unsupported dtype: " << opt.dtype << "\n";
        std::exit(1);
    }

    ov::intel_cpu::node::jit_topk_config_params jcp{};
    jcp.precision = ctx.precision;
    jcp.data_size = static_cast<int>(ctx.data_size);
    jcp.blk_size = static_cast<int>(opt.layout == "blocked" ? opt.blk : 1);
    jcp.layout = (opt.layout == "blocked") ? ov::intel_cpu::node::TopKLayoutType::topk_blocked
                                            : (opt.layout == "nspc" ? ov::intel_cpu::node::TopKLayoutType::topk_nspc
                                                                   : ov::intel_cpu::node::TopKLayoutType::topk_ncsp);
    jcp.top_k = static_cast<int>(opt.top_k);
    jcp.axis_dim = static_cast<int>(opt.A);
    jcp.mode_max = (opt.mode == "max");
    jcp.sort_index = opt.sort_index;
    jcp.topk_innermost = opt.topk_innermost;
    jcp.algorithm = ov::intel_cpu::node::TopKAlgorithm::topk_bubble_sort;
    jcp.bubble_inplace = false;
    jcp.stable = opt.stable;
    jcp.sort_stride = static_cast<int>(opt.I);
    jcp.work_amount = static_cast<int>(opt.I);

    auto kernel = ov::intel_cpu::node::create_topk_kernel_aarch64(jcp);
    if (!kernel) {
        std::cerr << "Failed to create TopK kernel\n";
        std::exit(1);
    }
    kernel->create_ker();

    ov::intel_cpu::node::jit_topk_call_args args{};
    args.axis_dim = opt.A;
    args.top_k = opt.top_k;
    args.sort_stride = opt.I;
    args.config = &jcp;

    const auto run_once = [&]() {
        if (opt.layout == "blocked" && opt.topk_innermost) {
            const size_t IA = div_up(opt.A, opt.blk);
            const size_t OA = div_up(opt.top_k, opt.blk);
            for (size_t o = 0; o < opt.O; ++o) {
                for (size_t i = 0; i < opt.I; ++i) {
                    const size_t base_in = (o * IA * opt.I + i) * opt.blk;
                    const size_t base_out = (o * OA * opt.I + i) * opt.blk;
                    args.src = ctx.src.data() + base_in * ctx.data_size;
                    args.dst = ctx.dst.data() + base_out * ctx.data_size;
                    args.index = ctx.idx.data() + base_out * sizeof(int32_t);
                    args.work_amount = 1;
                    (*kernel)(&args);
                }
            }
        } else {
            const size_t blk = (opt.layout == "blocked") ? opt.blk : 1;
            const size_t blocks = opt.I / blk;
            const size_t tail = opt.I % blk;
            for (size_t o = 0; o < opt.O; ++o) {
                for (size_t k = 0; k < blocks; ++k) {
                    const size_t base = (o * opt.A * opt.I + k * blk);
                    const size_t out_base = (o * opt.top_k * opt.I + k * blk);
                    args.src = ctx.src.data() + base * ctx.data_size;
                    args.dst = ctx.dst.data() + out_base * ctx.data_size;
                    args.index = ctx.idx.data() + out_base * sizeof(int32_t);
                    args.work_amount = blk;
                    (*kernel)(&args);
                }
                if (tail) {
                    const size_t base = (o * opt.A * opt.I + blocks * blk);
                    const size_t out_base = (o * opt.top_k * opt.I + blocks * blk);
                    args.src = ctx.src.data() + base * ctx.data_size;
                    args.dst = ctx.dst.data() + out_base * ctx.data_size;
                    args.index = ctx.idx.data() + out_base * sizeof(int32_t);
                    args.work_amount = tail;
                    (*kernel)(&args);
                }
            }
        }
    };

    for (size_t i = 0; i < opt.warmup; ++i) {
        run_once();
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < opt.iters; ++i) {
        run_once();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const double us_per_iter = static_cast<double>(us) / static_cast<double>(opt.iters);

    const size_t elems = opt.O * opt.A * opt.I;
    const double elems_per_us = static_cast<double>(elems) / us_per_iter;

    std::cout << "TopK microbench (AArch64)\n"
              << "  ISA: " << isa_name() << "\n"
              << "  dtype: " << opt.dtype << "\n"
              << "  layout: " << opt.layout << "\n"
              << "  O=" << opt.O << " A=" << opt.A << " I=" << opt.I
              << " top_k=" << opt.top_k << " blk=" << (opt.layout == "blocked" ? opt.blk : 1)
              << " topk_innermost=" << (opt.topk_innermost ? 1 : 0) << "\n"
              << "  mode=" << opt.mode << " sort_index=" << (opt.sort_index ? 1 : 0)
              << " stable=" << (opt.stable ? 1 : 0) << "\n"
              << "  iters=" << opt.iters << " warmup=" << opt.warmup << "\n"
              << "  time: " << us_per_iter << " us/iter\n"
              << "  throughput: " << elems_per_us << " elems/us\n";
}

}  // namespace

int main(int argc, char** argv) {
    Options opt{};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            print_usage();
            return 0;
        }
        if (arg.rfind("--", 0) != 0) {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage();
            return 1;
        }
        std::string key, value;
        if (!parse_kv(arg, key, value)) {
            std::cerr << "Bad argument: " << arg << "\n";
            print_usage();
            return 1;
        }
        if (key == "dtype") opt.dtype = value;
        else if (key == "layout") opt.layout = value;
        else if (key == "mode") opt.mode = value;
        else if (key == "O") opt.O = static_cast<size_t>(std::stoull(value));
        else if (key == "A") opt.A = static_cast<size_t>(std::stoull(value));
        else if (key == "I") opt.I = static_cast<size_t>(std::stoull(value));
        else if (key == "top_k") opt.top_k = static_cast<size_t>(std::stoull(value));
        else if (key == "blk") opt.blk = static_cast<size_t>(std::stoull(value));
        else if (key == "topk_innermost") opt.topk_innermost = to_bool(value);
        else if (key == "sort_index") opt.sort_index = to_bool(value);
        else if (key == "stable") opt.stable = to_bool(value);
        else if (key == "iters") opt.iters = static_cast<size_t>(std::stoull(value));
        else if (key == "warmup") opt.warmup = static_cast<size_t>(std::stoull(value));
        else if (key == "seed") opt.seed = static_cast<uint32_t>(std::stoul(value));
        else {
            std::cerr << "Unknown option: " << key << "\n";
            print_usage();
            return 1;
        }
    }

    if (opt.layout != "blocked") {
        opt.blk = 1;
    }

    run_bench(opt);
    return 0;
}

#endif  // defined(OPENVINO_ARCH_ARM64)
