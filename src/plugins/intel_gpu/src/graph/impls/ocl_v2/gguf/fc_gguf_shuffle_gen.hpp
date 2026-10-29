// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// fc_gguf_shuffle_gen.hpp — Shared OCL shuffle GEMV kernel generators for
// GGUF FC nodes.
//
// Included by both the OCL impl (fc_gguf_opt.cpp) and the CM impl
// (fc_gguf_cm.cpp) so both share the same shuffle GEMV kernel classes without
// code duplication.  These generators target the fc_gguf_q{4k,5k,6k}_sg.cl
// and fc_gguf_q{4_0,4_1,8_0}_sg.cl OCL kernels that operate on the
// SG-transposed plane-separated weight layout produced by
// RepackGGUFWeightsShuffle.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "../utils/fused_ops_jitter.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

// Subgroup width for the K-split GEMV: one subgroup (this many lanes) cooperatively computes one
// output, the reduction blocks of a row striped across its lanes.  16 matches the BMG/Xe2 native
// SIMD width and divides every shape's blocks-per-row (K is a multiple of 256 -> >= 16 blocks).
constexpr int GGUF_GEMV_SG_SIZE = 16;

// K-split (occupancy) factor for the shuffle GEMV.
// Mirrors choose_ksplit() in q4k_gemv/test_gemv_sg_kernels.py.
// Overridable via OV_GPU_GGUF_SHUFFLE_KSPLIT.
inline int gguf_shuffle_choose_ksplit(size_t N, size_t K) {
    const long TARGET_THREADS = 4096;
    const long BASE_KEEP      = 512;
    const long MAX_KSPLIT     = 16;
    const long MIN_BLK_PER_SG = 2;
    const long MIN_THREADS    = 1024;
    const long OPG            = GGUF_GEMV_SG_SIZE;

    if (const char* env = std::getenv("OV_GPU_GGUF_SHUFFLE_KSPLIT")) {
        const long v = std::atol(env);
        if (v >= 1) return static_cast<int>(v);
    }

    const long nbpr = static_cast<long>(K) / 256;
    const long base = std::max<long>(1, static_cast<long>(N) / OPG);
    if (base >= BASE_KEEP) return 1;

    const long want = (TARGET_THREADS + base / 2) / base;
    long capped = std::max<long>(1, want);
    capped = std::min<long>(capped, nbpr);
    capped = std::min<long>(capped, MAX_KSPLIT);
    long ks = 1;
    while (ks * 2 <= capped) ks *= 2;
    while (ks > 1 && (nbpr / ks) < MIN_BLK_PER_SG && base * (ks / 2) >= MIN_THREADS) ks /= 2;
    return static_cast<int>(ks);
}

// =================================================================================================
// Q4_K / Q5_K / Q6_K / Q4_0 / Q4_1 / Q8_0 weight-shuffle GEMV generator.
// Targets fc_gguf_q{4k,5k,6k}_sg.cl and fc_gguf_q{4_0,4_1,8_0}_sg.cl.
// Reads the SG-transposed plane-separated weight with intel_sub_group_block_read.
// Activation is f16 (INPUT0). Explicit-args; static K, N + output dtype baked in.
// =================================================================================================
class FCGGUFShuffleGenerator : public KernelGenerator {
public:
    explicit FCGGUFShuffleGenerator(const char* kernel_name) : KernelGenerator(kernel_name) {}

protected:
    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override {
        const auto& in1 = params.input_layouts[1];
        std::string name = get_kernel_name() +
                           "_K" + std::to_string(in1.get_shape()[1]) +
                           "_N" + std::to_string(in1.get_shape()[0]);
        if (params.is_dynamic()) {
            name += "__sa";  // shape-agnostic; distinguishes from the static-shape variant
        }
        return name;
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);
        const auto& in0 = params.input_layouts[0];  // activation [BM, K]
        const auto& in1 = params.input_layouts[1];  // gguf weight [N, K]
        const size_t N  = in1.get_shape()[0];
        const size_t K  = in1.get_shape()[1];
        jit.add(make_tensors_jit_constants(params));
        jit.add(make_type_jit_constants("INPUT0", in0.data_type));
        jit.add(make_type_jit_constants("OUTPUT", params.output_layouts[0].data_type));
        jit.add({
            make_jit_constant("K_SIZE",  static_cast<int>(K)),
            make_jit_constant("N_SIZE",  static_cast<int>(N)),
            make_jit_constant("SG_SIZE", GGUF_GEMV_SG_SIZE),
            make_jit_constant("OPG",     GGUF_GEMV_SG_SIZE),
            make_jit_constant("KSPLIT",  gguf_shuffle_choose_ksplit(N, K)),
        });
        if (params.has_fused_primitives()) {
            const auto& out_l = params.output_layouts[0];
            const std::vector<std::string> idx_order = out_l.get_partial_shape().size() == 2
                ? std::vector<std::string>{"bm", "n", "0", "0"}
                : std::vector<std::string>{"out_b", "out_f", "n", "0"};
            FusedOpsConfiguration conf = {"", idx_order, "dequantized", out_l.data_type};
            jit.add(make_fused_ops_jit_constants(params, {conf}));
        }
        return jit;
    }

    // Empty descriptor: explicit-args dispatch via execute_shuffle_gemv.
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {}};
    }
};

// Per-format subclasses — one per OCL kernel entry point.
class FCGGUFQ4kSgGenerator : public FCGGUFShuffleGenerator {
public: FCGGUFQ4kSgGenerator() : FCGGUFShuffleGenerator("fc_gguf_q4k_sg") {}
};
class FCGGUFQ5kSgGenerator : public FCGGUFShuffleGenerator {
public: FCGGUFQ5kSgGenerator() : FCGGUFShuffleGenerator("fc_gguf_q5k_sg") {}
};
class FCGGUFQ6kSgGenerator : public FCGGUFShuffleGenerator {
public: FCGGUFQ6kSgGenerator() : FCGGUFShuffleGenerator("fc_gguf_q6k_sg") {}
};
class FCGGUFQ40SgGenerator : public FCGGUFShuffleGenerator {
public: FCGGUFQ40SgGenerator() : FCGGUFShuffleGenerator("fc_gguf_q4_0_sg") {}
};
class FCGGUFQ41SgGenerator : public FCGGUFShuffleGenerator {
public: FCGGUFQ41SgGenerator() : FCGGUFShuffleGenerator("fc_gguf_q4_1_sg") {}
};
class FCGGUFQ80SgGenerator : public FCGGUFShuffleGenerator {
public: FCGGUFQ80SgGenerator() : FCGGUFShuffleGenerator("fc_gguf_q8_0_sg") {}
};

}  // namespace ov::intel_gpu::ocl
