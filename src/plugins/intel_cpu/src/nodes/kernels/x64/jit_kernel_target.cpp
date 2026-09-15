// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_target.hpp"

#include <cpu/x64/cpu_isa_traits.hpp>

#include <cstdlib>
#include <limits>
#include <string>

namespace ov::intel_cpu {

namespace {

using namespace dnnl::impl::cpu::x64;

// Debug override for the tail-folding decision, the counterpart of LLVM's
// -prefer-predicate-over-epilogue. Two reasons to have it: A/B measurement
// of the two strategies on the same host, and coverage — on an AVX-512
// machine the epilogue path is otherwise unreachable and would rot.
std::optional<vector_target::tail_folding> tail_folding_override() {
    static const std::optional<vector_target::tail_folding> value = [] {
        const char* env = std::getenv("OV_JIT_TAIL_FOLDING");
        if (env == nullptr) {
            return std::optional<vector_target::tail_folding>{};
        }
        const std::string requested(env);
        if (requested == "epilogue") {
            return std::optional{vector_target::tail_folding::epilogue};
        }
        if (requested == "mask") {
            return std::optional{vector_target::tail_folding::mask};
        }
        if (requested == "length") {
            return std::optional{vector_target::tail_folding::length};
        }
        return std::optional<vector_target::tail_folding>{};
    }();
    return value;
}

// AVX-512: every memory form the DSL emits has a masked encoding —
// vmovups{k}{z} for f32, vpmovzxbd/vpmovusdb for u8, vcvtph2ps/vcvtps2ph
// for f16, vpmovzxwd/vpmovdw for bf16.
struct avx512_target final : vector_target {
    [[nodiscard]] bool supports_masked_access(std::size_t elem_bytes) const override {
        return elem_bytes == 1 || elem_bytes == 2 || elem_bytes == 4;
    }

    // x86 has no interleaved-store instruction; store_interleaved3 builds
    // the interleave with permutes and blends, so a predicated form would
    // need three separately-derived masks. Not implemented, so the tail
    // falls back to the counted (scalarized) path for interleaved access
    // only.
    [[nodiscard]] bool supports_masked_interleaved_access() const override { return false; }

    [[nodiscard]] tail_folding preferred_tail_folding() const override {
        return tail_folding_override().value_or(tail_folding::mask);
    }

    // Any displacement an unrolled loop can produce rides in the SIB byte.
    [[nodiscard]] bool is_legal_access_offset(std::size_t /*elem_bytes*/,
                                              std::size_t /*vectors*/,
                                              std::size_t bytes) const override {
        return bytes <= std::numeric_limits<std::int32_t>::max();
    }

    // k1..k7: k0 exists but cannot be used as a write-mask.
    [[nodiscard]] const std::vector<std::uint32_t>& predicate_pool() const override {
        static const std::vector<std::uint32_t> pool{1, 2, 3, 4, 5, 6, 7};
        return pool;
    }
};

// AVX2 has vmaskmovps/vpmaskmovd for 4-byte elements only, and no masked
// form of the narrowing conversions the DSL uses, so predication cannot
// cover the type set. Report no masked access and fold tails with an
// epilogue, which is also what LLVM's cost model picks for pre-AVX-512 x86.
struct legacy_x86_target final : vector_target {
    [[nodiscard]] bool supports_masked_access(std::size_t /*elem_bytes*/) const override {
        return false;
    }
    [[nodiscard]] bool supports_masked_interleaved_access() const override { return false; }
    [[nodiscard]] tail_folding preferred_tail_folding() const override {
        return tail_folding_override().value_or(tail_folding::epilogue);
    }

    [[nodiscard]] bool is_legal_access_offset(std::size_t /*elem_bytes*/,
                                              std::size_t /*vectors*/,
                                              std::size_t bytes) const override {
        return bytes <= std::numeric_limits<std::int32_t>::max();
    }

    [[nodiscard]] const std::vector<std::uint32_t>& predicate_pool() const override {
        static const std::vector<std::uint32_t> empty;
        return empty;
    }
};

}  // namespace

const vector_target& host_vector_target() {
    if (mayiuse(cpu_isa_t::avx512_core)) {
        static const avx512_target target;
        return target;
    }
    static const legacy_x86_target target;
    return target;
}

}  // namespace ov::intel_cpu
