// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "emitters/snippets/x64/jit_brgemm_emitter.hpp"
#include "openvino/op/parameter.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

using namespace ov::intel_cpu;

namespace {

using dnnl::impl::cpu::x64::avx2_vnni;
using dnnl::impl::cpu::x64::avx2_vnni_2;
using dnnl::impl::cpu::x64::avx512_core;
using dnnl::impl::cpu::x64::avx512_core_vnni;
using dnnl::impl::cpu::x64::cpu_isa_t;

using ov::element::bf16;
using ov::element::f16;
using ov::element::f32;
using ov::element::i8;
using ov::element::u8;

using SupportedPrecisions = std::set<std::vector<ov::element::Type>>;
using IsaSupportedPrecisions = std::pair<cpu_isa_t, SupportedPrecisions>;

// Note: JIT_IMPL_NAME_HELPER needs the whole dnnl::impl::cpu::x64 namespace in scope, which makes
// avx and avx2 ambiguous with ov::intel_cpu::impl_desc_type.
std::string isa_name(cpu_isa_t isa) {
    switch (isa) {
    case avx2_vnni:
        return "avx2_vnni";
    case avx2_vnni_2:
        return "avx2_vnni_2";
    case avx512_core:
        return "avx512_core";
    case avx512_core_vnni:
        return "avx512_core_vnni";
    default:
        return "unknown_isa";
    }
}

class BrgemmSupportedPrecisions : public testing::TestWithParam<IsaSupportedPrecisions> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<IsaSupportedPrecisions>& obj) {
        return isa_name(obj.param.first);
    }
};

// Note: the ISA-taking BrgemmConfig constructor bypasses get_prim_isa(), which prefers amx and
// avx512_core_vnni over avx2_vnni_2 for int8, so the avx2_vnni_2 precision set is otherwise
// unreachable on a CI machine that has avx512.
TEST_P(BrgemmSupportedPrecisions, GetSupportedPrecisions) {
    const auto& [isa, expected_precisions] = GetParam();
    const auto a = std::make_shared<ov::op::v0::Parameter>(u8, ov::Shape{1, 1, 64, 64});
    const auto b = std::make_shared<ov::op::v0::Parameter>(i8, ov::Shape{1, 1, 64, 64});
    const brgemm_utils::BrgemmConfig config(isa, u8, i8, i8, false, false);
    const auto repacked_b = std::make_shared<BrgemmCopyB>(b, config);
    const auto brgemm = std::make_shared<BrgemmCPU>(ov::OutputVector{a, repacked_b->output(0)}, config);
    EXPECT_EQ(jit_brgemm_emitter::get_supported_precisions(brgemm), expected_precisions);
}

INSTANTIATE_TEST_SUITE_P(
    smoke_BrgemmSupportedPrecisions,
    BrgemmSupportedPrecisions,
    ::testing::Values(IsaSupportedPrecisions{avx512_core_vnni, {{f32, f32}, {u8, i8}}},
                      IsaSupportedPrecisions{avx2_vnni, {{f32, f32}, {u8, i8}}},
                      // Pins the gating too: without VNNI the u8 A operand must not be offered.
                      IsaSupportedPrecisions{avx512_core, {{f32, f32}}},
                      IsaSupportedPrecisions{avx2_vnni_2, {{f32, f32}, {bf16, bf16}, {f16, f16}, {i8, i8}, {u8, i8}}}),
    BrgemmSupportedPrecisions::getTestCaseName);

}  // namespace
