// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <vector>

#include "emitters/snippets/x64/jit_brgemm_emitter.hpp"
#include "openvino/op/parameter.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

using namespace ov::intel_cpu;

namespace {
// The ISA-taking BrgemmConfig constructor bypasses get_prim_isa(), which prefers amx and
// avx512_core_vnni over avx2_vnni_2 for int8, so the avx2_vnni_2 precision set is otherwise
// unreachable on a CI machine that has avx512.
std::set<std::vector<ov::element::Type>> u8i8_supported_precisions(dnnl::impl::cpu::x64::cpu_isa_t isa) {
    const auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{1, 1, 64, 64});
    const auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{1, 1, 64, 64});
    const brgemm_utils::BrgemmConfig config(isa, ov::element::u8, ov::element::i8, ov::element::i8, false, false);
    const auto repacked_b = std::make_shared<BrgemmCopyB>(b, config);
    const auto brgemm = std::make_shared<BrgemmCPU>(ov::OutputVector{a, repacked_b->output(0)}, config);
    return jit_brgemm_emitter::get_supported_precisions(brgemm);
}
}  // namespace

TEST(BrgemmSupportedPrecisions, U8AOperandGatingByIsa) {
    const std::set<std::vector<ov::element::Type>> f32_and_u8i8{{ov::element::f32, ov::element::f32},
                                                                {ov::element::u8, ov::element::i8}};
    EXPECT_EQ(u8i8_supported_precisions(dnnl::impl::cpu::x64::avx512_core_vnni), f32_and_u8i8);
    EXPECT_EQ(u8i8_supported_precisions(dnnl::impl::cpu::x64::avx2_vnni), f32_and_u8i8);
    // Pins the gating too: without VNNI the u8 A operand must not be offered.
    EXPECT_EQ(u8i8_supported_precisions(dnnl::impl::cpu::x64::avx512_core),
              (std::set<std::vector<ov::element::Type>>{{ov::element::f32, ov::element::f32}}));
    EXPECT_EQ(u8i8_supported_precisions(dnnl::impl::cpu::x64::avx2_vnni_2),
              (std::set<std::vector<ov::element::Type>>{{ov::element::f32, ov::element::f32},
                                                        {ov::element::bf16, ov::element::bf16},
                                                        {ov::element::f16, ov::element::f16},
                                                        {ov::element::i8, ov::element::i8},
                                                        {ov::element::u8, ov::element::i8}}));
}
