// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/gated_residual.hpp"

#include <utility>
#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace ov::test {

namespace {

using Mode = ov::op::internal::GatedResidual::Mode;

const GatedResidualWeightCompression kNoCompression{};

// Every case runs with a dynamic input shape: rank-2 [rows, C*D] as in the real IR, so C*D is fixed per case and only
// the row count changes. `second` lists the row counts of the successive inferences through the one compiled model,
// and one sequence covers everything the executor does with a changing M:
//   - small M buckets (2, 4, 8, 16) and their boundaries (1/2, 3/4, 5/8, 9/16, 17), where the bucketed GEMMs and
//     their SiLU/sigmoid/write-gate epilogues run over padded rows;
//   - M buckets in steps of 16 below 512 and two row counts sharing a bucket (100 and 110 -> 112);
//   - the 512-row tile cap: exactly 512 and 1024 (no remainder), 511 (just below), 513 (remainder of one row), and
//     several tiles with a bucketed remainder (1100, 2051);
//   - prefill -> decode flips (1100 -> 1, 2051 -> 1 -> 1), repeated counts and a tiny chunk, so update() runs with
//     changed, unchanged and revisited GEMM shapes.
const ov::test::InputShape kDynRows{{-1}, {{1},    {2},   {3},   {4},    {5},   {8},   {9},   {16},   {17},
                                           {31},   {100}, {110}, {300},  {511}, {512}, {513}, {1100}, {1024},
                                           {2051}, {1},   {1},   {1100}, {1},   {64},  {1}}};

// Shapes for the rank-3 coverage below (f32 and a converting precision).
const std::vector<GatedResidualShapeParams> batch_seq_shapes{
    {4, 8, 16, ov::element::f32, kNoCompression},
    {4, 8, 16, ov::element::bf16, kNoCompression},
};

// The op accepts any number of leading batch dims; this is the only place with rank 3 [B, T, C*D], to keep the
// flattening of leading dims covered (rank-2 is what the real IR feeds).
const ov::test::InputShape kDynBatchSeq{{-1, -1}, {{2, 3}, {4, 1}, {3, 7}, {2, 100}, {1, 1}}};

// Production shape from the Qwen3.8-Flash-Next IR (openvino_language_model_i4.xml): hc_count=4,
// hidden_size=2560 (so C*D = 10240), lowrank=320; 96 READ (with write gate), 96 WRITE and 1 FINAL_MIX
// layers. Rank-2 dynamic rows, kept short because the reference GEMMs are scalar.
// Prefill-sized chunk first, then decode (1 row), a short chunk, and decode again.
const ov::test::InputShape kDynQwenRows{{-1}, {{300}, {1}, {33}, {1}}};
const std::vector<GatedResidualShapeParams> qwen_shapes{
    {4, 2560, 320, ov::element::f32, kNoCompression},
    {4, 2560, 320, ov::element::bf16, kNoCompression},
};

// Uncompressed shapes: power-of-two and non-power-of-two C/D/L (exercising any SIMD-tail paths) at
// every supported data precision.
const std::vector<GatedResidualShapeParams> uncompressed_shapes{
    {4, 8, 16, ov::element::f32, kNoCompression},
    {4, 8, 16, ov::element::bf16, kNoCompression},
    {4, 8, 16, ov::element::f16, kNoCompression},
    {3, 5, 7, ov::element::f32, kNoCompression},
    {3, 5, 7, ov::element::bf16, kNoCompression},
    // D = 1: every branch is a single element, so the RMS kernel's scalar-scale path reads the gain at the port
    // precision and applies norm_gain_plus_one.
    {4, 1, 8, ov::element::f32, kNoCompression},
    {4, 1, 8, ov::element::bf16, kNoCompression},
    {4, 1, 8, ov::element::f16, kNoCompression},
};

// Compressed-weight shapes: u8/i8/u4/i4, per-channel and grouped, with and without zero points. Scales are [G, N]
// floating-point tensors (f16 / bf16 / f32) and zero points have the weights' type. The CPU node requires a scale to
// have a static group count G that divides K (the op validates it): 1 (per-channel), 2, 3, 4, ... CD/L below are sized
// per case so that the group size K / G is 16..128.
using Compression = GatedResidualWeightCompression;
const Compression u8_per_channel{true, ov::element::u8, false, 0, ov::element::f32};
const Compression u8_grouped_zp{true, ov::element::u8, true, 16, ov::element::f16};
const Compression u8_grouped_nozp{true, ov::element::u8, false, 16, ov::element::bf16};
const Compression i8_grouped_nozp{true, ov::element::i8, false, 16, ov::element::f16};
const Compression i4_per_channel{true, ov::element::i4, false, 0, ov::element::bf16};
const Compression i4_grouped_zp{true, ov::element::i4, true, 16, ov::element::f32};
const Compression i4_grouped_nozp{true, ov::element::i4, false, 16, ov::element::f16};  // symmetric i4 (no zero point)
const Compression u4_grouped_zp{true, ov::element::u4, true, 16, ov::element::f16};     // asymmetric u4 (zero point)
const Compression u4_grouped_zp_bf16{true, ov::element::u4, true, 16, ov::element::bf16};
const Compression u4_per_channel_zp{true, ov::element::u4, true, 0, ov::element::f32};
const Compression u4_grouped_zp_gs32{true, ov::element::u4, true, 32, ov::element::f16};   // CD = L = 64 -> G = 2
const Compression u4_grouped_zp_gs16{true, ov::element::u4, true, 16, ov::element::bf16};  // CD = L = 48 -> G = 3
const Compression u8_grouped_zp_gs32{true, ov::element::u8, true, 32, ov::element::f16};
const Compression u8_grouped_zp_gs64{true, ov::element::u8, true, 64, ov::element::f16};
const Compression u8_grouped_zp_gs128{true, ov::element::u8, true, 128, ov::element::f16};
const Compression u4_grouped_zp_gs128{true,
                                      ov::element::u4,
                                      true,
                                      128,
                                      ov::element::f16};  // the real model's group size

// All compressed shapes except i4_per_channel, instantiated with produce_write_gate=true to also
// exercise the inject projection.
const std::vector<GatedResidualShapeParams> compressed_shapes_with_inject{
    {4, 8, 16, ov::element::f32, u8_per_channel},
    {4, 16, 64, ov::element::f16, u8_grouped_zp},
    {4, 16, 64, ov::element::f16, u8_grouped_nozp},
    {4, 16, 64, ov::element::f16, i8_grouped_nozp},
    {4, 16, 64, ov::element::f16, i4_grouped_zp},
    {4, 16, 64, ov::element::f16, i4_grouped_nozp},
    {4, 16, 64, ov::element::f16, u4_grouped_zp},
    {4, 16, 64, ov::element::bf16, u4_grouped_zp_bf16},
    {4, 8, 16, ov::element::f32, u4_per_channel_zp},
    {4, 16, 64, ov::element::f16, u4_grouped_zp_gs32},     // G = 2 for w_down/w_inject (K = 64) and w_up (K = 64)
    {4, 12, 48, ov::element::f16, u4_grouped_zp_gs16},     // G = 3 for all three projections (K = 48)
    {4, 32, 128, ov::element::f16, u8_grouped_zp_gs32},    // GS=32: CD=L=128 (4 groups each)
    {4, 64, 256, ov::element::f16, u8_grouped_zp_gs64},    // GS=64: CD=L=256 (4 groups each)
    {4, 128, 512, ov::element::f16, u8_grouped_zp_gs128},  // GS=128: CD=L=512 (4 groups each)
    {4, 128, 512, ov::element::f16, u4_grouped_zp_gs128},  // GS=128 u4 + zero point
};

// Dedicated case covering the no-write-gate (no inject projection) path under compression.
const std::vector<GatedResidualShapeParams> compressed_shapes_no_inject{
    {4, 8, 16, ov::element::bf16, i4_per_channel},
};

// Every compressed shape above (both subsets), reused for Mode::FINAL_MIX, which always forces
// produce_write_gate off -- so i4_per_channel's "no inject" shape fits here naturally too.
std::vector<GatedResidualShapeParams> all_compressed_shapes() {
    std::vector<GatedResidualShapeParams> shapes = compressed_shapes_with_inject;
    shapes.insert(shapes.end(), compressed_shapes_no_inject.begin(), compressed_shapes_no_inject.end());
    return shapes;
}

}  // namespace

// One suite per mode: produce_write_gate is only valid for READ, and WRITE has no weights/RMSNorm
// gain, so a free Combine across Mode would generate cases the op rejects or that are redundant.
INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualRead,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::READ),
                                            ::testing::ValuesIn(uncompressed_shapes),
                                            ::testing::Values(false, true),  // produce_write_gate
                                            ::testing::Values(false, true),  // norm_gain_plus_one
                                            ::testing::Values(kDynRows),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualFinalMix,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::FINAL_MIX),
                                            ::testing::ValuesIn(uncompressed_shapes),
                                            ::testing::Values(false),        // produce_write_gate
                                            ::testing::Values(false, true),  // norm_gain_plus_one
                                            ::testing::Values(kDynRows),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualWrite,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::WRITE),
                                            ::testing::ValuesIn(uncompressed_shapes),
                                            ::testing::Values(false),  // produce_write_gate
                                            ::testing::Values(true),   // norm_gain_plus_one
                                            ::testing::Values(kDynRows),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

// Rank-3 [B, T, C*D] inputs: READ (mix and write gate) and WRITE each flatten the leading dims on their own.
INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualBatchSeqRead,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::READ),
                                            ::testing::ValuesIn(batch_seq_shapes),
                                            ::testing::Values(true),  // produce_write_gate
                                            ::testing::Values(true),  // norm_gain_plus_one
                                            ::testing::Values(kDynBatchSeq),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualBatchSeqWrite,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::WRITE),
                                            ::testing::ValuesIn(batch_seq_shapes),
                                            ::testing::Values(false),  // produce_write_gate
                                            ::testing::Values(true),   // norm_gain_plus_one
                                            ::testing::Values(kDynBatchSeq),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualQwenRead,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::READ),
                                            ::testing::ValuesIn(qwen_shapes),
                                            ::testing::Values(true),  // produce_write_gate
                                            ::testing::Values(true),  // norm_gain_plus_one
                                            ::testing::Values(kDynQwenRows),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualQwenWrite,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::WRITE),
                                            ::testing::ValuesIn(qwen_shapes),
                                            ::testing::Values(false),  // produce_write_gate
                                            ::testing::Values(false),  // norm_gain_plus_one
                                            ::testing::Values(kDynQwenRows),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualQwenFinalMix,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::FINAL_MIX),
                                            ::testing::ValuesIn(qwen_shapes),
                                            ::testing::Values(false),  // produce_write_gate
                                            ::testing::Values(true),   // norm_gain_plus_one
                                            ::testing::Values(kDynQwenRows),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

// WRITE is excluded from all three compressed suites below: it has no weight projections at all (its three inputs
// are residual, sub-layer output and write gate). Splitting READ and
// FINAL_MIX into separate Combine instantiations (rather than combining Mode freely against
// produce_write_gate) avoids generating the invalid Mode::FINAL_MIX + produce_write_gate=true
// combination the op also rejects -- execute_read_or_final_mix() never branches on mode, but
// covering both here means a future mode-dispatch regression can't hide behind "only READ was
// tested".
INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualCompressed,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::READ),
                                            ::testing::ValuesIn(compressed_shapes_with_inject),
                                            ::testing::Values(true),  // produce_write_gate
                                            ::testing::Values(true),  // norm_gain_plus_one
                                            ::testing::Values(kDynRows),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualCompressedNoWriteGate,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::READ),
                                            ::testing::ValuesIn(compressed_shapes_no_inject),
                                            ::testing::Values(false),  // produce_write_gate
                                            ::testing::Values(true),   // norm_gain_plus_one
                                            ::testing::Values(kDynRows),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualCompressedFinalMix,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::FINAL_MIX),
                                            ::testing::ValuesIn(all_compressed_shapes()),
                                            ::testing::Values(false),  // produce_write_gate
                                            ::testing::Values(true),   // norm_gain_plus_one
                                            ::testing::Values(kDynRows),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

}  // namespace ov::test
