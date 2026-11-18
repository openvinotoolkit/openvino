// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/gated_residual.hpp"

#include <utility>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace ov::test {

namespace {

using Mode = ov::op::internal::GatedResidual::Mode;

const GatedResidualWeightCompression kNoCompression{};

// Dynamic rows [rows, C*D]; `second` lists the row counts of successive inferences through one compiled model. Every
// case here runs a 512-row tile, so the partial tile only hits the <= 16 and round-up-to-16 buckets: 1 and 17 cover
// them, 100 a plain bucket, 513 and 1100 the tile cap with a remainder, and the repeated 1 a prefill -> decode flip.
const ov::test::InputShape kDynRows{{-1}, {{1}, {17}, {100}, {513}, {1100}, {1}}};

// The compressed cases need fewer row counts: the same buckets, without the plain one.
const ov::test::InputShape kDynCompressedRows{{-1}, {{1}, {17}, {513}, {1}}};

// Rank-3 [B, T, C*D] activations (rank-2 [rows, C*D] is the usual form of the other cases), covering the flattening of
// leading dims.
const ov::test::InputShape kDynBatchSeq{{-1, -1}, {{2, 3}, {4, 1}, {3, 7}, {2, 100}, {1, 1}}};

// Large-model shape (C=4, D=2560, L=320), rank-2 dynamic rows, kept short because the reference GEMMs are scalar. A
// prefill-sized chunk first, then decode.
const ov::test::InputShape kDynQwenRows{{-1}, {{33}, {1}, {1}}};

// The model is an f32 or f16 IR; bf16 is an inference precision (hint) that exists only on supporting hardware. f16
// inference is not covered: ConvertPrecision keeps the precision-sensitive norm in f32 there, which the pattern does
// not match, so the layer stays decomposed.
std::vector<ov::AnyMap> inference_precision_configs() {
    std::vector<ov::AnyMap> configs{{{ov::hint::inference_precision.name(), ov::element::f32}}};
    if (ov::with_cpu_x86_bfloat16()) {
        configs.push_back({{ov::hint::inference_precision.name(), ov::element::bf16}});
    }
    return configs;
}

std::vector<ov::AnyMap> bf16_inference_configs() {
    std::vector<ov::AnyMap> configs;
    if (ov::with_cpu_x86_bfloat16()) {
        configs.push_back({{ov::hint::inference_precision.name(), ov::element::bf16}});
    }
    return configs;
}

// Uncompressed shapes: power-of-two and other C/D/L (SIMD-tail paths) for f32 and f16 IRs. D = 1 makes every branch one
// element, which takes the RMS kernel's scalar-scale path.
const std::vector<GatedResidualShapeParams> uncompressed_shapes{
    {4, 8, 16, ov::element::f32, kNoCompression},
    {4, 8, 16, ov::element::f16, kNoCompression},
    {3, 5, 7, ov::element::f32, kNoCompression},
    {4, 1, 8, ov::element::f32, kNoCompression},
    {4, 1, 8, ov::element::f16, kNoCompression},
};

const std::vector<GatedResidualShapeParams> batch_seq_shapes{{4, 8, 16, ov::element::f32, kNoCompression}};

const std::vector<GatedResidualShapeParams> qwen_shapes{{4, 2560, 320, ov::element::f32, kNoCompression}};

// Compressed shapes: u8/i8/u4/i4, per-channel and grouped, with and without zero points; scales are f16/bf16/f32 [N,
// G], zero points have the weight type. G must be static and divide K; CD/L are sized so the group size K / G is
// 16..128.
using Compression = GatedResidualWeightCompression;
const Compression u8_per_channel{true, ov::element::u8, false, 0, ov::element::f32};
const Compression u8_grouped_zp{true, ov::element::u8, true, 16, ov::element::f16};
const Compression u8_grouped_nozp{true, ov::element::u8, false, 16, ov::element::bf16};
const Compression i8_grouped_nozp{true, ov::element::i8, false, 16, ov::element::f16};
const Compression i4_per_channel{true, ov::element::i4, false, 0, ov::element::bf16};
const Compression i4_grouped_zp{true, ov::element::i4, true, 16, ov::element::f32};
const Compression i4_grouped_nozp{true, ov::element::i4, false, 16, ov::element::f16};  // symmetric i4 (no zero point)
const Compression u4_grouped_zp{true, ov::element::u4, true, 16, ov::element::f16};     // asymmetric u4 (zero point)
const Compression u4_per_channel_zp{true, ov::element::u4, true, 0, ov::element::f32};
const Compression u4_grouped_zp_gs32{true, ov::element::u4, true, 32, ov::element::f16};   // CD = L = 64 -> G = 2
const Compression u4_grouped_zp_gs16{true, ov::element::u4, true, 16, ov::element::bf16};  // CD = L = 48 -> G = 3
const Compression u8_grouped_zp_gs128{true, ov::element::u8, true, 128, ov::element::f16};
const Compression u4_grouped_zp_gs128{true, ov::element::u4, true, 128, ov::element::f16};

// Compressed shapes with produce_write_gate=true, which also covers the inject projection.
const std::vector<GatedResidualShapeParams> compressed_shapes_with_inject{
    {4, 8, 16, ov::element::f32, u8_per_channel},
    {4, 16, 64, ov::element::f16, u8_grouped_zp},
    {4, 16, 64, ov::element::f16, u8_grouped_nozp},
    {4, 16, 64, ov::element::f16, i8_grouped_nozp},
    {4, 16, 64, ov::element::f16, i4_grouped_zp},
    {4, 16, 64, ov::element::f16, i4_grouped_nozp},
    {4, 16, 64, ov::element::f16, u4_grouped_zp},
    {4, 8, 16, ov::element::f32, u4_per_channel_zp},
    {4, 16, 64, ov::element::f16, u4_grouped_zp_gs32},  // G = 2 for w_down/w_inject (K = 64) and w_up (K = 64)
    {4, 12, 48, ov::element::f16, u4_grouped_zp_gs16},  // G = 3 for all three projections (K = 48)
};

// Group size 128 with C * D = L = 256 (G = 2). Dynamic quantization (on by default) applies to these weights, so they
// also run with it disabled to isolate the decompression numerics; the group-size-16 weights never use it.
const std::vector<GatedResidualShapeParams> compressed_shapes_group128{
    {4, 64, 256, ov::element::f16, u8_grouped_zp_gs128},
    {4, 64, 256, ov::element::f16, u4_grouped_zp_gs128},
};
const std::vector<ov::AnyMap> group128_configs{{}, {{ov::hint::dynamic_quantization_group_size.name(), uint64_t{0}}}};

// Without the write gate (no inject projection), per-channel and grouped.
const std::vector<GatedResidualShapeParams> compressed_shapes_no_inject{
    {4, 8, 16, ov::element::f32, i4_per_channel},
    {4, 16, 64, ov::element::f16, i4_grouped_zp},
};

// bf16 inference of compressed weights, where the hardware supports it.
const std::vector<GatedResidualShapeParams> compressed_shapes_bf16{
    {4, 16, 64, ov::element::f16, u4_grouped_zp},
    {4, 8, 16, ov::element::f32, i4_per_channel},
};

}  // namespace

// Only READ is instantiated per input form: the fusion emits READ, which also covers the FINAL_MIX math without the
// gate. WRITE has no weights or gain: the plugin runs it as one Eltwise, and these cases check that path.
INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualRead,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::READ),
                                            ::testing::ValuesIn(uncompressed_shapes),
                                            ::testing::Values(false, true),  // produce_write_gate
                                            ::testing::Values(true),         // norm_gain_plus_one
                                            ::testing::Values(kDynRows),
                                            ::testing::ValuesIn(inference_precision_configs()),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualWrite,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::WRITE),
                                            ::testing::ValuesIn(uncompressed_shapes),
                                            ::testing::Values(false),  // produce_write_gate
                                            ::testing::Values(true),   // norm_gain_plus_one
                                            ::testing::Values(kDynRows),
                                            ::testing::ValuesIn(inference_precision_configs()),
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
                                            ::testing::ValuesIn(inference_precision_configs()),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualBatchSeqWrite,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::WRITE),
                                            ::testing::ValuesIn(batch_seq_shapes),
                                            ::testing::Values(false),  // produce_write_gate
                                            ::testing::Values(true),   // norm_gain_plus_one
                                            ::testing::Values(kDynBatchSeq),
                                            ::testing::ValuesIn(inference_precision_configs()),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualQwenRead,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::READ),
                                            ::testing::ValuesIn(qwen_shapes),
                                            ::testing::Values(true),  // produce_write_gate
                                            ::testing::Values(true),  // norm_gain_plus_one
                                            ::testing::Values(kDynQwenRows),
                                            ::testing::ValuesIn(inference_precision_configs()),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualCompressed,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::READ),
                                            ::testing::ValuesIn(compressed_shapes_with_inject),
                                            ::testing::Values(true),  // produce_write_gate
                                            ::testing::Values(true),  // norm_gain_plus_one
                                            ::testing::Values(kDynCompressedRows),
                                            ::testing::Values(ov::AnyMap{}),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualCompressedGroup128,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::READ),
                                            ::testing::ValuesIn(compressed_shapes_group128),
                                            ::testing::Values(true),  // produce_write_gate
                                            ::testing::Values(true),  // norm_gain_plus_one
                                            ::testing::Values(kDynCompressedRows),
                                            ::testing::ValuesIn(group128_configs),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualCompressedNoWriteGate,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::READ),
                                            ::testing::ValuesIn(compressed_shapes_no_inject),
                                            ::testing::Values(false),  // produce_write_gate
                                            ::testing::Values(true),   // norm_gain_plus_one
                                            ::testing::Values(kDynCompressedRows),
                                            ::testing::Values(ov::AnyMap{}),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatedResidualCompressedBf16,
                         GatedResidualLayerCPUTest,
                         ::testing::Combine(::testing::Values(Mode::READ),
                                            ::testing::ValuesIn(compressed_shapes_bf16),
                                            ::testing::Values(true),  // produce_write_gate
                                            ::testing::Values(true),  // norm_gain_plus_one
                                            ::testing::Values(kDynCompressedRows),
                                            ::testing::ValuesIn(bf16_inference_configs()),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GatedResidualLayerCPUTest::getTestCaseName);

}  // namespace ov::test
