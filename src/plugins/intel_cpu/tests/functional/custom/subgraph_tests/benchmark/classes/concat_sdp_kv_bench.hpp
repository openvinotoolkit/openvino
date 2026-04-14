// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

// KV cache decode benchmark.
// Builds a stateful ReadValue/Gather/Concat/SDPA/Assign subgraph with L1=1
// (single-token decode).  Follows the standard SubgraphBaseTest lifecycle so
// it can be wrapped by BenchmarkLayerTest for automated warmup + timing.
//
// Each inferRequest.infer() call appends one token to the KV cache state.
// BenchmarkLayerTest's warmup loop fills the cache to a realistic length,
// then the benchmark iterations measure steady-state decode throughput.
//
// kCacheMode / vCacheMode select quantization per-cache:
//   "none" — f32 baseline (no quantization)
//   "u8"   — uniform asymmetric u8 with scale+zp
//   "tbq4" — TurboQuant 4-bit (Lloyd-Max codebook, random rotation)
//   "tbq3" — TurboQuant 3-bit
//   "polar4" / "polar3" — PolarQuant
//
// groupSize: quantization group size for u8 (0 = default = head_dim).
//   Only affects u8 mode. Codec modes ignore this.

typedef std::tuple<ov::AnyMap,             // additional plugin config (e.g. inference_precision)
                   std::vector<InputShape>,
                   std::string,   // kCacheMode
                   std::string,   // vCacheMode
                   size_t,        // groupSize (0 = default, for u8)
                   std::string>   // rotationMode: "wht", "dense"
    ConcatSDPKVBenchParams;

class ConcatSDPKVBenchBase : public testing::WithParamInterface<ConcatSDPKVBenchParams>,
                             virtual public ov::test::SubgraphBaseTest,
                             public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatSDPKVBenchParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

}  // namespace test
}  // namespace ov
