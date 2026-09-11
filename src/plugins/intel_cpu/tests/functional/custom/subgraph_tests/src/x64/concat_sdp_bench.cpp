// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Opt-in wall-clock benchmark comparing KV cache codecs (none / u8 / u4 / TBQ4 / OSCAR).
// Disabled by default; run with:
//   ./ov_cpu_func_tests --gtest_also_run_disabled_tests \
//     --gtest_filter=*ConcatSDPBench*
//
// Cannot reuse BenchmarkLayerTest<T> — it re-creates the infer request per infer(),
// which drops KV state; and ConcatSDPTest::run() runs a dual-model compare path.
// @todo claude: promote thresholds/warmup/attempts to CLI once useful.

#include <chrono>
#include <iomanip>
#include <iostream>

#include "custom/subgraph_tests/src/classes/concat_sdp.hpp"

namespace ov {
namespace test {
namespace {

class ConcatSDPBenchmarkTest : public ConcatSDPTest {
protected:
    void run() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        compiledModel = core->compile_model(function, targetDevice, configuration);
        auto req = compiledModel.create_infer_request();

        auto bind_inputs = [&](const std::vector<ov::Shape>& shapes) {
            generate_inputs(shapes);
            for (const auto& port : compiledModel.inputs()) {
                const auto& name = port.get_node()->get_friendly_name();
                for (const auto& [node, tensor] : inputs) {
                    if (node->get_friendly_name() == name) {
                        req.set_tensor(port, tensor);
                        break;
                    }
                }
            }
        };

        constexpr int kWarmupIters = 2;
        constexpr int kMeasureIters = 20;

        for (int w = 0; w < kWarmupIters; ++w) {
            m_iter = 0;
            m_accum_L_q = 0;
            for (const auto& shapes : targetStaticShapes) {
                bind_inputs(shapes);
                req.infer();
            }
        }

        using clk = std::chrono::steady_clock;
        uint64_t total_ns = 0;
        for (int i = 0; i < kMeasureIters; ++i) {
            m_iter = 0;
            m_accum_L_q = 0;
            const auto t0 = clk::now();
            for (const auto& shapes : targetStaticShapes) {
                bind_inputs(shapes);
                req.infer();
            }
            total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count();
        }
        const double avg_us = static_cast<double>(total_ns) / kMeasureIters / 1000.0;
        std::cout << "[bench] " << ::testing::UnitTest::GetInstance()->current_test_info()->name()
                  << " avg=" << std::fixed << std::setprecision(2) << avg_us << " us/sequence\n";
    }
};

TEST_P(ConcatSDPBenchmarkTest, DISABLED_Benchmark) {
    run();
}

// Long-prompt shape: 5×64 tokens = 320 total, crosses two OSCAR R=128 blocks.
const std::vector<std::vector<InputShape>> benchShapes = {
    {
        {{1, 8, -1, 64}, {{1, 8, 64, 64}, {1, 8, 64, 64}, {1, 8, 64, 64}, {1, 8, 64, 64}, {1, 8, 64, 64}}},
        {{1, 8, -1, 64}, {{1, 8, 0, 64}, {1, 8, 64, 64}, {1, 8, 128, 64}, {1, 8, 192, 64}, {1, 8, 256, 64}}},
    },
};

enum class KvCodec { NONE, U8, U4, TBQ4, OSCAR };
static ov::AnyMap kv_cfg(KvCodec c) {
    ov::AnyMap m;
    auto apply = [&](const char* prec, const char* alg) {
        switch (c) {
            case KvCodec::NONE:  break;
            case KvCodec::U8:    m[prec] = "u8"; break;
            case KvCodec::U4:    m[prec] = "u4"; break;
            case KvCodec::TBQ4:  m[prec] = "u4"; m[alg] = "TURBO"; break;
            case KvCodec::OSCAR: m[prec] = "u2"; m[alg] = "OSCAR"; break;
        }
    };
    apply("KEY_CACHE_PRECISION", "KEY_CACHE_QUANT_ALG");
    apply("VALUE_CACHE_PRECISION", "VALUE_CACHE_QUANT_ALG");
    return m;
}

INSTANTIATE_TEST_SUITE_P(ConcatSDPBench,
                         ConcatSDPBenchmarkTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(benchShapes),
                                            ::testing::Values(kv_cfg(KvCodec::NONE),
                                                              kv_cfg(KvCodec::U8),
                                                              kv_cfg(KvCodec::U4),
                                                              kv_cfg(KvCodec::TBQ4),
                                                              kv_cfg(KvCodec::OSCAR)),
                                            ::testing::Values(false),
                                            ::testing::Values<int64_t>(8),
                                            ::testing::Values<int64_t>(8)),
                         ConcatSDPTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
