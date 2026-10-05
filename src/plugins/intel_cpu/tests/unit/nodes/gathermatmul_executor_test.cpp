// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "cpu_memory.h"
#include "cpu_shape.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/executors/dnnl/dnnl_gathermatmul_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "weights_cache.hpp"

namespace ov::intel_cpu {

// Reads the executor state the tests assert on: which GEMM heights were built for the current input.
struct GatherMatmulDnnlExecutorTestPeer {
    static std::vector<size_t> gemmRowSizes(const GatherMatmulDnnlExecutor& executor) {
        std::vector<size_t> sizes;
        for (const auto& [rows, impl] : executor.m_gemmBySize) {
            if (impl) {
                sizes.push_back(rows);
            }
        }
        std::sort(sizes.begin(), sizes.end());
        return sizes;
    }
    static bool bf16AmxMode(const GatherMatmulDnnlExecutor& executor) {
        return executor.m_bf16AmxMode;
    }
};

}  // namespace ov::intel_cpu

using namespace ov::intel_cpu;

// ---- Row rounding arithmetic: no hardware dependency ------------------------------------------

using RowRoundingCase = std::tuple<size_t, size_t>;  // rows, expected normalizeM(rows)

class GatherMatmulRowRoundingTest : public testing::TestWithParam<RowRoundingCase> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RowRoundingCase>& obj) {
        const auto& [rows, expected] = obj.param;
        return "rows_" + std::to_string(rows) + "_to_" + std::to_string(expected);
    }
};

TEST_P(GatherMatmulRowRoundingTest, RoundsUpToTheStepOfItsBucket) {
    const auto& [rows, expected] = GetParam();
    EXPECT_EQ(gathermatmul::normalizeM(rows), expected);
}

// The step is 16 below 512 rows, 32 below 1024 and 256 from 1024 on. The cases sit on both sides of each
// bucket boundary and on exact multiples of each step.
INSTANTIATE_TEST_SUITE_P(smoke_GatherMatmulRowRounding,
                         GatherMatmulRowRoundingTest,
                         testing::Values(RowRoundingCase{1, 16},
                                         RowRoundingCase{15, 16},
                                         RowRoundingCase{16, 16},
                                         RowRoundingCase{17, 32},
                                         RowRoundingCase{496, 496},
                                         RowRoundingCase{497, 512},
                                         RowRoundingCase{511, 512},
                                         RowRoundingCase{512, 512},
                                         RowRoundingCase{513, 544},
                                         RowRoundingCase{544, 544},
                                         RowRoundingCase{992, 992},
                                         RowRoundingCase{993, 1024},
                                         RowRoundingCase{1023, 1024},
                                         RowRoundingCase{1024, 1024},
                                         RowRoundingCase{1025, 1280},
                                         RowRoundingCase{1280, 1280},
                                         RowRoundingCase{2048, 2048},
                                         RowRoundingCase{2049, 2304}),
                         GatherMatmulRowRoundingTest::getTestCaseName);

// A gather index that received num_valid_rows of an M-row input runs a GEMM of normalizeM(num_valid_rows)
// rows. normalizeM is monotonic and num_valid_rows <= M, so that never exceeds normalizeM(M), the height of
// the temporary buffer: the std::min guard in gemmRows() can never clamp.
TEST(GatherMatmulRowRoundingTest, GemmRowsNeverExceedTheBufferHeight) {
    for (const size_t M : {1U, 2U, 15U, 16U, 17U, 100U, 511U, 512U, 513U, 1000U, 1023U, 1024U, 1025U, 2048U, 2300U}) {
        const size_t M_size = gathermatmul::normalizeM(M);
        for (size_t rows = 1; rows <= M; ++rows) {
            const size_t gemm_rows = gathermatmul::gemmRows(rows, M_size);
            ASSERT_EQ(gemm_rows, gathermatmul::normalizeM(rows)) << "M=" << M << " rows=" << rows;
            ASSERT_LE(gemm_rows, M_size) << "M=" << M << " rows=" << rows;
        }
    }
}

// ---- The executor with hand-written routing: bf16 on AMX ---------------------------------------

namespace {

// src [topk, M, K], indices [M, topk] (i32), weights [E, N, K], dst [topk, M, N]
struct Problem {
    size_t topk;
    size_t M;
    size_t K;
    size_t N;
    size_t E;
    // routing[m][t] = gather index (expert) for row m, slot t
    std::vector<std::vector<int32_t>> routing;
    // u8 weights with one f32 scale per output channel (the compressed form the plugin serves int4/int8 MoE
    // models in), instead of plain bf16 weights
    bool compressedWeights = false;
};

constexpr float kWeightScale = 1.0F / 16.0F;

MemoryPtr makeMemory(const dnnl::engine& engine, ov::element::Type precision, const VectorDims& dims) {
    return std::make_shared<Memory>(engine, CpuBlockedMemoryDesc(precision, Shape(dims)));
}

float srcValue(size_t t, size_t m, size_t k) {
    return static_cast<float>(static_cast<int>((m * 7 + t * 3 + k) % 11) - 5) / 8.0F;  // exact in bf16
}

uint8_t weightCode(size_t e, size_t n, size_t k) {
    return static_cast<uint8_t>((e * 5 + n * 3 + k) % 13);
}

// The real weight the executor multiplies with: bf16 directly, or u8 code * per-channel scale.
float weightValue(const Problem& p, size_t e, size_t n, size_t k) {
    if (p.compressedWeights) {
        return static_cast<float>(weightCode(e, n, k)) * kWeightScale;
    }
    return static_cast<float>(static_cast<int>(weightCode(e, n, k)) - 6) / 16.0F;  // exact in bf16
}

struct Bound {
    MemoryArgs memory;
    MemoryPtr dst;
};

Bound bindProblem(const Problem& p, const GraphContext::CPtr& graphContext) {
    const auto& engine = graphContext->getEngine();
    auto src = makeMemory(engine, ov::element::bf16, {p.topk, p.M, p.K});
    auto indices = makeMemory(engine, ov::element::i32, {p.M, p.topk});
    auto weights = makeMemory(engine, p.compressedWeights ? ov::element::u8 : ov::element::bf16, {p.E, p.N, p.K});
    auto dst = makeMemory(engine, ov::element::bf16, {p.topk, p.M, p.N});
    MemoryPtr scales = MemoryDescUtils::makeEmptyMemory(graphContext);

    auto* srcData = src->getDataAs<ov::bfloat16>();
    for (size_t t = 0; t < p.topk; ++t) {
        for (size_t m = 0; m < p.M; ++m) {
            for (size_t k = 0; k < p.K; ++k) {
                srcData[(t * p.M + m) * p.K + k] = ov::bfloat16(srcValue(t, m, k));
            }
        }
    }
    auto* indexData = indices->getDataAs<int32_t>();
    for (size_t m = 0; m < p.M; ++m) {
        for (size_t t = 0; t < p.topk; ++t) {
            indexData[m * p.topk + t] = p.routing[m][t];
        }
    }
    if (p.compressedWeights) {
        auto* weightData = weights->getDataAs<uint8_t>();
        for (size_t e = 0; e < p.E; ++e) {
            for (size_t n = 0; n < p.N; ++n) {
                for (size_t k = 0; k < p.K; ++k) {
                    weightData[(e * p.N + n) * p.K + k] = weightCode(e, n, k);
                }
            }
        }
        scales = makeMemory(engine, ov::element::f32, {p.E, p.N, 1});
        std::fill_n(scales->getDataAs<float>(), p.E * p.N, kWeightScale);
    } else {
        auto* weightData = weights->getDataAs<ov::bfloat16>();
        for (size_t e = 0; e < p.E; ++e) {
            for (size_t n = 0; n < p.N; ++n) {
                for (size_t k = 0; k < p.K; ++k) {
                    weightData[(e * p.N + n) * p.K + k] = ov::bfloat16(weightValue(p, e, n, k));
                }
            }
        }
    }
    std::fill_n(dst->getDataAs<ov::bfloat16>(), p.topk * p.M * p.N, ov::bfloat16(0.0F));

    MemoryArgs memory;
    memory[ARG_SRC] = src;
    memory[ARG_SRC_1] = indices;
    memory[ARG_WEI] = weights;
    memory[ARG_BIAS] = MemoryDescUtils::makeEmptyMemory(graphContext);
    memory[ARG_SRC_3] = scales;
    memory[ARG_SRC_4] = MemoryDescUtils::makeEmptyMemory(graphContext);
    memory[ARG_DST] = dst;
    return {memory, dst};
}

// dst[t][m][n] = sum_k src[t][m][k] * weights[routing[m][t]][n][k]
void checkOutput(const Problem& p, const MemoryPtr& dst) {
    const auto* out = dst->getDataAs<ov::bfloat16>();
    for (size_t t = 0; t < p.topk; ++t) {
        for (size_t m = 0; m < p.M; ++m) {
            const auto e = static_cast<size_t>(p.routing[m][t]);
            for (size_t n = 0; n < p.N; ++n) {
                float ref = 0.0F;
                for (size_t k = 0; k < p.K; ++k) {
                    ref += srcValue(t, m, k) * weightValue(p, e, n, k);
                }
                const float got = static_cast<float>(out[(t * p.M + m) * p.N + n]);
                // bf16 output: 8 bits of mantissa
                const float tol = 0.01F * std::abs(ref) + 0.02F;
                ASSERT_NEAR(got, ref, tol) << "t=" << t << " m=" << m << " n=" << n << " expert=" << e;
            }
        }
    }
}

std::vector<size_t> expectedGemmRowSizes(const Problem& p) {
    std::vector<size_t> counts(p.E, 0);
    for (size_t m = 0; m < p.M; ++m) {
        for (size_t t = 0; t < p.topk; ++t) {
            counts[static_cast<size_t>(p.routing[m][t])]++;
        }
    }
    const size_t M_size = gathermatmul::normalizeM(p.M);
    std::vector<size_t> sizes{M_size};  // the full-length GEMM is always created in update()
    for (const size_t c : counts) {
        if (c > 0) {
            sizes.push_back(gathermatmul::gemmRows(c, M_size));
        }
    }
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    return sizes;
}

class GatherMatmulDnnlExecutorTest : public testing::Test {
protected:
    void SetUp() override {
        if (!ov::with_cpu_x86_avx512_core_amx()) {
            GTEST_SKIP() << "bf16 AMX mode of GatherMatmulDnnlExecutor needs AMX";
        }
        Config config;
        config.rtCacheCapacity = 100;
        graphContext = std::make_shared<GraphContext>(config, std::make_shared<WeightsSharing>(), false);
        context = std::make_shared<ExecutorContext>(graphContext,
                                                    std::vector<impl_desc_type>{},
                                                    std::make_shared<std::unordered_map<std::string, MemoryPtr>>());
    }

    GraphContext::CPtr graphContext;
    ExecutorContext::CPtr context;
};

}  // namespace

// Four experts with very different row counts in one input: one row, 86 rows, 513 rows (past the 512-row
// bucket boundary), every row (which is the full-length GEMM), and four experts with no rows at all.
// The executor has to build exactly one GEMM per distinct rounded height and produce the right output.
TEST_F(GatherMatmulDnnlExecutorTest, EachGatherIndexRunsAGemmOfItsOwnHeight) {
    Problem p{2, 600, 64, 32, 8, {}};
    p.routing.resize(p.M);
    for (size_t m = 0; m < p.M; ++m) {
        const int32_t slot0 = m < 513 ? 2 : (m == 599 ? 0 : 1);
        p.routing[m] = {slot0, 3};
    }
    // expert 0: 1 row -> 16, expert 1: 86 rows -> 96, expert 2: 513 rows -> 544, expert 3: 600 rows -> 608 (= M_size)
    ASSERT_EQ(expectedGemmRowSizes(p), (std::vector<size_t>{16, 96, 544, 608}));

    auto bound = bindProblem(p, graphContext);
    GatherMatmulDnnlExecutor executor(GatherMatmulAttrs{}, bound.memory, context);
    ASSERT_TRUE(GatherMatmulDnnlExecutorTestPeer::bf16AmxMode(executor));

    ASSERT_TRUE(executor.update(bound.memory));
    executor.execute(bound.memory);

    EXPECT_EQ(GatherMatmulDnnlExecutorTestPeer::gemmRowSizes(executor), expectedGemmRowSizes(p));
    checkOutput(p, bound.dst);

    // A second execution of the same shape reuses what was built: nothing new, nothing lost.
    std::fill_n(bound.dst->getDataAs<ov::bfloat16>(), p.topk * p.M * p.N, ov::bfloat16(0.0F));
    executor.execute(bound.memory);
    EXPECT_EQ(GatherMatmulDnnlExecutorTestPeer::gemmRowSizes(executor), expectedGemmRowSizes(p));
    checkOutput(p, bound.dst);
}

// The defect the sizing fix removes. Before it, every active gather index ran the full-length GEMM, so the
// executor held exactly one GEMM height, M_size, however few rows an index received. Here three indices get
// 1, 2 and 3 rows and one gets the other 250: the executor must hold a 16-row GEMM for the small ones, not
// run them at 256 rows.
TEST_F(GatherMatmulDnnlExecutorTest, DoesNotRunEveryGatherIndexAtTheFullInputLength) {
    Problem p{1, 256, 64, 32, 4, {}};
    p.routing.resize(p.M);
    for (size_t m = 0; m < p.M; ++m) {
        p.routing[m] = {m < 1 ? 0 : (m < 3 ? 1 : (m < 6 ? 2 : 3))};
    }
    const size_t M_size = gathermatmul::normalizeM(p.M);
    ASSERT_EQ(M_size, 256U);

    auto bound = bindProblem(p, graphContext);
    GatherMatmulDnnlExecutor executor(GatherMatmulAttrs{}, bound.memory, context);
    ASSERT_TRUE(executor.update(bound.memory));
    executor.execute(bound.memory);

    const auto sizes = GatherMatmulDnnlExecutorTestPeer::gemmRowSizes(executor);
    EXPECT_NE(sizes, std::vector<size_t>{M_size}) << "every gather index ran a GEMM of the full input length ("
                                                  << M_size << " rows): the behaviour before the fix";
    EXPECT_EQ(sizes, (std::vector<size_t>{16, 256}));
    checkOutput(p, bound.dst);
}

// A gather index outside [0, E) must be rejected, not used to read past the weights.
TEST_F(GatherMatmulDnnlExecutorTest, RejectsAGatherIndexOutOfRange) {
    Problem p{1, 32, 64, 32, 4, {}};
    p.routing.assign(p.M, {1});
    p.routing[5] = {4};  // E == 4
    auto bound = bindProblem(p, graphContext);
    GatherMatmulDnnlExecutor executor(GatherMatmulAttrs{}, bound.memory, context);
    ASSERT_TRUE(executor.update(bound.memory));
    EXPECT_THROW(executor.execute(bound.memory), ov::Exception);
}

// The input length changes: the per-height table is rebuilt for the new length and holds only the heights
// the new routing needs. Primitives already built for a height come back from the runtime cache.
TEST_F(GatherMatmulDnnlExecutorTest, RebuildsTheHeightTableWhenTheInputLengthChanges) {
    Problem big{2, 600, 64, 32, 8, {}};
    big.routing.resize(big.M);
    for (size_t m = 0; m < big.M; ++m) {
        big.routing[m] = {m < 300 ? 1 : 2, 3};  // 300 -> 304, 300 -> 304, 600 -> 608
    }
    Problem small{2, 40, 64, 32, 8, {}};
    small.routing.resize(small.M);
    for (size_t m = 0; m < small.M; ++m) {
        small.routing[m] = {1, m < 7 ? 5 : 6};  // 40 -> 48 (= M_size), 7 -> 16, 33 -> 48
    }
    ASSERT_EQ(expectedGemmRowSizes(big), (std::vector<size_t>{304, 608}));
    ASSERT_EQ(expectedGemmRowSizes(small), (std::vector<size_t>{16, 48}));

    auto boundBig = bindProblem(big, graphContext);
    GatherMatmulDnnlExecutor executor(GatherMatmulAttrs{}, boundBig.memory, context);
    ASSERT_TRUE(executor.update(boundBig.memory));
    executor.execute(boundBig.memory);
    EXPECT_EQ(GatherMatmulDnnlExecutorTestPeer::gemmRowSizes(executor), expectedGemmRowSizes(big));
    checkOutput(big, boundBig.dst);

    auto boundSmall = bindProblem(small, graphContext);
    ASSERT_TRUE(executor.update(boundSmall.memory));
    executor.execute(boundSmall.memory);
    EXPECT_EQ(GatherMatmulDnnlExecutorTestPeer::gemmRowSizes(executor), expectedGemmRowSizes(small));
    checkOutput(small, boundSmall.dst);

    // and back to the long input
    std::fill_n(boundBig.dst->getDataAs<ov::bfloat16>(), big.topk * big.M * big.N, ov::bfloat16(0.0F));
    ASSERT_TRUE(executor.update(boundBig.memory));
    executor.execute(boundBig.memory);
    EXPECT_EQ(GatherMatmulDnnlExecutorTestPeer::gemmRowSizes(executor), expectedGemmRowSizes(big));
    checkOutput(big, boundBig.dst);
}

// A single-row input (decode) takes the GEMV path and must still produce the right output after multi-row
// inputs have been executed.
TEST_F(GatherMatmulDnnlExecutorTest, SingleRowInputUsesTheGemvPath) {
    Problem multi{2, 48, 64, 32, 8, {}};
    multi.routing.resize(multi.M);
    for (size_t m = 0; m < multi.M; ++m) {
        multi.routing[m] = {static_cast<int32_t>(m % 4), 7};
    }
    Problem single{2, 1, 64, 32, 8, {{4, 6}}};

    auto boundMulti = bindProblem(multi, graphContext);
    GatherMatmulDnnlExecutor executor(GatherMatmulAttrs{}, boundMulti.memory, context);
    ASSERT_TRUE(executor.update(boundMulti.memory));
    executor.execute(boundMulti.memory);
    checkOutput(multi, boundMulti.dst);
    EXPECT_NE(std::string(impl_type_to_string(executor.implType())).find("amx"), std::string::npos);

    auto boundSingle = bindProblem(single, graphContext);
    ASSERT_TRUE(executor.update(boundSingle.memory));
    executor.execute(boundSingle.memory);
    checkOutput(single, boundSingle.dst);
}

// Compressed weights (u8 with per-channel scales), the form int4/int8 MoE models are served in. The sizing
// has to hold there too, and this is where the implementation-type label matters: before the fix the node
// reported the GEMV's kernel after a multi-row input, which for compressed weights is not an AMX kernel,
// although the AMX GEMM is what ran.
TEST_F(GatherMatmulDnnlExecutorTest, CompressedWeightsSizeEachGatherIndexAndReportTheGemmKernel) {
    Problem p{2, 300, 64, 32, 8, {}, true};
    p.routing.resize(p.M);
    for (size_t m = 0; m < p.M; ++m) {
        p.routing[m] = {m < 5 ? 0 : 1, m < 200 ? 2 : 3};  // 5 -> 16, 295 -> 304, 200 -> 208, 100 -> 112
    }
    ASSERT_EQ(expectedGemmRowSizes(p), (std::vector<size_t>{16, 112, 208, 304}));

    auto bound = bindProblem(p, graphContext);
    GatherMatmulDnnlExecutor executor(GatherMatmulAttrs{}, bound.memory, context);
    ASSERT_TRUE(GatherMatmulDnnlExecutorTestPeer::bf16AmxMode(executor));
    ASSERT_TRUE(executor.update(bound.memory));
    executor.execute(bound.memory);

    EXPECT_EQ(GatherMatmulDnnlExecutorTestPeer::gemmRowSizes(executor), expectedGemmRowSizes(p));
    checkOutput(p, bound.dst);
    const std::string implType = impl_type_to_string(executor.implType());
    EXPECT_NE(implType.find("amx"), std::string::npos)
        << "after a " << p.M << "-row input the node reports " << implType
        << ", the GEMV kernel: the label before the fix, although the AMX GEMM executed";

    // Back to one row: the GEMV runs and is what gets reported.
    Problem single{2, 1, 64, 32, 8, {{6, 7}}, true};
    auto boundSingle = bindProblem(single, graphContext);
    ASSERT_TRUE(executor.update(boundSingle.memory));
    executor.execute(boundSingle.memory);
    checkOutput(single, boundSingle.dst);
}
