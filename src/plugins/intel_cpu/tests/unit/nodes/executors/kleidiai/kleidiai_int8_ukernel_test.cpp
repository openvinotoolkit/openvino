// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// MSVC builds only the KleidiAI assembly kernels: each INT8 pair falls back to a single kernel there
#if !defined(_MSC_VER)

#    include <gtest/gtest.h>

#    include <algorithm>
#    include <cmath>
#    include <cstddef>
#    include <cstdint>
#    include <cstring>
#    include <limits>
#    include <memory>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <random>
#    include <string>
#    include <tuple>
#    include <vector>

#    include "cpu_memory.h"
#    include "cpu_parallel.hpp"
#    include "cpu_shape.h"
#    include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#    include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"
#    include "memory_desc/cpu_blocked_memory_desc.h"
#    include "nodes/executors/kleidiai/kleidiai_common.hpp"
#    include "openvino/core/type/element_type.hpp"
#    include "openvino/runtime/intel_cpu/properties.hpp"
#    include "utils/arm_isa_support.h"

using namespace ov::intel_cpu;
using kai_common::KAIKernelTag;

namespace {

using DotprodKernel = kai_common::uKernelQai8dxpQsi8cxp<KAIKernelTag::I8_NEON_DOTPROD>;

const std::vector<size_t> multi_row_counts = {0, 2, 3, 4, 5, 15, 16, 17, 33, 1024};

// The GEMV runs only for single-row calls, and only when allowed (it needs FEAT_DotProd).
TEST(KleidiAIInt8KernelSelection, DotprodPair) {
    EXPECT_TRUE(DotprodKernel::selectKernel(1, true).run_matmul ==
                &kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod);
    EXPECT_TRUE(DotprodKernel::selectKernel(1, false).run_matmul ==
                &kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod);
    for (const size_t M : multi_row_counts) {
        EXPECT_TRUE(DotprodKernel::selectKernel(M, true).run_matmul ==
                    &kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod)
            << "M=" << M;
    }
}

MemoryPtr makeMemory(const dnnl::engine& eng, ov::element::Type prc, const VectorDims& dims) {
    return std::make_shared<Memory>(eng, std::make_shared<CpuBlockedMemoryDesc>(prc, Shape(dims)));
}

constexpr size_t canary_bytes = 256;
constexpr uint8_t canary_value = 0xA5;

// Runs one execute() call the way MatMulKleidiAIExecutor does: update() sizes the LHS scratchpad
// with getLHSPackedSize(M), then execute(). Checks that the call stays inside the scratchpad and
// writes every output element.
template <KAIKernelTag tag>
std::vector<float> run(kai_common::uKernelQai8dxpQsi8cxp<tag>& kernel,
                       MemoryPtr& lhsPackedMem,
                       const dnnl::engine& eng,
                       const CpuParallelPtr& cpuParallel,
                       const float* lhs,
                       size_t M,
                       size_t N,
                       size_t K) {
    const size_t lhsPackedSize = kernel.getLHSPackedSize(M);
    lhsPackedMem = makeMemory(eng, ov::element::i8, {lhsPackedSize + canary_bytes});
    auto* canary = lhsPackedMem->getDataAs<uint8_t>() + lhsPackedSize;
    std::memset(canary, canary_value, canary_bytes);

    auto srcMem = makeMemory(eng, ov::element::f32, {M, K});
    std::memcpy(srcMem->getData(), lhs, M * K * sizeof(float));
    auto dstMem = makeMemory(eng, ov::element::f32, {M, N});
    std::fill_n(dstMem->getDataAs<float>(), M * N, std::numeric_limits<float>::quiet_NaN());

    kernel.execute(cpuParallel, M, K, dstMem, srcMem);

    EXPECT_TRUE(std::all_of(canary,
                            canary + canary_bytes,
                            [](uint8_t b) {
                                return b == canary_value;
                            }))
        << "LHS packing wrote past getLHSPackedSize(" << M << ")";
    const auto* dst = dstMem->getDataAs<const float>();
    EXPECT_TRUE(std::none_of(dst,
                             dst + M * N,
                             [](float v) {
                                 return std::isnan(v);
                             }))
        << "output not fully written for M=" << M;
    return {dst, dst + M * N};
}

using Int8UKernelParams = std::tuple<KAIKernelTag, size_t /*N*/, size_t /*K*/>;

class KleidiAIInt8GemvGemmTest : public ::testing::TestWithParam<Int8UKernelParams> {
public:
    static std::string getTestCaseName(const ::testing::TestParamInfo<Int8UKernelParams>& obj) {
        const auto& [tag, N, K] = obj.param;
        return std::string(tag == KAIKernelTag::I8_NEON_DOTPROD ? "dotprod" : "i8mm") + "_N" + std::to_string(N) +
               "_K" + std::to_string(K);
    }

protected:
    template <KAIKernelTag tag>
    void check(size_t N, size_t K) {
        const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        const auto cpuParallel = std::make_shared<CpuParallel>(ov::intel_cpu::TbbPartitioner::STATIC);
        std::mt19937 gen(static_cast<uint32_t>(N * 7919 + K));

        // Weights K x N (kxn layout, as packed by the executor), per-channel scales, bias
        auto weiMem = makeMemory(eng, ov::element::i8, {K, N});
        auto* wei = weiMem->getDataAs<int8_t>();
        std::uniform_int_distribution<int> weiDist(-127, 127);
        std::generate_n(wei, K * N, [&] {
            return static_cast<int8_t>(weiDist(gen));
        });
        std::uniform_real_distribution<float> scaleDist(0.001F, 0.01F);
        std::vector<float> scales(N);
        std::generate(scales.begin(), scales.end(), [&] {
            return scaleDist(gen);
        });
        std::uniform_real_distribution<float> unitDist(-1.F, 1.F);
        auto biasMem = makeMemory(eng, ov::element::f32, {N});
        std::generate_n(biasMem->getDataAs<float>(), N, [&] {
            return unitDist(gen);
        });

        // Both objects work on one packed RHS buffer and one LHS scratchpad, so the GEMV
        // has to read the layout the GEMM packs
        MemoryPtr lhsPackedMem;
        kai_common::uKernelQai8dxpQsi8cxp<tag> withGemv(N, K, lhsPackedMem, true);
        kai_common::uKernelQai8dxpQsi8cxp<tag> gemmOnly(N, K, lhsPackedMem, false);
        auto rhsPacked = makeMemory(eng, ov::element::i8, {gemmOnly.get_rhsPackedSize()});
        gemmOnly.packData(false, weiMem, biasMem, true, scales.data(), rhsPacked);
        withGemv.packData(false, weiMem, biasMem, true, scales.data(), rhsPacked);

        for (const size_t M : {1, 2, 3, 4, 5, 15, 16, 17, 33}) {
            SCOPED_TRACE("M=" + std::to_string(M));
            std::vector<float> lhs(M * K);
            std::generate(lhs.begin(), lhs.end(), [&] {
                return unitDist(gen);
            });

            // One call with all rows: GEMM when M > 1, GEMV when M == 1
            const auto batch = run<tag>(withGemv, lhsPackedMem, eng, cpuParallel, lhs.data(), M, N, K);

            for (size_t r = 0; r < M; ++r) {
                const float* row = lhs.data() + r * K;
                // GEMV on the row alone, and the GEMM on the same single row
                const auto gemvRow = run<tag>(withGemv, lhsPackedMem, eng, cpuParallel, row, 1, N, K);
                const auto gemmRow = run<tag>(gemmOnly, lhsPackedMem, eng, cpuParallel, row, 1, N, K);
                // Per-row LHS quantization and the same epilogue make the kernels bit-identical
                EXPECT_EQ(std::memcmp(gemvRow.data(), batch.data() + r * N, N * sizeof(float)), 0)
                    << "GEMV row differs from the batched result, row " << r;
                EXPECT_EQ(std::memcmp(gemvRow.data(), gemmRow.data(), N * sizeof(float)), 0)
                    << "GEMV and GEMM differ on a single row, row " << r;
            }
        }
    }
};

TEST_P(KleidiAIInt8GemvGemmTest, RowsMatchBitExact) {
    const auto& [tag, N, K] = GetParam();
    if (!hasArmISASupport(ArmISA::DOTPROD)) {
        GTEST_SKIP() << "FEAT_DotProd is not available";
    }
    check<KAIKernelTag::I8_NEON_DOTPROD>(N, K);
}

INSTANTIATE_TEST_SUITE_P(smoke_KleidiAIInt8,
                         KleidiAIInt8GemvGemmTest,
                         ::testing::Combine(::testing::Values(KAIKernelTag::I8_NEON_DOTPROD),
                                            ::testing::Values(size_t{3}, size_t{30}),
                                            ::testing::Values(size_t{32}, size_t{77}, size_t{513})),
                         KleidiAIInt8GemvGemmTest::getTestCaseName);

}  // namespace

#endif  // !defined(_MSC_VER)
