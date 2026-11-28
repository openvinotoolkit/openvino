// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include "openvino/runtime/system_conf.hpp"

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
#    include "utils/arm_isa_support.h"
#endif

TEST(KleidiAIFCRegression, SME2CapabilityDetectionIsConsistent) {
    EXPECT_NO_THROW((void)ov::with_cpu_arm_sme2());
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    EXPECT_EQ(ov::with_cpu_arm_sme2(), ov::intel_cpu::hasArmISASupport(ov::intel_cpu::ArmISA::SME2));
#endif
}

#if defined(OV_CPU_WITH_KLEIDIAI)

#    include <oneapi/dnnl/dnnl.hpp>

#    include "cpu_memory.h"
#    include "cpu_parallel.hpp"
#    include "dnnl_scratch_pad.h"
#    include "graph_context.h"
#    include "memory_desc/cpu_blocked_memory_desc.h"
#    include "memory_desc/empty_memory_desc.h"
#    include "nodes/executors/executor.hpp"
#    include "nodes/executors/fullyconnected_config.hpp"
#    include "nodes/executors/kleidiai/kleidiai_mm.hpp"
#    include "nodes/executors/memory_arguments.hpp"
#    include "weights_cache.hpp"

#    if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
TEST(KleidiAIFCRegression, SME2LhsPackingNeedsNoBufferForAtMostOneRow) {
    ov::intel_cpu::MemoryPtr lhsPackedMemory;
    ov::intel_cpu::kai_common::uKernel<ov::intel_cpu::kai_common::KAIKernelTag::F32_SME2_MOPA> kernel(17,
                                                                                                      64,
                                                                                                      lhsPackedMemory);
    EXPECT_EQ(kernel.getLHSPackedSize(0), 0U);
    EXPECT_EQ(kernel.getLHSPackedSize(1), 0U);
}
#    endif

namespace ov::intel_cpu::test {
namespace {

struct KleidiAIFCMatMulCase {
    size_t m;
    size_t n;
    size_t k;
    bool weightsNonTransposed;
    bool hasBias;
};

using ShapeSequence = std::vector<size_t>;

MemoryDescPtr makeKleidiAIFCDesc(const ov::element::Type& precision, const ShapeSequence& dims) {
    return std::make_shared<CpuBlockedMemoryDesc>(precision, Shape(dims));
}

MemoryPtr makeKleidiAIFCMemory(const dnnl::engine& engine, MemoryDescPtr desc) {
    return std::make_shared<Memory>(engine, std::move(desc));
}

float kleidiAIFCInputValue(size_t m, size_t k) {
    return static_cast<float>(static_cast<int>((m * 13 + k * 7) % 19) - 9) * 0.03125F;
}

float kleidiAIFCWeightValue(size_t n, size_t k) {
    return static_cast<float>(static_cast<int>((n * 5 + k * 11) % 17) - 8) * 0.015625F;
}

float kleidiAIFCBiasValue(size_t n) {
    return static_cast<float>(static_cast<int>(n % 7) - 3) * 0.0625F;
}

template <typename ParamT>
class KleidiAIFCFixture : public ::testing::TestWithParam<ParamT> {
protected:
    KleidiAIFCFixture()
        : engine(dnnl::engine::kind::cpu, 0),
          cpuParallel(std::make_shared<CpuParallel>(ov::intel_cpu::TbbPartitioner::STATIC)),
          graphContext(std::make_shared<GraphContext>(Config{},
                                                      std::make_shared<WeightsSharing>(),
                                                      false,
                                                      nullptr,
                                                      cpuParallel)),
          executorContext(std::make_shared<ExecutorContext>(graphContext, std::vector<impl_desc_type>{})) {
        scratchPad = graphContext->getScratchPad();
    }

    MemoryArgs makeArgs(size_t m, size_t n, size_t k, bool hasBias, const MemoryPtr& weights, const MemoryPtr& bias)
        const {
        MemoryArgs memory;
        memory[ARG_SRC] = makeKleidiAIFCMemory(engine, makeKleidiAIFCDesc(ov::element::f32, {m, k}));
        memory[ARG_WEI] = weights;
        memory[ARG_BIAS] = bias;
        memory[ARG_DST] = makeKleidiAIFCMemory(engine, makeKleidiAIFCDesc(ov::element::f32, {m, n}));

        auto* src = memory.at(ARG_SRC)->getDataAs<float>();
        for (size_t row = 0; row < m; ++row) {
            for (size_t col = 0; col < k; ++col) {
                src[row * k + col] = kleidiAIFCInputValue(row, col);
            }
        }
        if (hasBias) {
            auto* biasData = bias->getDataAs<float>();
            for (size_t col = 0; col < n; ++col) {
                biasData[col] = kleidiAIFCBiasValue(col);
            }
        }
        return memory;
    }

    void runAndCompare(const KleidiAIFCMatMulCase& testCase, const ShapeSequence& mValues) {
        const auto weightsDesc = makeKleidiAIFCDesc(ov::element::f32, {testCase.n, testCase.k});
        auto weights = makeKleidiAIFCMemory(engine, weightsDesc);
        auto* weightData = weights->getDataAs<float>();
        for (size_t n = 0; n < testCase.n; ++n) {
            for (size_t k = 0; k < testCase.k; ++k) {
                const size_t offset = testCase.weightsNonTransposed ? k * testCase.n + n : n * testCase.k + k;
                weightData[offset] = kleidiAIFCWeightValue(n, k);
            }
        }

        auto bias = testCase.hasBias ? makeKleidiAIFCMemory(engine, makeKleidiAIFCDesc(ov::element::f32, {testCase.n}))
                                     : makeKleidiAIFCMemory(engine, std::make_shared<EmptyMemoryDesc>());
        if (testCase.hasBias) {
            auto* biasData = bias->getDataAs<float>();
            for (size_t n = 0; n < testCase.n; ++n) {
                biasData[n] = kleidiAIFCBiasValue(n);
            }
        }

        FCAttrs attrs;
        attrs.weightsNonTransposed = testCase.weightsNonTransposed;
        auto memory = makeArgs(mValues.front(), testCase.n, testCase.k, testCase.hasBias, weights, bias);
        MatMulKleidiAIExecutor executor(attrs, memory, executorContext);

        for (const auto m : mValues) {
            memory = makeArgs(m, testCase.n, testCase.k, testCase.hasBias, weights, bias);
            ASSERT_TRUE(executor.update(memory));
            executor.execute(memory);

            const auto* actual = memory.at(ARG_DST)->template getDataAs<float>();
            for (size_t row = 0; row < m; ++row) {
                for (size_t col = 0; col < testCase.n; ++col) {
                    float reference = testCase.hasBias ? kleidiAIFCBiasValue(col) : 0.0F;
                    for (size_t reduction = 0; reduction < testCase.k; ++reduction) {
                        reference += kleidiAIFCInputValue(row, reduction) * kleidiAIFCWeightValue(col, reduction);
                    }
                    ASSERT_NEAR(actual[row * testCase.n + col], reference, 1.0e-4F)
                        << "m=" << m << ", n=" << col << ", row=" << row
                        << ", weightsNonTransposed=" << testCase.weightsNonTransposed
                        << ", hasBias=" << testCase.hasBias;
                }
            }
        }
    }

    dnnl::engine engine;
    std::shared_ptr<CpuParallel> cpuParallel;
    GraphContext::Ptr graphContext;
    ExecutorContext::CPtr executorContext;
    DnnlScratchPadPtr scratchPad;
};

class KleidiAIFCLayoutSupportTest : public ::testing::TestWithParam<bool> {};

TEST_P(KleidiAIFCLayoutSupportTest, SupportsFP32Weights) {
#    if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    const bool weightsNonTransposed = GetParam();
    FCConfig config;
    config.attrs.weightsNonTransposed = weightsNonTransposed;
    config.descs[ARG_WEI] = std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32, Shape({17, 64}));
    EXPECT_TRUE(MatMulKleidiAIExecutor::supports(config));
#    else
    GTEST_SKIP() << "KleidiAI FP32 executor is ARM-only";
#    endif
}

INSTANTIATE_TEST_SUITE_P(smoke_KleidiAIFC, KleidiAIFCLayoutSupportTest, ::testing::Bool());

class KleidiAIFCShapeTest : public KleidiAIFCFixture<KleidiAIFCMatMulCase> {};

TEST_P(KleidiAIFCShapeTest, MatchesFP32Reference) {
    const auto testCase = GetParam();
    runAndCompare(testCase, {testCase.m});
}

INSTANTIATE_TEST_SUITE_P(smoke_KleidiAIFC,
                         KleidiAIFCShapeTest,
                         ::testing::Values(KleidiAIFCMatMulCase{1, 17, 64, false, false},
                                           KleidiAIFCMatMulCase{1, 17, 64, false, true},
                                           KleidiAIFCMatMulCase{1, 17, 64, true, false},
                                           KleidiAIFCMatMulCase{1, 17, 64, true, true},
                                           KleidiAIFCMatMulCase{32, 64, 64, false, false},
                                           KleidiAIFCMatMulCase{32, 64, 64, false, true},
                                           KleidiAIFCMatMulCase{32, 64, 64, true, false},
                                           KleidiAIFCMatMulCase{32, 64, 64, true, true},
                                           KleidiAIFCMatMulCase{37, 17, 65, false, false},
                                           KleidiAIFCMatMulCase{37, 17, 65, false, true},
                                           KleidiAIFCMatMulCase{37, 17, 65, true, false},
                                           KleidiAIFCMatMulCase{37, 17, 65, true, true}));

class KleidiAIFCDynamicShapeTest : public KleidiAIFCFixture<bool> {};

TEST_P(KleidiAIFCDynamicShapeTest, UpdatesAndExecutesAcrossMChanges) {
    const bool weightsNonTransposed = GetParam();
    for (const bool hasBias : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << "weightsNonTransposed=" << weightsNonTransposed << ", hasBias=" << hasBias);
        runAndCompare(KleidiAIFCMatMulCase{1, 17, 65, weightsNonTransposed, hasBias}, {1, 37, 1, 64});
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_KleidiAIFC, KleidiAIFCDynamicShapeTest, ::testing::Bool());

}  // namespace
}  // namespace ov::intel_cpu::test

#endif  // OV_CPU_WITH_KLEIDIAI
