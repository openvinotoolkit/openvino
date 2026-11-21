// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The oneDNN GatherMatmul executor builds the inner-product output with the ARG_DST precision, which may differ from
// the source precision (bf16 source -> f32 destination). The M > 1 path goes through a temporary output buffer, so
// both the buffer layout and the copy back to the destination have to follow the destination precision.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "cpu_memory.h"
#include "cpu_shape.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/empty_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_gathermatmul_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "weights_cache.hpp"

namespace ov::intel_cpu {

namespace {

using DstPrecisionParams = std::tuple<ov::element::Type /* src and weights */, ov::element::Type /* dst */>;

class GatherMatmulDstPrecisionTest : public ::testing::TestWithParam<DstPrecisionParams> {
public:
    static std::string getTestCaseName(const ::testing::TestParamInfo<DstPrecisionParams>& obj) {
        const auto& [src_prc, dst_prc] = obj.param;
        return "src_" + src_prc.get_type_name() + "_dst_" + dst_prc.get_type_name();
    }

protected:
    static MemoryPtr make_memory(const dnnl::engine& eng, ov::element::Type prc, const VectorDims& dims) {
        return std::make_shared<Memory>(eng, std::make_shared<CpuBlockedMemoryDesc>(prc, Shape(dims)));
    }

    static MemoryPtr make_empty_memory(const dnnl::engine& eng) {
        return std::make_shared<Memory>(eng, std::make_shared<EmptyMemoryDesc>());
    }

    // Stores a float with the precision of the tensor.
    static void store(MemoryPtr& mem, size_t idx, float value) {
        if (mem->getDesc().getPrecision() == ov::element::bf16) {
            mem->getDataAs<ov::bfloat16>()[idx] = ov::bfloat16(value);
        } else {
            mem->getDataAs<float>()[idx] = value;
        }
    }

    static float load(const MemoryPtr& mem, size_t idx) {
        if (mem->getDesc().getPrecision() == ov::element::bf16) {
            return static_cast<float>(mem->getDataAs<ov::bfloat16>()[idx]);
        }
        return mem->getDataAs<float>()[idx];
    }
};

TEST_P(GatherMatmulDstPrecisionTest, OutputFollowsDstPrecision) {
#ifndef OPENVINO_ARCH_X86_64
    GTEST_SKIP() << "The oneDNN GatherMatmul executor is x64 only";
#else
    const auto& [src_prc, dst_prc] = GetParam();
    if (src_prc == ov::element::bf16 && !ov::with_cpu_x86_bfloat16()) {
        GTEST_SKIP() << "bf16 is not supported on this CPU";
    }

    // G experts, each routed by TOPK slots; every expert receives M rows, which is enough for the GEMM path
    // (M > 1 with at least 4 rows per expert) and not a multiple of the GEMM row rounding.
    constexpr size_t G = 2;
    constexpr size_t TOPK = 2;
    constexpr size_t M = 37;
    constexpr size_t K = 64;
    constexpr size_t N = 48;

    Config conf;
    conf.rtCacheCapacity = 100;
    auto graph_context = std::make_shared<GraphContext>(conf, std::make_shared<WeightsSharing>(), false);
    auto context = std::make_shared<const ExecutorContext>(graph_context,
                                                            std::vector<impl_desc_type>{},
                                                            std::make_shared<std::unordered_map<std::string, MemoryPtr>>());
    const auto& eng = graph_context->getEngine();

    auto src = make_memory(eng, src_prc, {TOPK, M, K});
    auto wei = make_memory(eng, src_prc, {G, N, K});
    auto idx = make_memory(eng, ov::element::i32, {M, TOPK});
    auto dst = make_memory(eng, dst_prc, {TOPK, M, N});

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.F, 1.F);
    for (size_t i = 0; i < src->getShape().getElementsCount(); i++) {
        store(src, i, dist(gen));
    }
    for (size_t i = 0; i < wei->getShape().getElementsCount(); i++) {
        store(wei, i, dist(gen));
    }
    // Every row is routed to both experts: slot t -> expert t.
    auto* idx_data = idx->getDataAs<int32_t>();
    for (size_t m = 0; m < M; m++) {
        for (size_t t = 0; t < TOPK; t++) {
            idx_data[m * TOPK + t] = static_cast<int32_t>(t);
        }
    }
    // Sentinel to notice rows that are never written or written with a wrong element size.
    constexpr float sentinel = 1234.F;
    for (size_t i = 0; i < dst->getShape().getElementsCount(); i++) {
        store(dst, i, sentinel);
    }

    MemoryArgs args{{ARG_SRC, src},
                    {ARG_WEI, wei},
                    {ARG_SRC_1, idx},
                    {ARG_BIAS, make_empty_memory(eng)},
                    {ARG_SRC_3, make_empty_memory(eng)},
                    {ARG_SRC_4, make_empty_memory(eng)},
                    {ARG_DST, dst}};

    GatherMatmulDnnlExecutor executor(GatherMatmulAttrs{}, args, context);
    ASSERT_TRUE(executor.update(args));
    executor.execute(args);

    // The reference uses the (possibly bf16-rounded) inputs, accumulating in f32.
    const float tolerance = dst_prc == ov::element::bf16 ? 5e-2F : 1e-3F;
    for (size_t t = 0; t < TOPK; t++) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float ref = 0.F;
                for (size_t k = 0; k < K; k++) {
                    ref += load(src, (t * M + m) * K + k) * load(wei, (t * N + n) * K + k);
                }
                const float actual = load(dst, (t * M + m) * N + n);
                ASSERT_NEAR(ref, actual, tolerance * std::max(1.F, std::abs(ref)))
                    << "slot " << t << " row " << m << " col " << n;
            }
        }
    }
#endif
}

// f32 -> f32 and bf16 -> bf16 keep the source precision. bf16 -> f32 is the case where the output differs from the
// source, and it needs avx512_core_bf16 (skipped otherwise).
INSTANTIATE_TEST_SUITE_P(smoke_GatherMatmulDstPrecision,
                         GatherMatmulDstPrecisionTest,
                         ::testing::Values(DstPrecisionParams{ov::element::f32, ov::element::f32},
                                           DstPrecisionParams{ov::element::bf16, ov::element::bf16},
                                           DstPrecisionParams{ov::element::bf16, ov::element::f32}),
                         GatherMatmulDstPrecisionTest::getTestCaseName);

}  // namespace

}  // namespace ov::intel_cpu
