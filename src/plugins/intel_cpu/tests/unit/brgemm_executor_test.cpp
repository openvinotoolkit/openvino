// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <utility>

#include "common_test_utils/test_common.hpp"
#include "nodes/kernels/x64/brgemm_kernel.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "dnnl_postops_composer.h"
#include "post_ops.hpp"
#include "cpu_memory.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/executors/memory_arguments.hpp"

using BrgemmKernelParams = std::tuple<ov::element::Type,
                                      size_t,  // M
                                      size_t,  // N
                                      size_t,  // K
                                      bool>;

namespace brgemmUnitTest {
class BrgemmKernelTest : public ov::test::TestsCommon, public testing::WithParamInterface<BrgemmKernelParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BrgemmKernelParams>& obj) {
        const auto& [rtPrec, M, N, K, postScale] = obj.param;
        std::ostringstream result;
        result << "Prec=" << rtPrec.to_string();
        result << ",M=" << M;
        result << ",N=" << N;
        result << ",K=" << K;
        result << ",WithpostScale=" << postScale;
        return result.str();
    }
};

template <typename T>
void run_test(ov::element::Type rtPrec, size_t M, size_t N, size_t K) {
    ov::intel_cpu::BrgemmKernel gemm(M, N, K, K, N, N, false, rtPrec);
    size_t nthr = 8;
    bool is_f32 = (rtPrec == ov::element::f32);
    std::vector<T> a_data(M * K, (1.0f / K));
    std::vector<T> b_data(K * N, 4.0f);
    std::vector<float> c_data(nthr * M * N, 0.0f);
    std::vector<size_t> wsp(nthr * 4 * 1024, 0.0f);
    std::vector<uint8_t> a_scratch(gemm.get_scratch_a_size(), 0.0f);
    std::vector<uint8_t> b_scratch(gemm.get_scratch_b_size(), 0.0f);
    if (!is_f32) {
        gemm.copy_buffer_b(b_data.data(), b_scratch.data());
    }
    auto m_block_size = gemm.get_mblk_size();
    auto m_blocks = (M + gemm.get_mblk_size() - 1) / m_block_size;
    void* b_ptr = !is_f32 ? static_cast<void*>(b_scratch.data()) : static_cast<void*>(b_data.data());
    ov::parallel_for2d(nthr, m_blocks, [&](size_t i, size_t m_blk) {
        auto m_start = m_blk * m_block_size;
        auto m_end = std::min(m_start + m_block_size, M);
        auto m_cnt = m_end - m_start;
        gemm.executeGemm(m_cnt < m_block_size,
                         a_data.data() + m_start * K,
                         b_ptr,
                         c_data.data() + i * M * N + m_start * N,
                         nullptr,
                         nullptr,
                         wsp.data() + i * 4 * 1024,
                         a_scratch.data());
    });
    ov::parallel_for(nthr, [&](size_t i) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float expected_value = 4.0f;
                double abs = std::fabs(expected_value - c_data[i * M * N + m * N + n]);
                double rel = expected_value ? (abs / std::fabs(expected_value)) : abs;
                if (rel > 0.01f) {
                    std::ostringstream out_stream;
                    out_stream << "actual " << c_data[m * N + n] << "|expected|" << expected_value << std::endl;
                    throw std::runtime_error(out_stream.str());
                }
            }
        }
    });
}

template <>
void run_test<int8_t>(ov::element::Type rtPrec, size_t M, size_t N, size_t K) {
    ov::intel_cpu::BrgemmKernel gemm(M, N, K, K + 4, K + 4, N, true, rtPrec);
    size_t nthr = 8;
    bool is_f32 = (rtPrec == ov::element::f32);
    std::vector<int8_t> a_data(M * (K + 4));
    std::vector<int8_t> b_data(N * (K + 4), 0);
    std::vector<int32_t> c_data(nthr * M * N, 0.0f);
    std::vector<size_t> wsp(nthr * 4 * 1024, 0.0f);
    std::vector<uint8_t> a_scratch(gemm.get_scratch_a_size(), 0.0f);
    std::vector<uint8_t> b_scratch(gemm.get_scratch_b_size(), 0.0f);
    for (size_t i = 0; i < M; i++) {
        std::fill_n(a_data.begin() + i * (K + 4), 4, 4096);
        std::iota(a_data.begin() + 4 + i * (K + 4), a_data.begin() + 4 + i * (K + 4) + K, 1);
    }
    for (size_t i = 0; i < N; i++) {
        std::fill_n(a_data.begin() + i * (K + 4), 4, 4096);
        std::fill_n(b_data.begin() + 4 + i * (K + 4), K, i + 1);
    }
    if (!is_f32) {
        gemm.copy_buffer_b(b_data.data() + 4, b_scratch.data());
    }
    auto m_block_size = gemm.get_mblk_size();
    auto m_blocks = (M + gemm.get_mblk_size() - 1) / m_block_size;
    void* b_ptr = !is_f32 ? static_cast<void*>(b_scratch.data()) : static_cast<void*>(b_data.data());
    ov::parallel_for2d(nthr, m_blocks, [&](size_t i, size_t m_blk) {
        auto m_start = m_blk * m_block_size;
        auto m_end = std::min(m_start + m_block_size, M);
        auto m_cnt = m_end - m_start;
        gemm.executeGemm(m_cnt < m_block_size,
                         a_data.data() + 4 + m_start * K,
                         b_ptr,
                         c_data.data() + i * M * N + m_start * N,
                         nullptr,
                         nullptr,
                         wsp.data() + i * 4 * 1024,
                         a_scratch.data());
    });
    ov::parallel_for(nthr, [&](size_t i) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                int32_t expected_value = (1 + K) * K / 2 * (n + 1);
                if (expected_value != c_data[i * M * N + m * N + n]) {
                    std::ostringstream out_stream;
                    out_stream << m << "|" << n << "|actual " << c_data[m * N + n] << "|expected|" << expected_value
                               << std::endl;
                    throw std::runtime_error(out_stream.str());
                }
            }
        }
    });
}

static void run_test_post_scales(ov::element::Type rtPrec, size_t M, size_t N, size_t K) {
    ov::intel_cpu::BrgemmKernelQuantized gemm(M,
                                              N,
                                              K,
                                              K + 4,
                                              K + 4,
                                              N,
                                              N,
                                              true,
                                              rtPrec,
                                              ov::element::f32,
                                              ov::intel_cpu::BrgemmKernel::ScaleType::PER_CHANNEL,
                                              false);
    size_t nthr = 8;
    bool is_f32 = (rtPrec == ov::element::f32);
    std::vector<int8_t> a_data(M * (K + 4));
    std::vector<int8_t> b_data(N * (K + 4), 0);
    std::vector<int32_t> c_data(nthr * M * N, 0.0f);
    std::vector<float> d_data(nthr * M * N, 0.0f);
    std::vector<float> b_scale(N, 2.0f);
    std::vector<size_t> wsp(nthr * 4 * 1024, 0.0f);
    std::vector<uint8_t> a_scratch(gemm.get_scratch_a_size(), 0.0f);
    std::vector<uint8_t> b_scratch(gemm.get_scratch_b_size(), 0.0f);
    for (size_t i = 0; i < M; i++) {
        std::fill_n(a_data.begin() + i * (K + 4), 4, 4096);
        std::iota(a_data.begin() + 4 + i * (K + 4), a_data.begin() + 4 + i * (K + 4) + K, 1);
    }
    for (size_t i = 0; i < N; i++) {
        std::fill_n(a_data.begin() + i * (K + 4), 4, 4096);
        std::fill_n(b_data.begin() + 4 + i * (K + 4), K, i + 1);
    }
    if (!is_f32) {
        gemm.copy_buffer_b(b_data.data() + 4, b_scratch.data());
    }
    auto m_block_size = gemm.get_mblk_size();
    auto m_blocks = (M + gemm.get_mblk_size() - 1) / m_block_size;
    void* b_ptr = !is_f32 ? static_cast<void*>(b_scratch.data()) : static_cast<void*>(b_data.data());
    ov::parallel_for2d(nthr, m_blocks, [&](size_t i, size_t m_blk) {
        auto m_start = m_blk * m_block_size;
        auto m_end = std::min(m_start + m_block_size, M);
        auto m_cnt = m_end - m_start;
        gemm.executeGemm(m_cnt < m_block_size,
                         a_data.data() + 4 + m_start * K,
                         b_ptr,
                         c_data.data() + i * M * N + m_start * N,
                         d_data.data() + i * M * N + m_start * N,
                         b_scale.data(),
                         wsp.data() + i * 4 * 1024,
                         a_scratch.data());
    });

    ov::parallel_for(nthr, [&](size_t i) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float expected_value = (1 + K) * K / 2 * (n + 1) * 2.0f;
                if (expected_value != d_data[i * M * N + m * N + n]) {
                    std::ostringstream out_stream;
                    out_stream << m << "|" << n << "|actual " << d_data[m * N + n] << "|expected|" << expected_value
                               << std::endl;
                    throw std::runtime_error(out_stream.str());
                }
            }
        }
    });
}

TEST_P(BrgemmKernelTest, simpleGemmTest) {
    const auto& [rtPrec, M, N, K, postScale] = this->GetParam();
    if (rtPrec == ov::element::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();
    if (rtPrec == ov::element::f16 && !ov::with_cpu_x86_avx512_core_fp16())
        GTEST_SKIP();
    if (rtPrec == ov::element::i8 && !(ov::with_cpu_x86_avx512_core_amx_int8() ||
                                       dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::cpu_isa_t::avx2_vnni_2)))
        GTEST_SKIP();

    if (rtPrec == ov::element::bf16) {
        run_test<ov::bfloat16>(rtPrec, M, N, K);
    } else if (rtPrec == ov::element::f16) {
        run_test<ov::float16>(rtPrec, M, N, K);
    } else if (rtPrec == ov::element::f32) {
        run_test<float>(rtPrec, M, N, K);
    } else {
        if (postScale) {
            run_test_post_scales(rtPrec, M, N, K);
        } else {
            run_test<int8_t>(rtPrec, M, N, K);
        }
    }
}

const std::vector<BrgemmKernelParams> params = {{ov::element::f32, 33, 32, 33, false},
                                                {ov::element::bf16, 33, 32, 33, false},
                                                {ov::element::f16, 33, 32, 33, false},
                                                {ov::element::i8, 32, 32, 80, true},
                                                {ov::element::i8, 32, 32, 64, true}};

INSTANTIATE_TEST_SUITE_P(BrgemmKernelUnitTest,
                         BrgemmKernelTest,
                         ::testing::ValuesIn(params),
                         BrgemmKernelTest::getTestCaseName);
}  // namespace brgemmUnitTest

namespace brgemmPostOpsUnitTest {

using namespace ov::intel_cpu;

static BrgemmKernelBinaryArgs extractBinaryPostOpArgs(const DnnlPrimitiveAttrs& primAttrs) {
    BrgemmKernelBinaryArgs binaryArgs;
    auto* primitiveAttr = primAttrs.attr.get();
    const auto& postOps = primitiveAttr->post_ops_;
    binaryArgs.reserve(postOps.entry_.size());
    unsigned idx = 0;
    for (const auto& postOp : postOps.entry_) {
        if (postOp.is_binary() || postOp.is_depthwise() || postOp.is_quantization()) {
            const auto it = primAttrs.cpuArgs.find(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1);
            OPENVINO_ASSERT(it != primAttrs.cpuArgs.end() && it->second);
            binaryArgs.emplace_back(it->second->getData());
        }
        ++idx;
    }
    return binaryArgs;
}

// Diagnostic: verify GEMM works with post-ops-enabled kernel but no actual post-ops content
TEST(BrgemmKernelPostOpsTest, f32GemmDiagnosticNoPostOps) {
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2))
        GTEST_SKIP();

    const size_t M = 4;
    const size_t N = 32;
    const size_t K = 16;

    std::vector<float> a_data(M * K, 1.0f / K);
    std::vector<float> b_data(K * N);
    for (size_t k = 0; k < K; k++)
        for (size_t n = 0; n < N; n++)
            b_data[k * N + n] = static_cast<float>(n + 1);

    // Kernel WITHOUT post-ops (baseline)
    BrgemmKernel gemmPlain(M, N, K, K, N, N, false, ov::element::f32, false);

    std::vector<float> c_plain(M * N, 0.0f);
    std::vector<uint8_t> wsp(BrgemmKernel::get_wsp_size(), 0);
    std::vector<uint8_t> scratch_a(gemmPlain.get_scratch_a_size(), 0);

    auto mblk = BrgemmKernel::get_mblk_size();
    size_t m_blocks = (M + mblk - 1) / mblk;
    for (size_t mb = 0; mb < m_blocks; mb++) {
        size_t row = mb * mblk;
        bool isTail = row + mblk > M;
        gemmPlain.executeGemm(isTail,
                              a_data.data() + row * K,
                              b_data.data(),
                              c_plain.data() + row * N,
                              nullptr, nullptr,
                              wsp.data(), scratch_a.data());
    }

    // Verify plain GEMM result
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float expected = static_cast<float>(n + 1);
            float actual = c_plain[m * N + n];
            EXPECT_NEAR(actual, expected, 0.01f)
                << "Plain GEMM mismatch at [" << m << "," << n << "]";
        }
    }

    // Now create kernel WITH post-ops enabled but empty post-ops
    BrgemmKernelPostOpsConfig postOpsConfig;
    postOpsConfig.dstType = ov::element::f32;
    postOpsConfig.biasType = ov::element::dynamic;
    postOpsConfig.enabled = true;

    BrgemmKernel gemmWithPostOps(M, N, K, K, N, N, false, ov::element::f32, postOpsConfig, false);

    std::vector<float> c_postops(M * N, 0.0f);
    std::vector<uint8_t> wsp2(BrgemmKernel::get_wsp_size(), 0);
    std::vector<uint8_t> scratch_a2(gemmWithPostOps.get_scratch_a_size(), 0);

    for (size_t mb = 0; mb < m_blocks; mb++) {
        size_t row = mb * mblk;
        bool isTail = row + mblk > M;

        BrgemmKernelPostOpsCallArgs callArgs;
        callArgs.bias = nullptr;
        callArgs.dstDataAnchor = reinterpret_cast<const char*>(c_postops.data());
        callArgs.dstRowLogicalOffset = row * N;

        gemmWithPostOps.executeGemmWithPostOps(isTail,
                                                a_data.data() + row * K,
                                                b_data.data(),
                                                c_postops.data() + row * N,
                                                c_postops.data() + row * N,
                                                nullptr,
                                                wsp2.data(),
                                                scratch_a2.data(),
                                                callArgs);
    }

    // Verify: should be same as plain GEMM (no actual post-ops)
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float expected = static_cast<float>(n + 1);
            float actual = c_postops[m * N + n];
            EXPECT_NEAR(actual, expected, 0.01f)
                << "PostOps GEMM mismatch at [" << m << "," << n << "]"
                << " (plain=" << c_plain[m * N + n] << ")";
        }
    }
}

// Test BrgemmKernel with binary_add post-op (simulates fusingBias)
TEST(BrgemmKernelPostOpsTest, f32GemmWithBiasPostOp) {
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2))
        GTEST_SKIP();

    const size_t M = 4;
    const size_t N = 32;
    const size_t K = 16;

    // Prepare input data: A(M,K) * B(K,N) -> C(M,N)
    // A = all 1/K so each row of A sums to 1.0
    // B[k][n] = n+1, so GEMM result C[m][n] = n+1
    std::vector<float> a_data(M * K, 1.0f / K);
    std::vector<float> b_data(K * N);
    for (size_t k = 0; k < K; k++) {
        for (size_t n = 0; n < N; n++) {
            b_data[k * N + n] = static_cast<float>(n + 1);
        }
    }

    // Bias: bias[n] = 100 * (n + 1)
    std::vector<float> bias_data(N);
    for (size_t n = 0; n < N; n++) {
        bias_data[n] = 100.0f * (n + 1);
    }

    // Build post-ops via DnnlPostOpsComposer
    PostOps postOps;
    postOps.push_back(ScaleShiftPostOp(ScaleShiftPostOp::Type::add, {}, bias_data));

    auto engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    MemoryArgs memory;
    DnnlPostOpsComposer composer(postOps,
                                 engine,
                                 {M, N},
                                 1,       // idxOC = 1 (N dimension)
                                 false,
                                 1 << 0,
                                 memory,
                                 dnnl::memory::data_type::f32);
    auto primAttrs = composer.compose();
    auto binaryArgs = extractBinaryPostOpArgs(primAttrs);

    // Create BrgemmKernel with post-ops
    BrgemmKernelPostOpsConfig postOpsConfig;
    postOpsConfig.attr = primAttrs.attr;
    postOpsConfig.cpuArgs = primAttrs.cpuArgs;
    postOpsConfig.dstType = ov::element::f32;
    postOpsConfig.biasType = ov::element::dynamic;  // no FC ARG_BIAS, bias is via binary post-op
    postOpsConfig.enabled = true;

    BrgemmKernel gemm(M, N, K, K, N, N, false, ov::element::f32, postOpsConfig, false);
    gemm.setPostOpBinaryArgs(std::move(binaryArgs));

    // Prepare buffers
    std::vector<float> dst_data(M * N, 0.0f);
    std::vector<uint8_t> wsp(BrgemmKernel::get_wsp_size(), 0);
    std::vector<uint8_t> scratch_a(gemm.get_scratch_a_size(), 0);

    // For f32, use b_data directly (no copy_buffer_b needed - brgCopyBKernel is null for f32)
    void* b_ptr = static_cast<void*>(b_data.data());

    // Execute with post-ops
    auto mblk = BrgemmKernel::get_mblk_size();
    size_t m_blocks = (M + mblk - 1) / mblk;
    for (size_t mb = 0; mb < m_blocks; mb++) {
        size_t row = mb * mblk;
        bool isTail = row + mblk > M;

        BrgemmKernelPostOpsCallArgs callArgs;
        callArgs.bias = nullptr;
        callArgs.dstDataAnchor = reinterpret_cast<const char*>(dst_data.data());
        callArgs.dstRowLogicalOffset = row * N;

        gemm.executeGemmWithPostOps(isTail,
                                    a_data.data() + row * K,
                                    b_ptr,
                                    dst_data.data() + row * N,
                                    dst_data.data() + row * N,
                                    nullptr,
                                    wsp.data(),
                                    scratch_a.data(),
                                    callArgs);
    }

    // Verify: expected[m][n] = GEMM[m][n] + bias[n] = (n+1) + 100*(n+1) = 101*(n+1)
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float expected = 101.0f * (n + 1);
            float actual = dst_data[m * N + n];
            float diff = std::fabs(expected - actual);
            EXPECT_LT(diff, 0.01f)
                << "Mismatch at [" << m << "," << n << "]: expected=" << expected << " actual=" << actual;
        }
    }
}

// Test BrgemmKernel with explicit bias via brgemm_desc_set_postops bias type
TEST(BrgemmKernelPostOpsTest, f32GemmWithExplicitBias) {
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2))
        GTEST_SKIP();

    const size_t M = 4;
    const size_t N = 32;
    const size_t K = 16;

    std::vector<float> a_data(M * K, 1.0f / K);
    std::vector<float> b_data(K * N);
    for (size_t k = 0; k < K; k++) {
        for (size_t n = 0; n < N; n++) {
            b_data[k * N + n] = static_cast<float>(n + 1);
        }
    }

    // Explicit bias
    std::vector<float> bias_data(N);
    for (size_t n = 0; n < N; n++) {
        bias_data[n] = 100.0f * (n + 1);
    }

    // Create BrgemmKernel with bias-only post-ops (no binary, just bias type)
    BrgemmKernelPostOpsConfig postOpsConfig;
    postOpsConfig.dstType = ov::element::f32;
    postOpsConfig.biasType = ov::element::f32;
    postOpsConfig.enabled = true;

    BrgemmKernel gemm(M, N, K, K, N, N, false, ov::element::f32, postOpsConfig, false);

    std::vector<float> dst_data(M * N, 0.0f);
    std::vector<uint8_t> wsp(BrgemmKernel::get_wsp_size(), 0);
    std::vector<uint8_t> scratch_a(gemm.get_scratch_a_size(), 0);

    // For f32, use b_data directly (no copy_buffer_b needed)
    void* b_ptr = static_cast<void*>(b_data.data());

    auto mblk = BrgemmKernel::get_mblk_size();
    size_t m_blocks = (M + mblk - 1) / mblk;
    for (size_t mb = 0; mb < m_blocks; mb++) {
        size_t row = mb * mblk;
        bool isTail = row + mblk > M;

        BrgemmKernelPostOpsCallArgs callArgs;
        callArgs.bias = bias_data.data();
        callArgs.dstDataAnchor = reinterpret_cast<const char*>(dst_data.data());
        callArgs.dstRowLogicalOffset = row * N;

        gemm.executeGemmWithPostOps(isTail,
                                    a_data.data() + row * K,
                                    b_ptr,
                                    dst_data.data() + row * N,
                                    dst_data.data() + row * N,
                                    nullptr,
                                    wsp.data(),
                                    scratch_a.data(),
                                    callArgs);
    }

    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float expected = 101.0f * (n + 1);
            float actual = dst_data[m * N + n];
            float diff = std::fabs(expected - actual);
            EXPECT_LT(diff, 0.01f)
                << "Mismatch at [" << m << "," << n << "]: expected=" << expected << " actual=" << actual;
        }
    }
}

}  // namespace brgemmPostOpsUnitTest