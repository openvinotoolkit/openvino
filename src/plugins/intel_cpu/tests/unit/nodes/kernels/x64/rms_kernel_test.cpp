// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/x64/rms_kernel.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::test {
namespace {

using namespace dnnl::impl::cpu::x64;
using namespace ov::intel_cpu::kernel;

class RMSJitKernelTest : public testing::Test {
protected:
    void SetUp() override {
        if (!mayiuse(avx2)) {
            GTEST_SKIP() << "RMSNorm JIT requires AVX2 or AVX-512.";
        }
    }
};

void compute_rms_reference(const float* src, const float* scale, float* dst, size_t size, size_t scale_size, float eps) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum_sq += src[i] * src[i];
    }
    const float mean = sum_sq / static_cast<float>(size);
    const float rsqrt = 1.0f / std::sqrt(mean + eps);

    for (size_t i = 0; i < size; ++i) {
        const float sc = (scale_size == 1) ? scale[0] : scale[i];
        dst[i] = src[i] * rsqrt * sc;
    }
}

template <cpu_isa_t isa>
void run_rms_test(size_t data_size, size_t scale_size, float eps = 1e-5f) {
    jit_rms_compile_params jcp;
    jcp.src_prc = ov::element::f32;
    jcp.dst_prc = ov::element::f32;
    jcp.data_size = data_size;
    jcp.scale_size = scale_size;
    jcp.eps = eps;

    jit_rms_kernel<isa> kernel(jcp);
    ASSERT_NO_THROW(kernel.create_kernel());

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);

    std::vector<float> src(data_size);
    for (auto& v : src) {
        v = dis(gen);
    }

    std::vector<float> scale(scale_size);
    for (auto& v : scale) {
        v = dis(gen);
    }

    std::vector<float> dst(data_size, 0.0f);
    std::vector<float> ref(data_size, 0.0f);

    compute_rms_reference(src.data(), scale.data(), ref.data(), data_size, scale_size, eps);

    jit_rms_call_args args;
    args.src = reinterpret_cast<const uint8_t*>(src.data());
    args.scale = scale.data();
    args.dst = reinterpret_cast<uint8_t*>(dst.data());

    kernel(&args);

    for (size_t i = 0; i < data_size; ++i) {
        EXPECT_NEAR(dst[i], ref[i], 1e-4f) << "Mismatch at index " << i << " for size " << data_size;
    }
}

template <cpu_isa_t isa>
void benchmark_rms(size_t data_size, size_t scale_size, int iterations = 10000) {
    jit_rms_compile_params jcp;
    jcp.src_prc = ov::element::f32;
    jcp.dst_prc = ov::element::f32;
    jcp.data_size = data_size;
    jcp.scale_size = scale_size;
    jcp.eps = 1e-5f;

    jit_rms_kernel<isa> kernel(jcp);
    kernel.create_kernel();

    std::vector<float> src(data_size, 1.25f);
    std::vector<float> scale(scale_size, 0.95f);
    std::vector<float> dst(data_size, 0.0f);

    jit_rms_call_args args;
    args.src = reinterpret_cast<const uint8_t*>(src.data());
    args.scale = scale.data();
    args.dst = reinterpret_cast<uint8_t*>(dst.data());

    // Warmup
    for (int i = 0; i < 500; ++i) {
        kernel(&args);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        kernel(&args);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = total_us / iterations;

    std::cout << "[BENCHMARK] Size: " << data_size
              << " | ScaleSize: " << scale_size
              << " | Avg: " << avg_us << " us (" << iterations << " runs)" << std::endl;
}

template <cpu_isa_t isa>
void run_rms_custom_test(const std::vector<float>& src, const std::vector<float>& scale, float eps = 1e-5f) {
    const size_t data_size = src.size();
    const size_t scale_size = scale.size();
    jit_rms_compile_params jcp;
    jcp.src_prc = ov::element::f32;
    jcp.dst_prc = ov::element::f32;
    jcp.data_size = data_size;
    jcp.scale_size = scale_size;
    jcp.eps = eps;

    jit_rms_kernel<isa> kernel(jcp);
    ASSERT_NO_THROW(kernel.create_kernel());

    std::vector<float> dst(data_size, 0.0f);
    std::vector<float> ref(data_size, 0.0f);

    compute_rms_reference(src.data(), scale.data(), ref.data(), data_size, scale_size, eps);

    jit_rms_call_args args;
    args.src = reinterpret_cast<const uint8_t*>(src.data());
    args.scale = scale.data();
    args.dst = reinterpret_cast<uint8_t*>(dst.data());

    kernel(&args);

    for (size_t i = 0; i < data_size; ++i) {
        EXPECT_NEAR(dst[i], ref[i], 1e-4f) << "Mismatch at index " << i << " for size " << data_size;
    }
}

TEST_F(RMSJitKernelTest, AccuracyAcrossShapes) {
    const std::vector<size_t> test_sizes = {31, 64, 512, 1024, 1041, 2048, 4096};
    for (auto size : test_sizes) {
        if (mayiuse(avx512_core)) {
            run_rms_test<avx512_core>(size, size);
            run_rms_test<avx512_core>(size, 1);
        } else if (mayiuse(avx2)) {
            run_rms_test<avx2>(size, size);
            run_rms_test<avx2>(size, 1);
        }
    }
}

TEST_F(RMSJitKernelTest, AllZerosHandledCorrectly) {
    const std::vector<float> src(2048, 0.0f);
    const std::vector<float> scale(2048, 1.0f);
    if (mayiuse(avx512_core)) {
        run_rms_custom_test<avx512_core>(src, scale);
    } else if (mayiuse(avx2)) {
        run_rms_custom_test<avx2>(src, scale);
    }
}

TEST_F(RMSJitKernelTest, NegativeAndMixedValues) {
    std::vector<float> src(2048);
    const std::vector<float> scale(2048, 0.8f);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = (i % 2 == 0) ? -2.5f : 1.75f;
    }
    if (mayiuse(avx512_core)) {
        run_rms_custom_test<avx512_core>(src, scale);
    } else if (mayiuse(avx2)) {
        run_rms_custom_test<avx2>(src, scale);
    }
}

TEST_F(RMSJitKernelTest, ExtremeDynamicRange) {
    std::vector<float> src(4096);
    const std::vector<float> scale(4096, 1.0f);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = (i % 3 == 0) ? 1e-4f : (i % 3 == 1) ? 50.0f : -25.0f;
    }
    if (mayiuse(avx512_core)) {
        run_rms_custom_test<avx512_core>(src, scale);
    } else if (mayiuse(avx2)) {
        run_rms_custom_test<avx2>(src, scale);
    }
}

TEST_F(RMSJitKernelTest, BenchmarkBaseline) {
    std::cout << "\n=== RMSNorm JIT Kernel Benchmark ===" << std::endl;
    for (auto size : {1024, 2048, 4096}) {
        if (mayiuse(avx512_core)) {
            benchmark_rms<avx512_core>(size, size);
            benchmark_rms<avx512_core>(size, 1);
        } else if (mayiuse(avx2)) {
            benchmark_rms<avx2>(size, size);
            benchmark_rms<avx2>(size, 1);
        }
    }
    std::cout << "====================================\n" << std::endl;
}

}  // namespace
}  // namespace ov::intel_cpu::test
