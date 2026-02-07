// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#if defined(OPENVINO_ARCH_ARM64)
#    include "nodes/kernels/aarch64/jit_int8_conv_kernel.hpp"
#    include "utils/cpu_utils.hpp"
#    if defined(__linux__)
#        include <sys/auxv.h>
#    endif
#    if defined(__linux__) && defined(__aarch64__)
#        include <asm/hwcap.h>
#    endif

namespace {
void pack_mmla_block(const int8_t* src, size_t K, size_t oc_block, int8_t* dst) {
    const size_t k_blocks = K / 8;
    size_t offset = 0;
    for (size_t kb = 0; kb < k_blocks; ++kb) {
        const size_t k_off = kb * 8;
        for (size_t oc = 0; oc + 1 < oc_block; oc += 2) {
            const int8_t* s0 = src + oc * K + k_off;
            const int8_t* s1 = src + (oc + 1) * K + k_off;
            std::copy_n(s0, 8, dst + offset);
            std::copy_n(s1, 8, dst + offset + 8);
            offset += 16;
        }
    }
}
}  // namespace
#endif

TEST(JitInt8ConvKernel, BasicU8S8) {
#if defined(OPENVINO_ARCH_ARM64)
    ov::intel_cpu::aarch64::jit_int8_dot_kernel ker(false);
    ker.create_ker();
    const uint8_t src[4] = {1, 2, 3, 4};
    const int8_t wei[4] = {1, -1, 2, -2};
    int32_t dst = 0;
    ker.ker()(src, wei, &dst, 4, 0);
    EXPECT_EQ(dst, 1 * 1 + 2 * -1 + 3 * 2 + 4 * -2);
#else
    GTEST_SKIP() << "AArch64-only test";
#endif
}

TEST(JitInt8ConvKernel, BasicS8S8) {
#if defined(OPENVINO_ARCH_ARM64)
    ov::intel_cpu::aarch64::jit_int8_dot_kernel ker(true);
    ker.create_ker();
    const int8_t src_signed[4] = {-1, 2, -3, 4};
    const uint8_t* src = reinterpret_cast<const uint8_t*>(src_signed);
    const int8_t wei[4] = {1, -1, 2, -2};
    int32_t dst = 0;
    ker.ker()(src, wei, &dst, 4, 0);
    EXPECT_EQ(dst, -1 * 1 + 2 * -1 + -3 * 2 + 4 * -2);
#else
    GTEST_SKIP() << "AArch64-only test";
#endif
}

TEST(JitInt8ConvKernel, DotAccumulateU8S8) {
#if defined(OPENVINO_ARCH_ARM64)
    ov::intel_cpu::aarch64::jit_int8_dot_kernel ker(false);
    ker.create_ker();
    const uint8_t src[4] = {1, 2, 3, 4};
    const int8_t wei[4] = {1, -1, 2, -2};
    int32_t dst = 0;
    ker.ker()(src, wei, &dst, 4, 0);
    const int32_t first = dst;
    ker.ker()(src, wei, &dst, 4, 1);
    EXPECT_EQ(dst, first * 2);
#else
    GTEST_SKIP() << "AArch64-only test";
#endif
}

TEST(JitInt8ConvKernel, Block4U8S8) {
#if defined(OPENVINO_ARCH_ARM64)
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x4 ker(false);
    ker.create_ker();
    const uint8_t src[4] = {1, 2, 3, 4};
    const int8_t wei[16] = {
        1, -1, 2, -2,
        2, 1, -2, -1,
        -1, -1, 1, 1,
        0, 1, 0, 1,
    };
    int32_t dst[4] = {};
    ker.ker()(src, wei, dst, 4, 4, 0);
    EXPECT_EQ(dst[0], 1 * 1 + 2 * -1 + 3 * 2 + 4 * -2);
    EXPECT_EQ(dst[1], 1 * 2 + 2 * 1 + 3 * -2 + 4 * -1);
    EXPECT_EQ(dst[2], 1 * -1 + 2 * -1 + 3 * 1 + 4 * 1);
    EXPECT_EQ(dst[3], 1 * 0 + 2 * 1 + 3 * 0 + 4 * 1);
#else
    GTEST_SKIP() << "AArch64-only test";
#endif
}

TEST(JitInt8ConvKernel, Block4DotS8S8) {
#if defined(OPENVINO_ARCH_ARM64) && defined(__linux__) && defined(__aarch64__) && defined(HWCAP_ASIMDDP)
    if ((getauxval(AT_HWCAP) & HWCAP_ASIMDDP) == 0) {
        GTEST_SKIP() << "Dot product ISA not available";
    }
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x4_dot ker;
    ker.create_ker();
    const int8_t src[16] = {1, -2, 3, -4, 5, -6, 7, -8, 1, -2, 3, -4, 5, -6, 7, -8};
    const int8_t wei[64] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    };
    int32_t dst[4] = {};
    ker.ker()(src, wei, dst, 16, 16, 0);
    int32_t sum = 0;
    for (int i = 0; i < 16; ++i) {
        sum += src[i];
    }
    EXPECT_EQ(dst[0], sum);
    EXPECT_EQ(dst[1], -sum);
    EXPECT_EQ(dst[2], 2 * sum);
    EXPECT_EQ(dst[3], -2 * sum);
#else
    GTEST_SKIP() << "Dot product ISA not available or not ARM64";
#endif
}

TEST(JitInt8ConvKernel, Block8DotS8S8) {
#if defined(OPENVINO_ARCH_ARM64) && defined(__linux__) && defined(__aarch64__) && defined(HWCAP_ASIMDDP)
    if ((getauxval(AT_HWCAP) & HWCAP_ASIMDDP) == 0) {
        GTEST_SKIP() << "Dot product ISA not available";
    }
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_1x8_dot ker;
    ker.create_ker();
    const int8_t src[16] = {1, -2, 3, -4, 5, -6, 7, -8, 1, -2, 3, -4, 5, -6, 7, -8};
    int8_t wei[8 * 16];
    for (int i = 0; i < 16; ++i) {
        wei[0 * 16 + i] = 1;
        wei[1 * 16 + i] = -1;
        wei[2 * 16 + i] = 2;
        wei[3 * 16 + i] = -2;
        wei[4 * 16 + i] = 3;
        wei[5 * 16 + i] = -3;
        wei[6 * 16 + i] = 4;
        wei[7 * 16 + i] = -4;
    }
    int32_t dst[8] = {};
    ker.ker()(src, wei, dst, 16, 16, 0);
    int32_t sum = 0;
    for (int i = 0; i < 16; ++i) {
        sum += src[i];
    }
    const int32_t scales[8] = {1, -1, 2, -2, 3, -3, 4, -4};
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(dst[i], scales[i] * sum);
    }
#else
    GTEST_SKIP() << "Dot product ISA not available or not ARM64";
#endif
}

TEST(JitInt8ConvKernel, Block4x4DotS8S8) {
#if defined(OPENVINO_ARCH_ARM64) && defined(__linux__) && defined(__aarch64__) && defined(HWCAP_ASIMDDP)
    if ((getauxval(AT_HWCAP) & HWCAP_ASIMDDP) == 0) {
        GTEST_SKIP() << "Dot product ISA not available";
    }
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_dot ker;
    ker.create_ker();
    const int8_t src0[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    const int8_t src1[16] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    const int8_t src2[16] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    const int8_t src3[16] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    const int8_t* src_ptrs[4] = {src0, src1, src2, src3};
    const int8_t wei[64] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    };
    int32_t dst[16] = {};
    ker.ker()(src_ptrs, wei, dst, 16, 16, 4 * sizeof(int32_t), 0);
    const int32_t sums[4] = {16, 32, -16, 48};
    const int32_t scales[4] = {1, -1, 2, -2};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            EXPECT_EQ(dst[r * 4 + c], sums[r] * scales[c]);
        }
    }
#else
    GTEST_SKIP() << "Dot product ISA not available or not ARM64";
#endif
}

TEST(JitInt8ConvKernel, Block4x4DotS8S8Tail) {
#if defined(OPENVINO_ARCH_ARM64) && defined(__linux__) && defined(__aarch64__) && defined(HWCAP_ASIMDDP)
    if ((getauxval(AT_HWCAP) & HWCAP_ASIMDDP) == 0) {
        GTEST_SKIP() << "Dot product ISA not available";
    }
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x4_dot ker;
    ker.create_ker();
    constexpr int K = 18;
    int8_t src0[K];
    int8_t src1[K];
    int8_t src2[K];
    int8_t src3[K];
    for (int i = 0; i < K; ++i) {
        src0[i] = 1;
        src1[i] = 2;
        src2[i] = -1;
        src3[i] = 3;
    }
    const int8_t* src_ptrs[4] = {src0, src1, src2, src3};
    int8_t wei[4 * K];
    for (int i = 0; i < K; ++i) {
        wei[0 * K + i] = 1;
        wei[1 * K + i] = -1;
        wei[2 * K + i] = 2;
        wei[3 * K + i] = -2;
    }
    int32_t dst[16] = {};
    ker.ker()(src_ptrs, wei, dst, K, K, 4 * sizeof(int32_t), 0);
    const int32_t sums[4] = {K, 2 * K, -K, 3 * K};
    const int32_t scales[4] = {1, -1, 2, -2};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            EXPECT_EQ(dst[r * 4 + c], sums[r] * scales[c]);
        }
    }
#else
    GTEST_SKIP() << "Dot product ISA not available or not ARM64";
#endif
}

TEST(JitInt8ConvKernel, Block4x8MmlaPackedS8S8) {
#if defined(OPENVINO_ARCH_ARM64)
    if (!ov::intel_cpu::hasInt8MMSupport()) {
        GTEST_SKIP() << "I8MM ISA not available";
    }
    constexpr int K = 8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_smmla_packed ker;
    ker.create_ker();
    const int8_t src0[K] = {1, 1, 1, 1, 1, 1, 1, 1};
    const int8_t src1[K] = {2, 2, 2, 2, 2, 2, 2, 2};
    const int8_t src2[K] = {-1, -1, -1, -1, -1, -1, -1, -1};
    const int8_t src3[K] = {3, 3, 3, 3, 3, 3, 3, 3};
    const int8_t* src_ptrs[4] = {src0, src1, src2, src3};
    int8_t wei[8 * K];
    const int8_t scales[8] = {1, -1, 2, -2, 3, -3, 4, -4};
    for (int oc = 0; oc < 8; ++oc) {
        for (int k = 0; k < K; ++k) {
            wei[oc * K + k] = scales[oc];
        }
    }
    const size_t stride = (K / 8) * 8 * 8;
    std::vector<int8_t> wei_packed(stride);
    pack_mmla_block(wei, K, 8, wei_packed.data());
    int32_t dst[4 * 8] = {};
    ker.ker()(src_ptrs, wei_packed.data(), dst, K, 0, 8 * sizeof(int32_t), 0);
    const int32_t sums[4] = {K, 2 * K, -K, 3 * K};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 8; ++c) {
            EXPECT_EQ(dst[r * 8 + c], sums[r] * scales[c]);
        }
    }
#else
    GTEST_SKIP() << "AArch64-only test";
#endif
}

TEST(JitInt8ConvKernel, Block4x8MmlaPackedU8S8) {
#if defined(OPENVINO_ARCH_ARM64)
    if (!ov::intel_cpu::hasInt8MMSupport()) {
        GTEST_SKIP() << "I8MM ISA not available";
    }
    constexpr int K = 8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x8_usmmla_packed ker;
    ker.create_ker();
    const uint8_t src0[K] = {1, 1, 1, 1, 1, 1, 1, 1};
    const uint8_t src1[K] = {2, 2, 2, 2, 2, 2, 2, 2};
    const uint8_t src2[K] = {3, 3, 3, 3, 3, 3, 3, 3};
    const uint8_t src3[K] = {4, 4, 4, 4, 4, 4, 4, 4};
    const uint8_t* src_ptrs[4] = {src0, src1, src2, src3};
    int8_t wei[8 * K];
    const int8_t scales[8] = {1, -1, 2, -2, 3, -3, 4, -4};
    for (int oc = 0; oc < 8; ++oc) {
        for (int k = 0; k < K; ++k) {
            wei[oc * K + k] = scales[oc];
        }
    }
    const size_t stride = (K / 8) * 8 * 8;
    std::vector<int8_t> wei_packed(stride);
    pack_mmla_block(wei, K, 8, wei_packed.data());
    int32_t dst[4 * 8] = {};
    ker.ker()(src_ptrs, wei_packed.data(), dst, K, 0, 8 * sizeof(int32_t), 0);
    const int32_t sums[4] = {K, 2 * K, 3 * K, 4 * K};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 8; ++c) {
            EXPECT_EQ(dst[r * 8 + c], sums[r] * scales[c]);
        }
    }
#else
    GTEST_SKIP() << "AArch64-only test";
#endif
}

TEST(JitInt8ConvKernel, Block4x16MmlaPackedS8S8) {
#if defined(OPENVINO_ARCH_ARM64)
    if (!ov::intel_cpu::hasInt8MMSupport()) {
        GTEST_SKIP() << "I8MM ISA not available";
    }
    constexpr int K = 8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_smmla_packed ker;
    ker.create_ker();
    const int8_t src0[K] = {1, 1, 1, 1, 1, 1, 1, 1};
    const int8_t src1[K] = {2, 2, 2, 2, 2, 2, 2, 2};
    const int8_t src2[K] = {-1, -1, -1, -1, -1, -1, -1, -1};
    const int8_t src3[K] = {3, 3, 3, 3, 3, 3, 3, 3};
    const int8_t* src_ptrs[4] = {src0, src1, src2, src3};
    int8_t wei[16 * K];
    const int8_t scales[16] = {1,  -1, 2,  -2, 3,  -3, 4,  -4,
                               5,  -5, 6,  -6, 7,  -7, 8,  -8};
    for (int oc = 0; oc < 16; ++oc) {
        for (int k = 0; k < K; ++k) {
            wei[oc * K + k] = scales[oc];
        }
    }
    const size_t stride = (K / 8) * 16 * 8;
    std::vector<int8_t> wei_packed(stride);
    pack_mmla_block(wei, K, 16, wei_packed.data());
    int32_t dst[4 * 16] = {};
    ker.ker()(src_ptrs, wei_packed.data(), dst, K, 0, 16 * sizeof(int32_t), 0);
    const int32_t sums[4] = {K, 2 * K, -K, 3 * K};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 16; ++c) {
            EXPECT_EQ(dst[r * 16 + c], sums[r] * scales[c]);
        }
    }
#else
    GTEST_SKIP() << "AArch64-only test";
#endif
}

TEST(JitInt8ConvKernel, Block4x16MmlaPackedU8S8) {
#if defined(OPENVINO_ARCH_ARM64)
    if (!ov::intel_cpu::hasInt8MMSupport()) {
        GTEST_SKIP() << "I8MM ISA not available";
    }
    constexpr int K = 8;
    ov::intel_cpu::aarch64::jit_int8_brgemm_kernel_4x16_usmmla_packed ker;
    ker.create_ker();
    const uint8_t src0[K] = {1, 1, 1, 1, 1, 1, 1, 1};
    const uint8_t src1[K] = {2, 2, 2, 2, 2, 2, 2, 2};
    const uint8_t src2[K] = {3, 3, 3, 3, 3, 3, 3, 3};
    const uint8_t src3[K] = {4, 4, 4, 4, 4, 4, 4, 4};
    const uint8_t* src_ptrs[4] = {src0, src1, src2, src3};
    int8_t wei[16 * K];
    const int8_t scales[16] = {1,  -1, 2,  -2, 3,  -3, 4,  -4,
                               5,  -5, 6,  -6, 7,  -7, 8,  -8};
    for (int oc = 0; oc < 16; ++oc) {
        for (int k = 0; k < K; ++k) {
            wei[oc * K + k] = scales[oc];
        }
    }
    const size_t stride = (K / 8) * 16 * 8;
    std::vector<int8_t> wei_packed(stride);
    pack_mmla_block(wei, K, 16, wei_packed.data());
    int32_t dst[4 * 16] = {};
    ker.ker()(src_ptrs, wei_packed.data(), dst, K, 0, 16 * sizeof(int32_t), 0);
    const int32_t sums[4] = {K, 2 * K, 3 * K, 4 * K};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 16; ++c) {
            EXPECT_EQ(dst[r * 16 + c], sums[r] * scales[c]);
        }
    }
#else
    GTEST_SKIP() << "AArch64-only test";
#endif
}
