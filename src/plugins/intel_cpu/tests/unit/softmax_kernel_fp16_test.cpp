// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "nodes/kernels/scaled_attn/softmax.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"

namespace {

// Masked positions must be filled with the lowest f16 value. std::numeric_limits<float16_t>::lowest() is 0
// because the __fp16 type is not specialized, which gave masked positions more weight than the real scores.
TEST(SoftmaxKernelFp16Test, CausalMaskedPositionsGetZeroWeight) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    std::vector<ov::float16> scores(16, ov::float16(-1.0F));
    std::vector<ov::float16> weights(scores.size());
    std::vector<uint8_t> causal_mask(scores.size(), 0);
    std::fill(causal_mask.begin() + 8, causal_mask.end(), 1);

    ov::Extensions::Cpu::XARCH::attn_softmax(scores.data(),
                                             weights.data(),
                                             1.0F,
                                             nullptr,
                                             nullptr,
                                             causal_mask.data(),
                                             false,
                                             scores.size(),
                                             scores.size(),
                                             ov::element::f16,
                                             ov::element::f16,
                                             ov::element::f16,
                                             nullptr);
    for (size_t i = 0; i < weights.size(); ++i) {
        EXPECT_NEAR(weights[i], causal_mask[i] ? 0.0F : 1.0F / 8, 1e-3F) << "at position " << i;
    }
#else
    GTEST_SKIP() << "The f16 softmax kernel is not built for this target.";
#endif
}

}  // namespace
