// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "../../../src/plugin/transformations_pipeline.cpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::test::intel_gpu {

// should_preserve_math_f16_rounding() decides whether Math ops (Cos, Sin, ...) that directly
// feed a Floor should have their f16 rounding preserved across the f16->f32 ConvertPrecision
// pass. It must follow the originally *requested* inference precision, not a value that may
// have already been normalized elsewhere in the pipeline (e.g. a f16 request silently falling
// back to f32 on devices without native f16 support).
TEST(PreserveMathF16RoundingTest, ExplicitF16RequestPreservedRegardlessOfDeviceFallback) {
    EXPECT_TRUE(ov::intel_gpu::should_preserve_math_f16_rounding(ov::element::f16, /*model_has_f16=*/false));
    EXPECT_TRUE(ov::intel_gpu::should_preserve_math_f16_rounding(ov::element::f16, /*model_has_f16=*/true));
}

TEST(PreserveMathF16RoundingTest, ExplicitNonF16RequestNeverPreserved) {
    EXPECT_FALSE(ov::intel_gpu::should_preserve_math_f16_rounding(ov::element::f32, /*model_has_f16=*/true));
    EXPECT_FALSE(ov::intel_gpu::should_preserve_math_f16_rounding(ov::element::bf16, /*model_has_f16=*/true));
}

TEST(PreserveMathF16RoundingTest, DynamicRequestFollowsModelContent) {
    EXPECT_TRUE(ov::intel_gpu::should_preserve_math_f16_rounding(ov::element::dynamic, /*model_has_f16=*/true));
    EXPECT_FALSE(ov::intel_gpu::should_preserve_math_f16_rounding(ov::element::dynamic, /*model_has_f16=*/false));
}

}  // namespace ov::test::intel_gpu
