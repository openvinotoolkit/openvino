// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "tf_utils.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow_lite::tests;

// Test that TFLQuantizeConvert pass doesn't optimize when quantize has multiple outputs
TEST(TFLQuantizeResolverTest, tflite_quantize_multi_outputs) {
    auto model = convert_model("quantize_multi_outs.tflite");

    // Verify that result2 type remains u8
    auto result2_type = model->get_results()[0]->get_element_type();
    EXPECT_EQ(result2_type, element::u8) << "TFLQuantize type should remain u8";
}
