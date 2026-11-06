// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../common/fake_quantize_tokenization_test.hpp"

#include <gtest/gtest.h>

#include "openvino/opsets/opset1.hpp"

namespace ov {
namespace test {
namespace snippets {

TEST_F(FakeQuantizeTokenizationTest, smoke_Snippets_ConvolutionWithFakeQuantize) {
    // RISC-V64 has no mark-skipped pass. X64 includes FakeQuantize in SupportsFusingWithConvolution_Simple.
    model = FakeQuantizeFunction::getOperationAndFakeQuantize({{1, 3, 16, 16}},
                                                              element::f32,
                                                              {{}, {}, {}, {}},
                                                              true,
                                                              FunctionHelper::makePrerequisitesOriginal(),
                                                              std::make_shared<ov::op::v1::Convolution>());

    model_ref = FakeQuantizeFunction::getSubgraphWithFakeQuantize({{1, 3, 16, 16}},
                                                                  element::f32,
                                                                  {{}, {}, {}, {}},
                                                                  true,
                                                                  makeFakeQuantizeConvolutionPrerequisites());
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
