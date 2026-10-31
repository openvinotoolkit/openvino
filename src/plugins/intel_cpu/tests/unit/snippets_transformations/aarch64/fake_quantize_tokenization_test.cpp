// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../common/fake_quantize_tokenization_test.hpp"

#include <gtest/gtest.h>

#include "openvino/opsets/opset1.hpp"
#include "transformations/snippets/aarch64/pass/snippets_mark_skipped.hpp"

namespace ov {
namespace test {
namespace snippets {

class Aarch64FakeQuantizeTokenizationTest : public FakeQuantizeTokenizationTest {
protected:
    void register_backend_specific_passes() override {
        manager.register_pass<ov::intel_cpu::SnippetsMarkSkipped>();
    }
};

TEST_F(Aarch64FakeQuantizeTokenizationTest, smoke_Snippets_ConvolutionWithFakeQuantize) {
    // ARM64 excludes FakeQuantize from SupportsFusingWithConvolution_Simple. RISC-V64 has no mark-skipped pass. X64
    // includes FakeQuantize in that list.
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
