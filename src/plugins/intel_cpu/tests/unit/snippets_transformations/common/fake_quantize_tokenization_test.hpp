// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "fake_quantize_helper.hpp"
#include "function_helper.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace test {
namespace snippets {

class FakeQuantizeTokenizationTest : public TransformationTestsF {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    virtual void register_backend_specific_passes() {}
};

std::vector<std::shared_ptr<ov::Node>> makeFakeQuantizeConvolutionPrerequisites();

}  // namespace snippets
}  // namespace test
}  // namespace ov
