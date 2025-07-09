// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_test_utils.hpp>

namespace ov {
namespace test {
namespace snippets {

class MoveBroadcastTests : public TransformationTestsF {
protected:
    void SetUp() override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
