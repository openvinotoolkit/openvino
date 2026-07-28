// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/paged_attention_token_type.hpp"

namespace ov {
namespace test {
TEST_P(PagedAttentionTokenTypeTest, CompareWithPytorch) {
    run();
};
}  // namespace test
}  // namespace ov
