// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/paged_attention.hpp"

namespace ov {
namespace test {
TEST_P(PagedAttentionExtensionLayerTest, Inference) {
    run();
};
}  // namespace test
}  // namespace ov
