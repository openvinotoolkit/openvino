// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/paged_gated_delta_net.hpp"

namespace ov::test {

TEST_P(PagedGatedDeltaNetLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

}  // namespace ov::test
