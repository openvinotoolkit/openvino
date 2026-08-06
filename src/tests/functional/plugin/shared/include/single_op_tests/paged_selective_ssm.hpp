// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/paged_selective_ssm.hpp"

namespace ov::test {

TEST_P(PagedSelectiveSSMLayerTest, Inference) {
    run();
}

}  // namespace ov::test
