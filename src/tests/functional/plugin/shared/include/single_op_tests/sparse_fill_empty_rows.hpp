// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/sparse_fill_empty_rows.hpp"

namespace ov::test {
TEST_P(SparseFillEmptyRowsLayerTest, CompareWithRefs) {
    run();
};
}  // namespace ov::test
