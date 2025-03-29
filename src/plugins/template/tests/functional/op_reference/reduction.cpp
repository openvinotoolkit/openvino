// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduction.hpp"

namespace reference_tests {
namespace ReductionOpsRefTestDefinitions {
namespace {
TEST_P(ReferenceReductionLayerTest, CompareWithHardcodedRefs) {
    Exec();
}
}  // namespace
}  // namespace ReductionOpsRefTestDefinitions
}  // namespace reference_tests
