// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "comparison.hpp"

namespace reference_tests {
namespace ComparisonOpsRefTestDefinitions {
namespace {
TEST_P(ReferenceComparisonLayerTest, CompareWithHardcodedRefs) {
    Exec();
}
}  // namespace
}  //  namespace ComparisonOpsRefTestDefinitions
}  //  namespace reference_tests
