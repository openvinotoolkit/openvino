// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion.hpp"

namespace reference_tests {
namespace ConversionOpsRefTestDefinitions {
namespace {
TEST_P(ReferenceConversionLayerTest, CompareWithHardcodedRefs) {
    Exec();
}
} // namespace
} // namespace ConversionOpsRefTestDefinitions
} // namespace reference_tests
