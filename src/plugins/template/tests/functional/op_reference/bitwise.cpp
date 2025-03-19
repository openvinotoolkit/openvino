// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bitwise.hpp"

namespace reference_tests {
namespace BitwiseOpsRefTestDefinitions {
namespace {

TEST_P(ReferenceBitwiseLayerTest, BitwiseWithHardcodedRefs) {
    Exec();
}

}  // namespace
}  // namespace BitwiseOpsRefTestDefinitions
}  // namespace reference_tests
