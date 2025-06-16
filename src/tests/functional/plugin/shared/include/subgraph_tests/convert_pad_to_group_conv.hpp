// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/convert_pad_to_group_conv.hpp"

namespace ov {
namespace test {

TEST_P(ConvertPadToConvTests, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
