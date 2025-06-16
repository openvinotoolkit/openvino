// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/split_conv_concat.hpp"

namespace ov {
namespace test {

TEST_P(SplitConvConcat, CompareWithRefImpl) {
    run();
};

TEST_P(SplitConvConcat, QueryModel) {
    query_model();
}

}  // namespace test
}  // namespace ov

