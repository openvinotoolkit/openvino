// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/integer_reduce_mean.hpp"

namespace ov {
namespace test {

TEST_P(IntegerReduceMeanTest, CompareWithRefs){
    run();
};

}  // namespace test
}  // namespace ov
