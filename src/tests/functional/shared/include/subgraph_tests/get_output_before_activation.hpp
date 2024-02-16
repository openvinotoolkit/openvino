// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "subgraph/get_output_before_activation.hpp"

namespace ov {
namespace test {

TEST_P(OutputBeforeActivation, CompareWithRefs) {
    run();
};

}  // namespace test
}  // namespace ov
