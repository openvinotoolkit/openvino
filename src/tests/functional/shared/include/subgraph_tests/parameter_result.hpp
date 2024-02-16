// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "subgraph/parameter_result.hpp"

namespace ov {
namespace test {

TEST_P(ParameterResultSubgraphTest, Inference) {
    run();
}

}  // namespace test
}  // namespace ov
