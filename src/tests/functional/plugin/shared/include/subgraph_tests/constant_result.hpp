// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/constant_result.hpp"

namespace ov {
namespace test {

TEST_P(ConstantResultSubgraphTest, Inference) {
    std::cout << "[DEBUG_CVS-172561] ConstantResultSubgraphTest::Inference started" << std::endl;
    std::cout << "[DEBUG_CVS-172561] Before run()" << std::endl;
    std::cout.flush();
    run();
    std::cout << "[DEBUG_CVS-172561] After run() - SUCCESS" << std::endl;
    std::cout << "[DEBUG_CVS-172561] Test body completed, preparing to exit" << std::endl;
    std::cout.flush();
}

}  // namespace test
}  // namespace ov
