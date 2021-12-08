// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/read_ir/read_ir.hpp"

namespace ov {
namespace test {
namespace subgraph {

TEST_P(ReadIRTest, ReadIR) {
    run();
}


TEST_P(ReadIRTest, QueryNetwork) {
    QueryNetwork();
}

} // namespace subgraph
} // namespace test
} // namespace ov
