// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/op_impl_check/op_impl_check.hpp"

namespace ov {
namespace test {
namespace subgraph {

TEST_P(OpImplCheckTest, checkPluginImplementation) {
    run();
}

}   // namespace subgraph
}   // namespace test
}   // namespace ov