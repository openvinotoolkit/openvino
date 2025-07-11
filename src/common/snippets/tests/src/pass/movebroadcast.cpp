// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/add.hpp"
#include "snippets/op/broadcastmove.hpp"
#include "snippets/pass/insert_movebroadcast.hpp"
#include "subgraph_movebroadcast.hpp"

#include "pass/movebroadcast.hpp"

using namespace testing;
using namespace ov;

namespace ov {
namespace test {
namespace snippets {

void MoveBroadcastTests::SetUp() {
    TransformationTestsF::SetUp();
    manager.register_pass<::snippets::pass::InsertMoveBroadcast>();
}

TEST_F(MoveBroadcastTests, InsertBroadcastMove) {
    std::vector<PartialShape> input_shapes = {Shape{2, 3}, Shape{1, 2, 1}};
    MoveBroadcastFunction function(input_shapes);
    model = function.getOriginal();
    model_ref = function.getReference();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
