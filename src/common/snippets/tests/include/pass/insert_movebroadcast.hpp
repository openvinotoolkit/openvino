// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "snippets_helpers.hpp"

/* The main purpose is to test whether BroadcastMove ops are inserted.
 * Conversion of Load + BroadcastMove to LoadBroadcastLoad is covered in insert_load_store.cpp
 */

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        Shape, // Input shape 0
        Shape, // Input shape 1
        Shape, // Broadcast shape 0
        Shape // Broadcast shape 1
> insertMoveBroadcastParams;

using ngraph::snippets::op::Subgraph;
class InsertMoveBroadcastTests : public LoweringTests, public testing::WithParamInterface<insertMoveBroadcastParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<insertMoveBroadcastParams> obj);
protected:
    void SetUp() override;
    std::shared_ptr<SnippetsFunctionBase> snippets_function;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
