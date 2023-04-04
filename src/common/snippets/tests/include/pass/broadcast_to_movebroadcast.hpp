// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {
typedef std::tuple<
        Shape, // Input shape 0
        Shape, // Input shape 1
        Shape  // Broadcast shape
> BroadcastParams;

class BroadcastToMoveBroadcastTests : public LoweringTests, public testing::WithParamInterface<BroadcastParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BroadcastParams> obj);
protected:
    void SetUp() override;
    std::shared_ptr<SnippetsFunctionBase> snippets_function;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
