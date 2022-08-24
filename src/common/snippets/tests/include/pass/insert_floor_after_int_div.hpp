// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "snippets_helpers.hpp"

/* The main purpose is to test that:
 * - Load/Store ops are inserted
 * - Load + BroadcastMove fuses to BroadcastLoad (not the main focus, but still had to cover; overlays with insert_movebroadcast.cpp)
 * - Insert Floor after Divide with integer inputs and output
 */

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        Shape,         // Input shapes
        bool,          // Python Div
        element::Type  // Input element type
> insertFloorAfterIntDivParams;

class InsertFloorAfterIntDivTests : public LoweringTests, public testing::WithParamInterface<insertFloorAfterIntDivParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<insertFloorAfterIntDivParams> obj);
protected:
    void SetUp() override;
    std::shared_ptr<SnippetsFunctionBase> snippets_function;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
