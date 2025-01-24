// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"
#include "subgraph_matmul.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        std::vector<InputShape>,        // Input  Shapes
        std::vector<ov::element::Type>, // Input Element types
        ov::element::Type,              // Inference precision
        size_t,                         // Expected num nodes
        size_t,                         // Expected num subgraphs
        std::string                     // Target Device
> MLPParams;

class MLP : public testing::WithParamInterface<ov::test::snippets::MLPParams>,
               virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::MLPParams> obj);

protected:
    void SetUp() override;
};


} // namespace snippets
} // namespace test
} // namespace ov