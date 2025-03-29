// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        InputShape,                  // Input 0 Shape
        InputShape,                  // Input 1 Shape
        InputShape,                  // Input 2 Shape
        ov::element::Type,           // Element type
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> SelectParams;

typedef std::tuple<
        InputShape,                  // Input 0 Shape
        InputShape,                  // Input 1 Shape
        InputShape,                  // Input 2 Shape
        ov::PartialShape,            // Input 3 Shape
        ov::element::Type,           // Element type
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> BroadcastSelectParams;

class Select : public testing::WithParamInterface<ov::test::snippets::SelectParams>,
               virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::SelectParams> obj);

protected:
    void SetUp() override;

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

class BroadcastSelect : public testing::WithParamInterface<ov::test::snippets::BroadcastSelectParams>,
                        virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::BroadcastSelectParams> obj);

protected:
    void SetUp() override;

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

} // namespace snippets
} // namespace test
} // namespace ov
