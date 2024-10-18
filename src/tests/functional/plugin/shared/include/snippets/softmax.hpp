// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        InputShape,                      // Input 0 Shape
        int,                             // Axis
        size_t,                          // Expected num nodes
        size_t,                          // Expected num subgraphs
        std::string                      // Target Device
> SoftmaxParams;

typedef std::tuple<
        std::pair<InputShape, InputShape>,// Input Shapes
        int,                              // Axis
        size_t,                           // Expected num nodes
        size_t,                           // Expected num subgraphs
        std::string                       // Target Device
> AddSoftmaxParams;

class Softmax : public testing::WithParamInterface<ov::test::snippets::SoftmaxParams>,
                virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::SoftmaxParams> obj);

protected:
    void SetUp() override;
};

class AddSoftmax : public testing::WithParamInterface<ov::test::snippets::AddSoftmaxParams>,
                   virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddSoftmaxParams> obj);

protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov