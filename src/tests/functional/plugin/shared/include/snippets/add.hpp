// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        ov::Shape,                   // Input 0 Shape
        ov::Shape,                   // Input 1 Shape
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> AddParams;

typedef std::tuple<
        ov::Shape,                   // Input 0 Shape
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> AddConstParams;

typedef std::tuple<
    InputShape,        // Input 0 Shape
    InputShape,        // Input 1 Shape
    size_t,                      // Expected num nodes
    size_t,                      // Expected num subgraphs
    std::string                  // Target Device
    > AddDynamicParams;

class Add : public testing::WithParamInterface<ov::test::snippets::AddParams>,
            virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddParams> obj);

protected:
    void SetUp() override;
};

class AddSinh : public Add {
protected:
    void SetUp() override;
};

class AddSinhConst : public testing::WithParamInterface<ov::test::snippets::AddConstParams>,
                     virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddConstParams> obj);
protected:
    void SetUp() override;
};

class AddSinhDynamic : public testing::WithParamInterface<ov::test::snippets::AddDynamicParams>,
            virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddDynamicParams> obj);

protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov