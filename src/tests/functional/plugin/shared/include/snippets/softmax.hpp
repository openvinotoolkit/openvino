// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {


typedef std::tuple<
        std::vector<InputShape>, // Input Shapes
        int,                     // Axis
        size_t,                  // Expected num nodes
        size_t,                  // Expected num subgraphs
        std::string              // Target Device
> SoftmaxParams;

class SoftmaxBase : public testing::WithParamInterface<ov::test::snippets::SoftmaxParams>,
                virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::test::snippets::SoftmaxParams>& obj);

protected:
    virtual std::shared_ptr<SnippetsFunctionBase> get_subgraph(const std::vector<PartialShape>& inputShapes,
                                                               int axis) const = 0;
    void SetUp() override;
};

class Softmax : public SoftmaxBase {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph(const std::vector<PartialShape>& inputShapes,
                                                       int axis) const override;
};

class AddSoftmax : public SoftmaxBase {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph(const std::vector<PartialShape>& inputShapes,
                                                       int axis) const override;
};

class SoftmaxSum : public SoftmaxBase {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph(const std::vector<PartialShape>& inputShapes,
                                                       int axis) const override;
};

} // namespace snippets
} // namespace test
} // namespace ov
