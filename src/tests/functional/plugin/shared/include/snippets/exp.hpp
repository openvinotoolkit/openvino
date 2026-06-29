// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        InputShape,                  // Input 0 Shape
        ov::element::Type,           // Element type
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> UnaryActivationParams;

class UnaryActivation : public testing::WithParamInterface<ov::test::snippets::UnaryActivationParams>,
                        virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::test::snippets::UnaryActivationParams>& obj);

protected:
    void SetUp() override;

    virtual std::shared_ptr<SnippetsFunctionBase> get_subgraph(const std::vector<PartialShape>& inputShapes) const = 0;
};

class Exp : public UnaryActivation {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph(const std::vector<PartialShape>& inputShapes) const override;
};

class ExpReciprocal : public UnaryActivation {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph(const std::vector<PartialShape>& inputShapes) const override;
};

class HSigmoid : public UnaryActivation {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph(const std::vector<PartialShape>& inputShapes) const override;
};

class SoftSign : public UnaryActivation {
protected:
    std::shared_ptr<SnippetsFunctionBase> get_subgraph(const std::vector<PartialShape>& inputShapes) const override;
};

} // namespace snippets
} // namespace test
} // namespace ov
