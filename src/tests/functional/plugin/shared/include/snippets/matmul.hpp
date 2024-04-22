// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        std::vector<ov::PartialShape>, // Input  Shapes
        std::vector<ov::element::Type>,// Input Element types
        size_t,                        // Expected num nodes
        size_t,                        // Expected num subgraphs
        std::string                    // Target Device
> MatMulParams;

class MatMul : public testing::WithParamInterface<ov::test::snippets::MatMulParams>,
               virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::MatMulParams> obj);

protected:
    void SetUp() override;

    virtual void init_subgraph(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& types);
};

class MatMulFQ : public MatMul {
protected:
    void init_subgraph(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& types) override;
};

class MatMulBias : public MatMul {
protected:
    void init_subgraph(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& types) override;
};

class MatMulBiasQuantized : public MatMul {
protected:
    void init_subgraph(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& types) override;
};

class MatMulQuantized : public MatMul {
protected:
    void init_subgraph(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& types) override;
};

class MatMulQuantizedSoftmax : public MatMul {
protected:
    void init_subgraph(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& types) override;
};

} // namespace snippets
} // namespace test
} // namespace ov