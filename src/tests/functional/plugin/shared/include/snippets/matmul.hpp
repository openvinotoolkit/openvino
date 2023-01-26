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
        ov::element::Type,             // Element type
        size_t,                        // Expected num nodes
        size_t,                        // Expected num subgraphs
        std::string                    // Target Device
> MatMulParams;

typedef std::tuple<
        std::vector<ov::PartialShape>, // Input  Shapes
        size_t ,                       // Transpose position
        ov::element::Type,             // Element type
        size_t,                        // Expected num nodes
        size_t,                        // Expected num subgraphs
        std::string                    // Target Device
> TransposeMatMulParams;

class MatMul : public testing::WithParamInterface<ov::test::snippets::MatMulParams>,
            virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::MatMulParams> obj);

protected:
    void SetUp() override;
};

class MatMulBias : public MatMul {
protected:
    void SetUp() override;
};

class ExplicitTransposeMatMul : public MatMul {
protected:
    void SetUp() override;
};

class ExplicitTransposeMatMulBias : public MatMul {
protected:
    void SetUp() override;
};

class ExplicitTransposeMulMatMulBias : public MatMul {
protected:
    void SetUp() override;
};

class TransposeMatMulTest : public testing::WithParamInterface<ov::test::snippets::TransposeMatMulParams>,
                        virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeMatMulParams> obj);

protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov