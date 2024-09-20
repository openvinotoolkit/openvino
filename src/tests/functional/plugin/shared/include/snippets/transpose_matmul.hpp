// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        std::vector<InputShape>, // Input  Shapes
        size_t ,                       // Transpose position
        std::vector<ov::element::Type>,// Input Element types
        size_t,                        // Expected num nodes
        size_t,                        // Expected num subgraphs
        std::string                    // Target Device
> TransposeMatMulParams;

class TransposeMatMul : public testing::WithParamInterface<ov::test::snippets::TransposeMatMulParams>,
               virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeMatMulParams> obj);

protected:
    void SetUp() override;
};

class TransposeMatMulFQ : public TransposeMatMul {
protected:
    void SetUp() override;
};

class ExplicitTransposeMatMul : public TransposeMatMul {
protected:
    void SetUp() override;
};

class ExplicitTransposeMatMulBias : public TransposeMatMul {
protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov