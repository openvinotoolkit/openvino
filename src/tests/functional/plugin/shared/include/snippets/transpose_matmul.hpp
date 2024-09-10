// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"
#include "snippets/matmul.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        std::vector<InputShape>, // Input  Shapes
        size_t ,                       // Transpose position
        std::vector<ov::element::Type>,// Input Element types
        MatMulType,
        size_t,                        // Expected num nodes
        size_t,                        // Expected num subgraphs
        std::string                    // Target Device
> TransposeMatMulParams;

class TransposeMatMul : public testing::WithParamInterface<ov::test::snippets::TransposeMatMulParams>,
                        virtual public MatMulBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeMatMulParams> obj);

protected:
    void SetUp() override;
    void init_subgraph(const std::vector<ov::element::Type>& types) override;

    size_t transpose_position;
};

class TransposeMatMulFQ : public TransposeMatMul {
protected:
    void SetUp() override {
        TransposeMatMul::SetUp();
        abs_threshold = 5e-6;
    }
    void init_subgraph(const std::vector<ov::element::Type>& types) override;
};

class ExplicitTransposeMatMul : public TransposeMatMul {
protected:
    void init_subgraph(const std::vector<ov::element::Type>& types) override;
};

class ExplicitTransposeMatMulBias : public TransposeMatMul {
protected:
    void init_subgraph(const std::vector<ov::element::Type>& types) override;
};

} // namespace snippets
} // namespace test
} // namespace ov