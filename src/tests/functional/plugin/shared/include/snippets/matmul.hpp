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
        std::vector<InputShape>,       // Input  Shapes
        std::vector<ov::element::Type>,// Input Element types
        MatMulType,
        size_t,                        // Expected num nodes
        size_t,                        // Expected num subgraphs
        std::string                    // Target Device
> MatMulParams;

class MatMulBase : public SnippetsTestsCommon {
protected:
    /**
     * @brief Erases shapes with the given indices from inputDynamicShapes and targetStaticShapes
     */
    void filter_shape_info(const std::set<size_t>& idces_to_remove);
    virtual std::shared_ptr<MatMulFunctionBase> get_builder(const std::vector<ov::element::Type>& types) = 0;

    MatMulType matmul_type;
};

class MatMul : public testing::WithParamInterface<ov::test::snippets::MatMulParams>,
               virtual public MatMulBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::MatMulParams> obj);

protected:
    void SetUp() override;
    std::shared_ptr<MatMulFunctionBase> get_builder(const std::vector<ov::element::Type>& types) override;
};

class MatMulTransposeB : public MatMul {
protected:
    std::shared_ptr<MatMulFunctionBase> get_builder(const std::vector<ov::element::Type>& types) override;
};

class MatMulFQ : public MatMul {
protected:
    std::shared_ptr<MatMulFunctionBase> get_builder(const std::vector<ov::element::Type>& types) override;
};

class MatMulBias : public MatMul {
protected:
    std::shared_ptr<MatMulFunctionBase> get_builder(const std::vector<ov::element::Type>& types) override;
};

class MatMulBiasQuantized : public MatMul {
protected:
    std::shared_ptr<MatMulFunctionBase> get_builder(const std::vector<ov::element::Type>& types) override;
};

class MatMulsQuantized : public MatMul {
protected:
    std::shared_ptr<MatMulFunctionBase> get_builder(const std::vector<ov::element::Type>& types) override;
};

class MatMulsQuantizedSoftmax : public MatMul {
protected:
    std::shared_ptr<MatMulFunctionBase> get_builder(const std::vector<ov::element::Type>& types) override;
};

class MatMulEltwiseChain : public MatMul {
protected:
    std::shared_ptr<MatMulFunctionBase> get_builder(const std::vector<ov::element::Type>& types) override;
};

class MatMulEltwiseChainCascade : public MatMul {
protected:
    std::shared_ptr<MatMulFunctionBase> get_builder(const std::vector<ov::element::Type>& types) override;
};

} // namespace snippets
} // namespace test
} // namespace ov