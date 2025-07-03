// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

struct MatMulConstTransposesExtractionTestShapeParams {
    ov::Shape input_shape;
    ov::Shape weights_shape;
    bool trans_b;
};

typedef std::tuple<MatMulConstTransposesExtractionTestShapeParams,
                   bool,        // whether Mul can be fused to MatMul in this case
                   std::string  // Device name
                   >
    MatMulConstTransposesExtractionTestParams;

class MatMulConstTransposesExtractionTest
    : public testing::WithParamInterface<MatMulConstTransposesExtractionTestParams>,
      virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulConstTransposesExtractionTestParams>& obj);

protected:
    void SetUp() override;
};

class QuantizedMatMulConstTransposesExtractionTest
    : public testing::WithParamInterface<MatMulConstTransposesExtractionTestParams>,
      virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulConstTransposesExtractionTestParams>& obj);

protected:
    void SetUp() override;
    void TearDown() override;
};

}  // namespace test
}  // namespace ov
