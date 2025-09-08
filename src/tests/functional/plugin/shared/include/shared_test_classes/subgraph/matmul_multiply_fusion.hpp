// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

struct MatMulMultiplyFusionShapeParams {
    ov::Shape input_shape;
    ov::Shape weights_shape;
    bool trans_b;
    ov::Shape const_shape;
};

typedef std::tuple<MatMulMultiplyFusionShapeParams,
                   bool,        // whether Mul can be fused to MatMul in this case
                   std::string  // Device name
                   >
    MatMulMultiplyFusionParams;

class MatMulMultiplyFusion : public testing::WithParamInterface<MatMulMultiplyFusionParams>,
                             virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulMultiplyFusionParams>& obj);

protected:
    void SetUp() override;
};

class QuantizedMatMulMultiplyFusion : public testing::WithParamInterface<MatMulMultiplyFusionParams>,
                                      virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulMultiplyFusionParams>& obj);

protected:
    void SetUp() override;
    void TearDown() override;
};

}  // namespace test
}  // namespace ov
