// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include <ngraph/shape.hpp>

namespace SubgraphTestsDefinitions {

struct MatMulConstTransposesExtractionTestShapeParams {
    ngraph::Shape input_shape;
    ngraph::Shape weights_shape;
    bool trans_b;
};

typedef std::tuple<
        MatMulConstTransposesExtractionTestShapeParams,
        bool,                       // whether Mul can be fused to MatMul in this case
        std::string                 // Device name
        > MatMulConstTransposesExtractionTestParams;

class MatMulConstTransposesExtractionTest
        : public testing::WithParamInterface<MatMulConstTransposesExtractionTestParams>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulConstTransposesExtractionTestParams> &obj);

protected:
    void SetUp() override;
};

class QuantizedMatMulConstTransposesExtractionTest
        : public testing::WithParamInterface<MatMulConstTransposesExtractionTestParams>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulConstTransposesExtractionTestParams> &obj);

protected:
    void SetUp() override;
    void TearDown() override;
};

} // namespace SubgraphTestsDefinitions
