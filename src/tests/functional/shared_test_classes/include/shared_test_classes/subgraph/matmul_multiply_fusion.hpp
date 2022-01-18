// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include <ngraph/shape.hpp>

namespace SubgraphTestsDefinitions {

struct MatMulMultiplyFusionShapeParams {
    ngraph::Shape input_shape;
    ngraph::Shape weights_shape;
    bool trans_b;
    ngraph::Shape const_shape;
};

typedef std::tuple<
        MatMulMultiplyFusionShapeParams,
        bool,                       // whether Mul can be fused to MatMul in this case
        std::string                 // Device name
        > MatMulMultiplyFusionParams;

class MatMulMultiplyFusion
        : public testing::WithParamInterface<MatMulMultiplyFusionParams>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulMultiplyFusionParams> &obj);

protected:
    void SetUp() override;
};

class QuantizedMatMulMultiplyFusion
        : public testing::WithParamInterface<MatMulMultiplyFusionParams>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulMultiplyFusionParams> &obj);

protected:
    void SetUp() override;
    void TearDown() override;
};

} // namespace SubgraphTestsDefinitions
