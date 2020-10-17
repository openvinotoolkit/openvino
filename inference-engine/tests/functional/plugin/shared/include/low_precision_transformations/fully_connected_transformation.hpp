// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

class MatMulShapes {
public:
    ngraph::Shape inputA;
    ngraph::Shape inputB;
    bool transposeA;
    bool transposeB;
};

typedef std::tuple<
    ngraph::element::Type,
    MatMulShapes,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params> FullyConnectedTransformationParams;

namespace LayerTestsDefinitions {

class FullyConnectedTransformation :
    public testing::WithParamInterface<FullyConnectedTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FullyConnectedTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
