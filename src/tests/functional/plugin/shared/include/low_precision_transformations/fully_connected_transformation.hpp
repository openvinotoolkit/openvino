// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

class MatMulShapes {
public:
    ngraph::PartialShape inputA;
    ngraph::PartialShape inputB;
    bool transposeA;
    bool transposeB;
};

typedef std::tuple<
    ngraph::element::Type,
    MatMulShapes,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params> FullyConnectedTransformationParams;

namespace LayerTestsDefinitions {

class FullyConnectedTransformation :
    public testing::WithParamInterface<FullyConnectedTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FullyConnectedTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
