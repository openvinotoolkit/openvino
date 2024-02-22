// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

class MatMulShapes {
public:
    ov::PartialShape inputA;
    ov::PartialShape inputB;
    bool transposeA;
    bool transposeB;
};

typedef std::tuple<
    ov::element::Type,
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
