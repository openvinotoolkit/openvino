// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

class FullyConnectedShapes {
public:
    ov::PartialShape inputA;
    ov::PartialShape inputB;
    bool transposeA;
    bool transposeB;
};

class FullyConnectedParams {
public:
    bool activation;
    bool perChannelWeights;
    bool fq;
    bool bias;
    std::string originalLayersNames;
};

typedef std::tuple<
    ov::element::Type,
    FullyConnectedShapes,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    ov::element::Type,
    FullyConnectedParams,
    std::string> FullyConnectedTransformationParams;

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
