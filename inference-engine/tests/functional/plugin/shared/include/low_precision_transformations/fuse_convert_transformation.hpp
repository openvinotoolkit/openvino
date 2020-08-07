// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {

class FuseConvertTransformationTestValues {
public:
    ngraph::Shape inputShape;
    std::vector<int> transposeConstValues;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::element::Type precisionBeforeFq;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
};

typedef std::tuple<
    ngraph::element::Type,
    std::string,
    LayerTestsUtils::LayerTransformation::LptVersion,
    FuseConvertTransformationTestValues> FuseConvertTransformationParams;

class FuseConvertTransformation :
    public testing::WithParamInterface<FuseConvertTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseConvertTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
