// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {

class GatherTransformationTestValues {
public:
    ngraph::PartialShape inputShape;
    std::vector<int> gatherIndicesValues;
    std::vector<int> axis;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::element::Type precisionBeforeFq;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData;
};

typedef std::tuple<
    ngraph::element::Type,
    std::string,
    GatherTransformationTestValues> GatherTransformationParams;

class GatherTransformation :
    public testing::WithParamInterface<GatherTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
