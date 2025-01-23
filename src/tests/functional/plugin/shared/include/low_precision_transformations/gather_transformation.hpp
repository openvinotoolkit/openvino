// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {

class GatherTransformationTestValues {
public:
    ov::PartialShape inputShape;
    std::vector<size_t> gatherIndicesShape;
    std::vector<int> gatherIndicesValues;
    std::vector<int> axis;
    int64_t batch_dims;
    ov::pass::low_precision::LayerTransformation::Params params;
    ov::element::Type precisionBeforeFq;
    ov::builder::subgraph::FakeQuantizeOnData fqOnData;
};

typedef std::tuple<
    ov::element::Type,
    std::string,
    GatherTransformationTestValues,
    int> GatherTransformationParams;

class GatherTransformation :
    public testing::WithParamInterface<GatherTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
