// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {
class ReduceMinTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::vector<int64_t> constantValues;
    bool keepDims;
    std::string layerName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    ReduceMinTransformationParam
> ReduceMinTransformationParams;

class ReduceMinTransformation :
    public testing::WithParamInterface<ReduceMinTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReduceMinTransformationParams>& obj);

protected:
    void SetUp() override;
    void Run() override;
};
}  // namespace LayerTestsDefinitions
