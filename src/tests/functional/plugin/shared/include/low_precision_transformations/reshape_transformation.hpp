// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"


namespace LayerTestsDefinitions {
class ReshapeTransformationParam {
public:
    ov::PartialShape inputShape;
    std::vector<int> reshapeConstValues;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::string layerType;
    std::string expectedKernelType;
    std::vector<std::string> executionOrder;
};

typedef std::tuple<
    ov::element::Type,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    ReshapeTransformationParam
> ReshapeTransformationParams;

class ReshapeTransformation :
    public testing::WithParamInterface<ReshapeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReshapeTransformationParams>& obj);

protected:
    void SetUp() override;
    void run() override;
};

}  // namespace LayerTestsDefinitions
