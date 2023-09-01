// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ov_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class MultiplyTestValues {
public:
    bool broadcast1;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    bool broadcast2;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeAfter;
    ngraph::element::Type expectedPrecisions;
    bool secondInputIsConstant;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    MultiplyTestValues
> MultiplyTransformationParams;

class MultiplyTransformation :
    public testing::WithParamInterface<MultiplyTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MultiplyTransformationParams>& obj);

protected:
    void SetUp() override;
    void Run() override;
};

}  // namespace LayerTestsDefinitions
