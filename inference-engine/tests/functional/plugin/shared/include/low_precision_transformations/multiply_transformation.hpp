// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class MultiplyTestValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    bool broadcast;
    std::vector<ngraph::element::Type> precisionOnActivations;
    std::vector<ngraph::element::Type> expectedPrecisions;
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
    static std::string getTestCaseName(testing::TestParamInfo<MultiplyTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
