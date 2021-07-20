// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    ngraph::builder::subgraph::FakeQuantizeOnData> FakeQuantizeAndMaxPoolTransformationParams;

class FakeQuantizeAndMaxPoolTransformation :
    public testing::WithParamInterface<FakeQuantizeAndMaxPoolTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeAndMaxPoolTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
