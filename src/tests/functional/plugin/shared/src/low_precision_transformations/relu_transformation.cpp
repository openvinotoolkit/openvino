// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/relu_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/relu.hpp"

namespace LayerTestsDefinitions {

std::string ReluTransformation::getTestCaseName(const testing::TestParamInfo<ReluTransformationParams>& obj) {
    ov::element::Type precision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ReluTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.fakeQuantize;

    return result.str();
}


void ReluTransformation::SetUp() {
    ov::element::Type precision;
    ov::PartialShape inputShape;
    ReluTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::ReluFunction::getOriginal(inputShape, precision, testValues.fakeQuantize);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(ReluTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
