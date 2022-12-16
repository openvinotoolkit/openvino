// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/gather_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/gather_function.hpp"

namespace LayerTestsDefinitions {

std::string GatherTransformation::getTestCaseName(const testing::TestParamInfo<GatherTransformationParams>& obj) {
    ngraph::element::Type precision;
    std::string targetDevice;
    GatherTransformationTestValues testValues;
    std::tie(precision, targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.inputShape;

    return result.str();
}

void GatherTransformation::SetUp() {
    ngraph::element::Type precision;
    GatherTransformationTestValues testValues;
    std::tie(precision, targetDevice, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::GatherFunction::getOriginal(
        testValues.inputShape,
        testValues.gatherIndicesShape,
        testValues.gatherIndicesValues,
        testValues.axis,
        testValues.batch_dims,
        testValues.precisionBeforeFq,
        testValues.fqOnData);
}

TEST_P(GatherTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
