// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/gather_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/gather.hpp"

namespace LayerTestsDefinitions {

std::string GatherTransformation::getTestCaseName(const testing::TestParamInfo<GatherTransformationParams>& obj) {
    ov::element::Type precision;
    std::string targetDevice;
    GatherTransformationTestValues testValues;
    int opset_version;
    std::tie(precision, targetDevice, testValues, opset_version) = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.inputShape << "_" <<
        opset_version;

    return result.str();
}

void GatherTransformation::SetUp() {
    ov::element::Type precision;
    GatherTransformationTestValues testValues;
    int opset_version;
    std::tie(precision, targetDevice, testValues, opset_version) = this->GetParam();

    init_input_shapes(testValues.inputShape);

    function = ov::builder::subgraph::GatherFunction::getOriginal(
        testValues.inputShape,
        testValues.gatherIndicesShape,
        testValues.gatherIndicesValues,
        testValues.axis,
        testValues.batch_dims,
        testValues.precisionBeforeFq,
        testValues.fqOnData,
        opset_version);
}

TEST_P(GatherTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
