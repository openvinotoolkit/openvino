// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/mvn_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"

#include "ov_lpt_models/mvn.hpp"

namespace LayerTestsDefinitions {

std::string MVNTransformation::getTestCaseName(const testing::TestParamInfo<MVNTransformationParams>& obj) {
    std::string targetDevice;
    ov::PartialShape shape;
    ov::element::Type precision;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    ov::AxisSet reductionAxes;
    bool normalizeVariance;
    std::tie(precision, shape, targetDevice, reductionAxes, normalizeVariance) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(precision, shape, targetDevice, params) <<
           "_" << reductionAxes << "_" << normalizeVariance;
    return result.str();
}

void MVNTransformation::SetUp() {
    ov::PartialShape shape;
    ov::element::Type precision;
    ov::AxisSet reductionAxes;
    bool normalizeVariance;
    std::tie(precision, shape, targetDevice, reductionAxes, normalizeVariance) = this->GetParam();

    init_input_shapes(shape);

    function = ov::builder::subgraph::MVNFunction::getOriginal(
        precision,
        shape,
        reductionAxes,
        normalizeVariance);
}

TEST_P(MVNTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
