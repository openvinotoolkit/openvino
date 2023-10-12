// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fully_connected_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/builders.hpp"
#include "ov_lpt_models/mat_mul.hpp"

namespace LayerTestsDefinitions {

std::string FullyConnectedTransformation::getTestCaseName(const testing::TestParamInfo<FullyConnectedTransformationParams>& obj) {
    ngraph::element::Type precision;
    MatMulShapes shapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(precision, shapes, targetDevice, params) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(precision, shapes.inputA, targetDevice, params) <<
        shapes.inputB << "_" <<
        shapes.transposeA << "_" <<
        shapes.transposeB;

    return result.str();
}

void FullyConnectedTransformation::SetUp() {
    ngraph::element::Type precision;
    MatMulShapes shapes;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(precision, shapes, targetDevice, params) = this->GetParam();

    function = ngraph::builder::subgraph::MatMulFunction::getOriginal(
        precision,
        shapes.inputA,
        shapes.inputB,
        shapes.transposeA,
        shapes.transposeB);
}

TEST_P(FullyConnectedTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
