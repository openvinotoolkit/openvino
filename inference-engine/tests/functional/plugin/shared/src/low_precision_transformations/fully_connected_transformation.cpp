// Copyright (C) 2019 Intel Corporation
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
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/builders.hpp"
#include "lpt_ngraph_functions/mat_mul_function.hpp"

namespace LayerTestsDefinitions {

std::string FullyConnectedTransformation::getTestCaseName(testing::TestParamInfo<FullyConnectedTransformationParams> obj) {
    ngraph::element::Type precision;
    MatMulShapes shapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
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
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(precision, shapes, targetDevice, params) = this->GetParam();

    function = ngraph::builder::subgraph::MatMulFunction::getOriginal(
        precision,
        shapes.inputA,
        shapes.inputB,
        shapes.transposeA,
        shapes.transposeB);

    validate();
}

void FullyConnectedTransformation::validate() {
    ngraph::element::Type precision;
    MatMulShapes shapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(precision, shapes, targetDevice, params) = this->GetParam();

    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));

    const auto output = transformed->get_output_op(0);
    const auto scaleShift = output->get_input_node_shared_ptr(0);
    const std::string typeName = scaleShift->get_type_name();
    ASSERT_EQ("ScaleShiftIE", typeName);
}

TEST_P(FullyConnectedTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
