// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/mat_mul_with_optimized_constant_fake_quantize_transformation.hpp"

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
#include "lpt_ngraph_functions/mat_mul_with_optimized_constant_fake_quantize_function.hpp"

namespace LayerTestsDefinitions {

std::string MatMulWithOptimizedConstantFakeQuantizeTransformation::getTestCaseName(
    testing::TestParamInfo<MatMulWithOptimizedConstantFakeQuantizeTransformationTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    MatMulWithOptimizedConstantFakeQuantizeTransformationTestValues param;

    std::tie(netPrecision, shapes, targetDevice, param) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" <<
        CommonTestUtils::vec2str(shapes.first) << "_" << CommonTestUtils::vec2str(shapes.second) << "_" <<
        targetDevice << "_"  <<
        param.fqOnData << "_" <<
        param.fqOnWeights;
    return result.str();
}

void MatMulWithOptimizedConstantFakeQuantizeTransformation::SetUp() {
    threshold = 0.01f;

    ngraph::element::Type precision;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    MatMulWithOptimizedConstantFakeQuantizeTransformationTestValues param;
    std::tie(precision, shapes, targetDevice, param) = this->GetParam();

    function = ngraph::builder::subgraph::MatMulWithOptimizedConstantFakeQuantizeFunction::getOriginal(
        precision,
        shapes.first,
        shapes.second,
        param.fqOnData,
        param.fqOnWeights);

    validate();
}

void MatMulWithOptimizedConstantFakeQuantizeTransformation::validate() {
    ngraph::element::Type precision;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    MatMulWithOptimizedConstantFakeQuantizeTransformationTestValues param;
    std::tie(precision, shapes, targetDevice, param) = this->GetParam();

    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));

    const auto output = transformed->get_output_op(0);
    const auto scaleShift = output->get_input_node_shared_ptr(0);
    const std::string typeName = scaleShift->get_type_name();
    ASSERT_EQ("ScaleShiftIE", typeName);
}

TEST_P(MatMulWithOptimizedConstantFakeQuantizeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
