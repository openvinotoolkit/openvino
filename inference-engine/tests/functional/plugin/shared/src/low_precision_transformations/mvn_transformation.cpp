// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/mvn_transformation.hpp"

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
#include "lpt_ngraph_functions/mvn_function.hpp"

namespace LayerTestsDefinitions {

std::string MVNTransformation::getTestCaseName(testing::TestParamInfo<MVNTransformationParams> obj) {
    std::string targetDevice;
    ngraph::Shape shape;
    ngraph::element::Type precision;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    ngraph::AxisSet reductionAxes;
    bool normalizeVariance;
    std::tie(precision, shape, targetDevice, reductionAxes, normalizeVariance) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, shape, targetDevice, params) <<
        "_" << reductionAxes << "_" << normalizeVariance;
    return result.str();
}

void MVNTransformation::SetUp() {
    ngraph::Shape shape;
    ngraph::element::Type precision;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    ngraph::AxisSet reductionAxes;
    bool normalizeVariance;
    std::tie(precision, shape, targetDevice, reductionAxes, normalizeVariance) = this->GetParam();

    function = ngraph::builder::subgraph::MVNFunction::getOriginal(
        precision,
        shape,
        reductionAxes,
        normalizeVariance);

    validate();
}

void MVNTransformation::validate() {
    ngraph::element::Type precision;
    ngraph::Shape shape;
    std::string targetDevice;
    ngraph::AxisSet reductionAxes;
    bool normalizeVariance;
    std::tie(precision, shape, targetDevice, reductionAxes, normalizeVariance) = this->GetParam();

    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));

    const auto output = transformed->get_output_op(0);
    const auto layer = output->get_input_node_shared_ptr(0);
    const std::string typeName = layer->get_type_name();
    if (normalizeVariance) {
        ASSERT_EQ("MVN", typeName);
    } else {
        ASSERT_EQ("ScaleShiftIE", typeName);
    }
}

TEST_P(MVNTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
