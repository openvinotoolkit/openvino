// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/gemm_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/builders.hpp"

#include "lpt_ngraph_functions/mat_mul_function.hpp"

namespace LayerTestsDefinitions {

std::string GemmTransformation::getTestCaseName(testing::TestParamInfo<GemmTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = obj.param;

    return getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params);
}

void GemmTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    const float low = params.precisionsOnActivations[0] == ngraph::element::u8 ? 0.f : -128.f;
    const float high = params.precisionsOnActivations[0] == ngraph::element::u8 ? 255.f : 127.f;

    function = ngraph::builder::subgraph::MatMulFunction::getOriginal(
        netPrecision,
        inputShape,
        low,
        high);

    validate();
}

void GemmTransformation::validate() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));
    EXPECT_EQ(1ul, transformed->get_output_size());

    const auto output = transformed->get_output_op(0);
    const auto scaleShift = output->get_input_node_shared_ptr(0);
    const std::string typeName = scaleShift->get_type_name();
    ASSERT_EQ("ScaleShiftIE", typeName);
}

TEST_P(GemmTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
