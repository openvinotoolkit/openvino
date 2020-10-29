// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_and_avg_pool_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
//#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/avg_pool_function.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeAndAvgPoolTransformation::getTestCaseName(testing::TestParamInfo<FakeQuantizeAndAvgPoolTransformationParams> obj) {
    ngraph::element::Type precision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::tie(precision, inputShapes, targetDevice, params, fakeQuantize) = obj.param;

    return getTestCaseNameByParams(precision, inputShapes, targetDevice, params);
}

void FakeQuantizeAndAvgPoolTransformation::SetUp() {
    threshold = 0.5f;
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::tie(precision, inputShape, targetDevice, params, fakeQuantize) = this->GetParam();

    function = ngraph::builder::subgraph::AvgPoolFunction::getOriginal(
        precision,
        inputShape,
        fakeQuantize);

    ngraph::pass::InitNodeInfo().run_on_function(function);
}

TEST_P(FakeQuantizeAndAvgPoolTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
