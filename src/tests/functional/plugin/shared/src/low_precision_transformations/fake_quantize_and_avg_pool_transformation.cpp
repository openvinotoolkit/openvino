// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_and_avg_pool_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
//#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/avg_pool.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeAndAvgPoolTransformation::getTestCaseName(const testing::TestParamInfo<FakeQuantizeAndAvgPoolTransformationParams>& obj) {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::tie(precision, inputShapes, targetDevice, params, fakeQuantize) = obj.param;

    return getTestCaseNameByParams(precision, inputShapes, targetDevice, params);
}

void FakeQuantizeAndAvgPoolTransformation::SetUp() {
    threshold = 0.5f;
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::tie(precision, inputShape, targetDevice, params, fakeQuantize) = this->GetParam();

    function = ngraph::builder::subgraph::AvgPoolFunction::getOriginal(
        precision,
        inputShape,
        fakeQuantize);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(FakeQuantizeAndAvgPoolTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
