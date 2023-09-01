// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_and_max_pool_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
//#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "lpt_ov_models/common/fake_quantize_on_data.hpp"
#include "lpt_ov_models/max_pool_function.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeAndMaxPoolTransformation::getTestCaseName(const testing::TestParamInfo<FakeQuantizeAndMaxPoolTransformationParams>& obj) {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::tie(precision, inputShapes, targetDevice, params, fakeQuantize) = obj.param;

    return getTestCaseNameByParams(precision, inputShapes, targetDevice, params);
}

void FakeQuantizeAndMaxPoolTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::tie(precision, inputShape, targetDevice, params, fakeQuantize) = this->GetParam();

    function = ov::builder::subgraph::MaxPoolFunction::getOriginal(
        precision,
        inputShape,
        fakeQuantize);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(FakeQuantizeAndMaxPoolTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
