// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_and_max_pool_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/max_pool.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeAndMaxPoolTransformation::getTestCaseName(const testing::TestParamInfo<FakeQuantizeAndMaxPoolTransformationParams>& obj) {
    ov::element::Type precision;
    ov::PartialShape inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::tie(precision, inputShapes, targetDevice, params, fakeQuantize) = obj.param;

    return get_test_case_name_by_params(precision, inputShapes, targetDevice, params);
}

void FakeQuantizeAndMaxPoolTransformation::SetUp() {
    ov::element::Type precision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::tie(precision, inputShape, targetDevice, params, fakeQuantize) = this->GetParam();

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::MaxPoolFunction::getOriginal(
        precision,
        inputShape,
        fakeQuantize);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(FakeQuantizeAndMaxPoolTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
