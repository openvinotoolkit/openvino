// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_and_avg_pool_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/avg_pool.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeAndAvgPoolTransformation::getTestCaseName(const testing::TestParamInfo<FakeQuantizeAndAvgPoolTransformationParams>& obj) {
    auto [precision, inputShapes, device, fakeQuantize] = obj.param;
    return get_test_case_name_by_params(precision, inputShapes, device);
}

void FakeQuantizeAndAvgPoolTransformation::SetUp() {
    auto [precision, inputShape, device, fakeQuantize] = this->GetParam();
    targetDevice = device;

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::AvgPoolFunction::getOriginal(
        precision,
        inputShape,
        fakeQuantize);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(FakeQuantizeAndAvgPoolTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
