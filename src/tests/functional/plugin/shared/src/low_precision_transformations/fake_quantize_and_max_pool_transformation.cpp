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
    auto [precision, inputShapes, device, fakeQuantize] = obj.param;
    return get_test_case_name_by_params(precision, inputShapes, device);
}

void FakeQuantizeAndMaxPoolTransformation::SetUp() {
    auto [precision, inputShape, device, fakeQuantize] = this->GetParam();
    targetDevice = device;
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
