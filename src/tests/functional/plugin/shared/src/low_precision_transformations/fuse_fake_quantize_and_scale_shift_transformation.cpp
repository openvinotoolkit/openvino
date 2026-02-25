// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_fake_quantize_and_scale_shift_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"

namespace LayerTestsDefinitions {

std::string FuseFakeQuantizeAndScaleShiftTransformation::getTestCaseName(const testing::TestParamInfo<FuseFakeQuantizeAndScaleShiftTransformationParams>& obj) {
    auto [netPrecision, inputShape, device, fakeQuantizeOnData] = obj.param;
    std::ostringstream result;
    result << netPrecision << "_" << device << "_" << fakeQuantizeOnData;
    return result.str();
}

void FuseFakeQuantizeAndScaleShiftTransformation::SetUp() {
    auto [netPrecision, inputShape, device, fakeQuantizeOnData] = this->GetParam();
    targetDevice = device;

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::FuseFakeQuantizeAndScaleShiftFunction::getOriginal(
        netPrecision,
        inputShape,
        fakeQuantizeOnData);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(FuseFakeQuantizeAndScaleShiftTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
