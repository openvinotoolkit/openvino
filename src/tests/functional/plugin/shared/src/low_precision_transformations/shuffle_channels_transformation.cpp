// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/shuffle_channels_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "ov_lpt_models/shuffle_channels.hpp"

namespace LayerTestsDefinitions {

std::string ShuffleChannelsTransformation::getTestCaseName(const testing::TestParamInfo<ShuffleChannelsTransformationParams>& obj) {
    auto [netPrecision, inputShape, device, param] = obj.param;
    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, device) << "_" <<
           param.fakeQuantizeOnData << "_axis_" << param.axis << "_group_" << param.group;
    return result.str();
}

void ShuffleChannelsTransformation::SetUp() {
    auto [netPrecision, inputShape, device, param] = this->GetParam();
    targetDevice = device;

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::ShuffleChannelsFunction::getOriginal(
        netPrecision,
        inputShape,
        param.fakeQuantizeOnData,
        param.axis,
        param.group);
}

void ShuffleChannelsTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<3>(GetParam());
    const auto actualType = get_runtime_precision(params.layerName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

TEST_P(ShuffleChannelsTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
