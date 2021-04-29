// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/shuffle_channels_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "lpt_ngraph_functions/shuffle_channels_function.hpp"

namespace LayerTestsDefinitions {

std::string ShuffleChannelsTransformation::getTestCaseName(testing::TestParamInfo<ShuffleChannelsTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ShuffleChannelsTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" <<
        param.fakeQuantizeOnData << "_axis_" << param.axis << "_group_" << param.group;
    return result.str();
}

void ShuffleChannelsTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ShuffleChannelsTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::ShuffleChannelsFunction::getOriginal(
        netPrecision,
        inputShape,
        param.fakeQuantizeOnData,
        param.axis,
        param.group);
}

void ShuffleChannelsTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<4>(GetParam());
    const auto actualType = getRuntimePrecision(params.layerName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

TEST_P(ShuffleChannelsTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
