// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/group_convolution_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_lpt_models/group_convolution.hpp"

namespace LayerTestsDefinitions {
std::string GroupConvolutionTransformation::getTestCaseName(const testing::TestParamInfo<GroupConvolutionTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::pair<ngraph::PartialShape, ngraph::Shape> inputShapes;
    GroupConvolutionTransformationParam param;
    bool addPrecisionPreserved;
    std::tie(netPrecision, targetDevice, params, inputShapes, param, addPrecisionPreserved) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, inputShapes.first, targetDevice, params) << "_" <<
        inputShapes.first.rank().get_length() << "D_" <<
        inputShapes.first << "_" <<
        inputShapes.second << "_" <<
        param.group << "_" <<
        param.groupCalculationDimention << "_" <<
        param.fakeQuantizeOnData << "_" <<
        (param.addReshape ? "reshape_on_weights_" : "wo_reshape_") <<
        (addPrecisionPreserved ? "max_pool_" : "") <<
        param.fakeQuantizeOnWeights;
    return result.str();
}

void GroupConvolutionTransformation::SetUp() {
    threshold = 0.1f;

    ngraph::element::Type netPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::pair<ngraph::PartialShape, ngraph::Shape> inputShapes;
    GroupConvolutionTransformationParam param;
    bool addPrecisionPreserved;
    std::tie(netPrecision, targetDevice, params, inputShapes, param, addPrecisionPreserved) = this->GetParam();

    while (param.fakeQuantizeOnData.constantShape.size() > inputShapes.first.size()) {
        param.fakeQuantizeOnData.constantShape.pop_back();
    }
    function = ngraph::builder::subgraph::GroupConvolutionFunction::getOriginal(
        netPrecision,
        inputShapes.first,
        inputShapes.second,
        param.group,
        param.groupCalculationDimention,
        param.fakeQuantizeOnData,
        param.fakeQuantizeOnWeights,
        param.addReshape,
        addPrecisionPreserved);
}

void GroupConvolutionTransformation::Run() {
    LayerTestsCommon::Run();

    const auto param = std::get<4>(GetParam());
    if (!param.layerName.empty()) {
        const auto actualPrecision = getRuntimePrecisionByType(param.layerName);
        auto expectedPrecision = param.expectedKernelType;
        if (expectedPrecision == "FP32" && std::get<0>(GetParam()) == ngraph::element::f16) {
            expectedPrecision = "FP16";
        }
        EXPECT_EQ(actualPrecision, expectedPrecision);
    }
}

TEST_P(GroupConvolutionTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
};

}  // namespace LayerTestsDefinitions
