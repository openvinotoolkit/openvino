// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/group_convolution_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "ov_lpt_models/group_convolution.hpp"

namespace LayerTestsDefinitions {
std::string GroupConvolutionTransformation::getTestCaseName(const testing::TestParamInfo<GroupConvolutionTransformationParams>& obj) {
    auto [netPrecision, device, inputShapes, param, addPrecisionPreserved] = obj.param;

    std::ostringstream result;
    result <<
           get_test_case_name_by_params(netPrecision, inputShapes.first, device) << "_" <<
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
    auto [netPrecision, device, inputShapes, param, addPrecisionPreserved] = this->GetParam();
    targetDevice = device;

    init_input_shapes(inputShapes.first);

    while (param.fakeQuantizeOnData.constantShape.size() > inputShapes.first.size()) {
        param.fakeQuantizeOnData.constantShape.pop_back();
    }
    function = ov::builder::subgraph::GroupConvolutionFunction::getOriginal(
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

void GroupConvolutionTransformation::run() {
    LayerTransformation::run();

    const auto param = std::get<3>(GetParam());
    if (!param.layerName.empty()) {
        const auto actualPrecision = get_runtime_precision_by_type(param.layerName);
        auto expectedPrecision = param.expectedKernelType;
        if (expectedPrecision == "f32" && std::get<0>(GetParam()) == ov::element::f16) {
            expectedPrecision = "f16";
        }
        EXPECT_EQ(actualPrecision, expectedPrecision);
    }
}

TEST_P(GroupConvolutionTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
