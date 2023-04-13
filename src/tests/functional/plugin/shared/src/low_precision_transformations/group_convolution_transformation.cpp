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
#include "ngraph_functions/pass/convert_prc.hpp"
#include "lpt_ngraph_functions/group_convolution_function.hpp"

namespace LayerTestsDefinitions {
namespace {
std::shared_ptr<ov::Model> replaceFQConstantsOnScalars(const std::shared_ptr<ov::Model>& m) {
    auto model = ngraph::clone_function(*m);
    for (const auto& op : model->get_ordered_ops()) {
        if (!ov::is_type<ov::opset10::FakeQuantize>(op))
            continue;
        for (size_t i = 1; i < op->get_input_size(); ++i) {
            if (auto constant = ov::as_type_ptr<ov::opset10::Constant>(op->get_input_node_shared_ptr(i))) {
                const auto& shape = constant->get_shape();
                if (ov::shape_size(shape) > 1) {
                    const auto values = constant->cast_vector<float>();
                    if (std::all_of(values.begin(), values.end(), [&](float x) { return x == values[0]; })) {
                        const auto scalar_constant = ov::opset10::Constant::create(constant->get_element_type(), {}, &values[0]);
                        ov::replace_node(constant, scalar_constant);
                    }
                }
            }
        }
    }
    return model;
}
}  // namespace

std::string GroupConvolutionTransformation::getTestCaseName(const testing::TestParamInfo<GroupConvolutionTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
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
    ngraph::pass::low_precision::LayerTransformation::Params params;
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
    // WA for issue #107400
    functionRefs = replaceFQConstantsOnScalars(function);
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
