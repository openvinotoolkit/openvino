// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/split_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "low_precision/split.hpp"
#include "lpt_ngraph_functions/split_function.hpp"

namespace LayerTestsDefinitions {
std::string SplitTransformation::getTestCaseName(const testing::TestParamInfo<SplitTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape  inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    SplitTransformationParam param;
    std::tie(netPrecision, inputShapes, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) << "_" <<
        param.fakeQuantize << "_axis=" << param.splitedAxis << "_n_splits=" << param.numSplit;
    return result.str();
}

ov::test::utils::InputsMap SplitTransformation::get_input_map() {
    auto generate_default = [this](const std::shared_ptr<ngraph::Node>&node,
                               size_t port,
                               const ov::element::Type & elemType,
                               const ov::Shape & targetShape) -> ov::runtime::Tensor {
        ngraph::element::Type precision;
        ngraph::PartialShape inputShape;
        std::string targetDevice;
        ngraph::pass::low_precision::LayerTransformation::Params params;
        SplitTransformationParam param;
        std::tie(precision, inputShape, targetDevice, params, param) = this->GetParam();
        const auto& fqOnData = param.fakeQuantize;

        const auto range = static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]);
        const double start_from = fqOnData.empty() ? -12.5 : fqOnData.outputLowValues[0];
        return ov::test::utils::create_and_fill_tensor(elemType, targetShape, range, start_from);
    };

    static ov::test::utils::InputsMap inputs_map{
        { ov::op::Op::get_type_info_static(), generate_default }
    };
    return inputs_map;
}

void SplitTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    SplitTransformationParam param;
    std::tie(precision, inputShape, targetDevice, params, param) = this->GetParam();

    init_input_shapes(inputShape);

    function = ngraph::builder::subgraph::SplitFunction::getOriginal(
        precision,
        inputShape,
        param.fakeQuantize,
        param.splitedAxis,
        param.numSplit);
}

TEST_P(SplitTransformation, CompareWithRefImpl) {
    run();
};
} // namespace LayerTestsDefinitions
