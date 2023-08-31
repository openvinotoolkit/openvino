// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/variadic_split_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "low_precision/variadic_split.hpp"
#include "lpt_ngraph_functions/variadic_split_function.hpp"

namespace LayerTestsDefinitions {
std::string VariadicSplitTransformation::getTestCaseName(const testing::TestParamInfo<VariadicSplitTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    VariadicSplitTransformationParam param;
    std::tie(netPrecision, inputShapes, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) << "_" <<
        param.fakeQuantize << "_axis=" << param.splitedAxis << "_splitLengths={ ";
    for (size_t i = 0; i < param.splitLengths.size(); ++i) {
        result << param.splitLengths[i];
        if (i != (param.splitLengths.size() - 1ul)) {
            result << ", ";
        }
    }
    result << " }";
    return result.str();
}

ov::test::utils::InputsMap VariadicSplitTransformation::get_input_map() {
    auto generate_default = [this](const std::shared_ptr<ngraph::Node>&node,
                               size_t port,
                               const ov::element::Type & elemType,
                               const ov::Shape & targetShape) -> ov::runtime::Tensor {
        ngraph::element::Type precision;
        ngraph::PartialShape inputShape;
        std::string targetDevice;
        ngraph::pass::low_precision::LayerTransformation::Params params;
        VariadicSplitTransformationParam param;
        std::tie(precision, inputShape, targetDevice, params, param) = this->GetParam();
        const auto& fqOnData = param.fakeQuantize;

        const float range = static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]);
        const double start_from = fqOnData.empty() ? -12.5 : fqOnData.outputLowValues[0];
        return ov::test::utils::create_and_fill_tensor(elemType, targetShape, range, start_from);
    };

    static ov::test::utils::InputsMap inputs_map{
        { ov::op::Op::get_type_info_static(), generate_default }
    };
    return inputs_map;
}

void VariadicSplitTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    VariadicSplitTransformationParam param;
    std::tie(precision, inputShape, targetDevice, params, param) = this->GetParam();

    init_input_shapes(inputShape);

    function = ngraph::builder::subgraph::VariadicSplitFunction::getOriginal(
        precision,
        inputShape,
        param.fakeQuantize,
        param.splitedAxis,
        param.splitLengths);
}

TEST_P(VariadicSplitTransformation, CompareWithRefImpl) {
    run();
};
} // namespace LayerTestsDefinitions
