// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_different_precision_on_children.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "lpt_ngraph_functions/concat_function.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace LayerTestsDefinitions {

std::string ConcatWithDifferentChildrenTransformation::getTestCaseName(const testing::TestParamInfo<ConcatWithDifferentChildrenTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ConcatWithDifferentChildrenTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) <<
        "_axis_" << param.axis << "_" << param.fqOnData1 << param.fqOnData2;

    return result.str();
}

ov::test::utils::InputsMap ConcatWithDifferentChildrenTransformation::get_input_map() {
    auto generate_default = [](const std::shared_ptr<ngraph::Node>&node,
                               size_t port,
                               const ov::element::Type & elemType,
                               const ov::Shape & targetShape) -> ov::runtime::Tensor {
        const auto name = node->get_friendly_name();
        if ((name != "fakeQuantize1") && (name != "fakeQuantize2")) {
            OPENVINO_THROW("unknown name: " + name);
        }
        const double k = (name == "fakeQuantize1") ? 1.0 : 2.0;
        const auto interval = LayerTestsUtils::LayerTransformation::getQuantizationInterval(ngraph::element::u8);
        const double low = interval.first / k;
        const double high = interval.second / k;

        return ov::test::utils::create_and_fill_tensor(elemType, targetShape, static_cast<uint32_t>(high - low), low);
    };

    static ov::test::utils::InputsMap inputs_map{
        { ov::op::Op::get_type_info_static(), generate_default }
    };
    return inputs_map;
}

void ConcatWithDifferentChildrenTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    ConcatWithDifferentChildrenTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, param, params) = this->GetParam();

    init_input_shapes({ inputShapes, inputShapes });

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithDifferentPrecisionOnChildren(
        netPrecision, inputShapes, param.axis, param.fqOnData1, param.fqOnData2);
}

TEST_P(ConcatWithDifferentChildrenTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
