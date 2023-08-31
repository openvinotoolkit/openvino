// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/concat_function.hpp"

namespace LayerTestsDefinitions {

std::string ConcatTransformation::getTestCaseName(const testing::TestParamInfo<ConcatTransformationParams>& obj) {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ConcatTransformationTestValues testValues;
    std::tie(precision, inputShapes, targetDevice, testValues) = obj.param;

    const auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, inputShapes, targetDevice, params) <<
        testValues.fqOnData1 <<
        testValues.dequantization1 <<
        testValues.fqOnData2 <<
        testValues.dequantization2;
    return result.str();
}

ov::test::utils::InputsMap ConcatTransformation::get_input_map() {
    auto generate_default = [](const std::shared_ptr<ngraph::Node>&node,
                               size_t port,
                               const ov::element::Type & elemType,
                               const ov::Shape & targetShape) -> ov::runtime::Tensor {
        const auto name = node->get_friendly_name();
        if ((name != "fakeQuantize1") && (name != "fakeQuantize2") && (name != "dequantization2")) {
            OPENVINO_THROW("unknown name: " + name);
        }
        const double k = (name == "fakeQuantize1") ? 1.0 : (name == "fakeQuantize2" ? 2.0 : 3.0);
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

void ConcatTransformation::SetUp() {
    ngraph::PartialShape inputShape;
    ngraph::element::Type precision;
    ConcatTransformationTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    std::vector<ngraph::PartialShape> inputs;
    if (testValues.input_constant1 == nullptr) {
        inputs.push_back(inputShape);
    }
    if (testValues.input_constant2 == nullptr) {
        inputs.push_back(inputShape);
    }
    init_input_shapes(inputs);

    function = ngraph::builder::subgraph::ConcatFunction::getOriginal(
        precision,
        inputShape,
        testValues.input_constant1,
        testValues.fqOnData1,
        testValues.dequantization1,
        testValues.input_constant2,
        testValues.fqOnData2,
        testValues.dequantization2);
}

TEST_P(ConcatTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
