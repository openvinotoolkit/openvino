// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/prelu_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "lpt_ngraph_functions/prelu_function.hpp"

namespace LayerTestsDefinitions {

std::string PReluTransformation::getTestCaseName(const testing::TestParamInfo<PReluTransformationParams>& obj) {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    PReluTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.fakeQuantize;

    return result.str();
}

ov::test::utils::InputsMap PReluTransformation::get_input_map() {
    auto generate_default = [this](const std::shared_ptr<ngraph::Node>&node,
                               size_t port,
                               const ov::element::Type & elemType,
                               const ov::Shape & targetShape) -> ov::runtime::Tensor {
        ngraph::element::Type precision;
        ngraph::PartialShape inputShape;
        std::string targetDevice;
        PReluTestValues testValues;
        std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

        const auto fqOnData = testValues.fakeQuantize;
        const uint32_t range = static_cast<uint32_t>(fqOnData.empty() ? 25 : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]);
        const double start_from = fqOnData.empty() ? -12.5 : fqOnData.outputLowValues[0];
        return ov::test::utils::create_and_fill_tensor(elemType, targetShape, range, start_from);
    };

    static ov::test::utils::InputsMap inputs_map{
        { ov::op::Op::get_type_info_static(), generate_default }
    };
    return inputs_map;
}

void PReluTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    PReluTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    init_input_shapes(inputShape);

    function = ngraph::builder::subgraph::PReluFunction::getOriginal(inputShape, precision, testValues.fakeQuantize);

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(PReluTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
