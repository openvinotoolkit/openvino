// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/mat_mul_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <queue>
#include <ie_core.hpp>

#include "ngraph/op/op.hpp"
#include <transformations/init_node_info.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "low_precision_transformations/mat_mul_transformation.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/mat_mul_function.hpp"

namespace LayerTestsDefinitions {

std::string MatMulTransformation::getTestCaseName(const testing::TestParamInfo<MatMulTransformationParams>& obj) {
    ngraph::element::Type precision;
    std::string targetDevice;
    MatMulTransformationTestValues testValues;
    std::tie(precision, targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.inputShape1 << "_" <<
        testValues.fqOnData1 << "_" <<
        testValues.inputShape2 << "_" <<
        testValues.fqOnData2;

    return result.str();
}

ov::test::utils::InputsMap MatMulTransformation::get_input_map() {
    auto generate_default = [](const std::shared_ptr<ngraph::Node>&node,
                               size_t port,
                               const ov::element::Type & elemType,
                               const ov::Shape & targetShape) -> ov::runtime::Tensor {
        const auto name = node->get_friendly_name();
        double low;
        double high;
        if (name == "fake_quantize1") {
            low = 1.0;
            high = 5.0;
        } else if (name == "fake_quantize2") {
            low = 5.0;
            high = 10.0;
        } else {
            IE_THROW() << "unexpected input name " << name;
        }

        return ov::test::utils::create_and_fill_tensor(elemType, targetShape, static_cast<uint32_t>(high - low), low);
    };

    static ov::test::utils::InputsMap inputs_map{
        { ov::op::Op::get_type_info_static(), generate_default }
    };
    return inputs_map;
}

void MatMulTransformation::SetUp() {
    ngraph::element::Type precision;
    MatMulTransformationTestValues testValues;
    std::tie(precision, targetDevice, testValues) = this->GetParam();

    init_input_shapes({ testValues.inputShape1, testValues.inputShape2 });

    function = ngraph::builder::subgraph::MatMulFunction::getOriginal(
        precision,
        testValues.inputShape1,
        testValues.fqOnData1,
        testValues.inputShape2,
        testValues.fqOnData2);

    ov::pass::InitNodeInfo().run_on_model(function);
}

void MatMulTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<2>(GetParam());
    const auto actualType = getRuntimePrecision(params.expectedKernelName);

    EXPECT_EQ(actualType, params.expectedRuntimePrecision);
}

TEST_P(MatMulTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
