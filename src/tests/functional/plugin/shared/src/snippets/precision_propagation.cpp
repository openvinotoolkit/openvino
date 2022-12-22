// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/precision_propagation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "transformations/init_node_info.hpp"
#include "precision_propagation_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string PrecisionPropagationTest::getTestCaseName(
    const testing::TestParamInfo<PrecisionPropagationTestParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    PrecisionPropagationTestValues param;
    std::tie(netPrecision, inputShapes, targetDevice, param) = obj.param;

    std::ostringstream result;
    result <<
        param.inputShape1 << "_" << param.precision1 << "_" <<
        param.inputShape2 << " " << param.precision2;
    return result.str();
}

void PrecisionPropagationTest::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    PrecisionPropagationTestValues param;
    std::tie(precision, inputShape, targetDevice, param) = this->GetParam();

    function = ov::test::snippets::PrecisionPropagationFunction::get(
        param.precision1,
        param.inputShape1,
        param.precision2,
        param.inputShape2);

    ngraph::pass::InitNodeInfo().run_on_function(function);

    init_input_shapes({
        {{}, {param.inputShape1.to_shape(), }}, 
        {{}, {param.inputShape2.to_shape(), }}
    });

    ngraph::pass::VisualizeTree("svg/test.actual.svg").run_on_model(function);
}

TEST_P(PrecisionPropagationTest, CompareWithRefImpl) {
    run();
    //validateNumSubgraphs();
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
