// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ngraph;
using ngraph::helpers::EltwiseTypes;

namespace SubgraphTestsDefinitions {

class TileWithTwoOutputEdges : public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        auto ngPrc = element::f32;
        ov::ParameterVector inputParams {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, 3, 12, 9})};

        auto tile = ngraph::builder::makeTile(inputParams[0], std::vector<int64_t>{1, 2, 1, 1});

        const auto const1 = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>{1, 6, 1, 1}, std::vector<float>{}, true);
        const auto const2 = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>{1, 6, 1, 1}, std::vector<float>{}, true);

        const auto add1 = ngraph::builder::makeEltwise(tile->output(0), const1, ngraph::helpers::EltwiseTypes::ADD);
        const auto add2 = ngraph::builder::makeEltwise(tile->output(0), const2, ngraph::helpers::EltwiseTypes::ADD);

        NodeVector results{add1, add2};
        function = std::make_shared<ngraph::Function>(results, inputParams, "TileWithTwoOutputEdges");
    }
};

TEST_F(TileWithTwoOutputEdges, smoke_CompareWithRefs) {
    Run();
}

} // namespace SubgraphTestsDefinitions
