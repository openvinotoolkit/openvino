// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/tile_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>
#include <ngraph/ngraph.hpp>

#include "lpt_ngraph_functions/tile_function.hpp"

namespace LayerTestsDefinitions {

std::string TileTransformation::getTestCaseName(const testing::TestParamInfo<TileTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    TileTransformationParam param;;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params) << "_" <<
           param.fakeQuantize << "_";
    return result.str();
}

void TileTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    TileTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = GetParam();

    function = ngraph::builder::subgraph::TileFunction::getOriginal(
            netPrecision,
            inputShape,
            param.fakeQuantize);
}

TEST_P(TileTransformation, CompareWithRefImpl) {
    Run();
};

} // namespace LayerTestsDefinitions
