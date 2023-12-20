// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_neighbors_graph_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ov_models/builders.hpp"
#include "ov_lpt_models/concat.hpp"

namespace LayerTestsDefinitions {

std::string ConcatWithNeighborsGraphTransformation::getTestCaseName(const testing::TestParamInfo<ConcatNeighboringGraphTransformationParams>& obj) {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(precision, inputShapes, targetDevice, params) = obj.param;

    return getTestCaseNameByParams(precision, inputShapes, targetDevice, params);
}


void ConcatWithNeighborsGraphTransformation::SetUp() {
    rel_threshold = 0.1;
    abs_threshold = 0.1;
    ngraph::element::Type ngPrecision;
    ngraph::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(ngPrecision, inputShape, targetDevice, params) = this->GetParam();

    init_input_shapes({ inputShape, inputShape, inputShape });

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithNeighbors(
        ngPrecision,
        inputShape,
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 3.f} },
        "concat",
        "");
}

TEST_P(ConcatWithNeighborsGraphTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
