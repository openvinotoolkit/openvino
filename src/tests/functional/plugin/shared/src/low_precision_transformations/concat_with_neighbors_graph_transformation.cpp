// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_neighbors_graph_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/concat.hpp"

namespace LayerTestsDefinitions {

std::string ConcatWithNeighborsGraphTransformation::getTestCaseName(const testing::TestParamInfo<ConcatNeighboringGraphTransformationParams>& obj) {
    ov::element::Type precision;
    ov::PartialShape inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(precision, inputShapes, targetDevice, params) = obj.param;

    return get_test_case_name_by_params(precision, inputShapes, targetDevice, params);
}


void ConcatWithNeighborsGraphTransformation::SetUp() {
    ov::element::Type ngPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(ngPrecision, inputShape, targetDevice, params) = this->GetParam();

    init_input_shapes({ inputShape, inputShape, inputShape });

    function = ov::builder::subgraph::ConcatFunction::getOriginalWithNeighbors(
        ngPrecision,
        inputShape,
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} },
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 3.f} },
        "concat",
        "");
}

TEST_P(ConcatWithNeighborsGraphTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
