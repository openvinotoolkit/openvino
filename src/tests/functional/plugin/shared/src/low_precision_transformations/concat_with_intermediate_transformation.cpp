// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_intermediate_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/concat.hpp"

namespace LayerTestsDefinitions {

std::string ConcatWithIntermediateTransformation::getTestCaseName(const testing::TestParamInfo<ConcatWithIntermediateTransformationParams>& obj) {
    auto [netPrecision, inputShapes, device, transparentIntermediate, multichannel] = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShapes, device)
           << (transparentIntermediate ? "" : "_notTransparentIntermediate") << (multichannel ? "_multichannel" : "");

    return result.str();
}


/*
* FQ       FQ
*  \       /
*   \  Intermediate (MaxPooling or Convolution)
*    \  /    \
*   Concat   Convolution
*/

void ConcatWithIntermediateTransformation::SetUp() {
    auto [netPrecision, inputShape, device, transparentIntermediate, multichannel] = this->GetParam();
    targetDevice = device;

    ov::PartialShape inputShape1 = inputShape;
    if (inputShape1[2].is_static() && transparentIntermediate) {
        inputShape1[2] = inputShape1[2].get_length() - 2;
    }

    if (inputShape1[3].is_static() && transparentIntermediate) {
        inputShape1[3] = inputShape1[3].get_length() - 2;
    }

    init_input_shapes({ inputShape1, inputShape });

    function = ov::builder::subgraph::ConcatFunction::getOriginalWithIntermediate(
        netPrecision,
        inputShape,
        transparentIntermediate,
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ov::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} });
}

TEST_P(ConcatWithIntermediateTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
