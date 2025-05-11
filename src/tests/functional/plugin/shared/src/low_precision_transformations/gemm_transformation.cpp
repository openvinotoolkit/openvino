// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/gemm_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"

#include "ov_lpt_models/mat_mul.hpp"

namespace LayerTestsDefinitions {

std::string GemmTransformation::getTestCaseName(const testing::TestParamInfo<GemmTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = obj.param;

    return get_test_case_name_by_params(netPrecision, inputShape, targetDevice, params);
}

void GemmTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    init_input_shapes({ inputShape, inputShape });

    const float low = 0.f; // params.precisionsOnActivations[0] == ov::element::u8 ? 0.f : -128.f;
    const float high = 255.f; // params.precisionsOnActivations[0] == ov::element::u8 ? 255.f : 127.f;

    function = ov::builder::subgraph::MatMulFunction::getOriginal(
        netPrecision,
        inputShape,
        low,
        high);
}

TEST_P(GemmTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
