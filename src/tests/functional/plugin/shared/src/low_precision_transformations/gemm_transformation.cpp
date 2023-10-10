// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/gemm_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ov_models/builders.hpp"

#include "ov_lpt_models/mat_mul.hpp"

namespace LayerTestsDefinitions {

std::string GemmTransformation::getTestCaseName(const testing::TestParamInfo<GemmTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = obj.param;

    return getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params);
}

void GemmTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    const float low = 0.f; // params.precisionsOnActivations[0] == ngraph::element::u8 ? 0.f : -128.f;
    const float high = 255.f; // params.precisionsOnActivations[0] == ngraph::element::u8 ? 255.f : 127.f;

    function = ngraph::builder::subgraph::MatMulFunction::getOriginal(
        netPrecision,
        inputShape,
        low,
        high);
}

TEST_P(GemmTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
