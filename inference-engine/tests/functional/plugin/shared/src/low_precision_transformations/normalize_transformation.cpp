// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/normalize_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"
#include "lpt_ngraph_functions/normalize_l2_function.hpp"

namespace LayerTestsDefinitions {

std::string NormalizeL2Transformation::getTestCaseName(testing::TestParamInfo<NormalizeL2TransformationParams> obj) {
    ngraph::element::Type netPrecision;
    std::pair<ngraph::PartialShape, ngraph::Shape> shapes;
    std::string targetDevice;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    std::vector<uint64_t> axes;
    bool fuseMultiply;
    bool shift;
    std::tie(netPrecision, shapes, targetDevice, axes, fuseMultiply, shift) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" <<
        shapes.first << "_" <<
        shapes.second << "_" <<
        targetDevice << "_" <<
        toString(params) << "_" <<
        "_axes" << axes.size() <<
        (fuseMultiply ? "_multiply" : "") <<
        (shift ? "_shift" : "");
    return result.str();
}

void NormalizeL2Transformation::SetUp() {
    threshold = 3.e-3;
    std::pair<ngraph::PartialShape, ngraph::Shape> shapes;
    ngraph::element::Type precision;
    std::vector<uint64_t> axes;
    bool fuseMultiply;
    bool shift;
    std::tie(precision, shapes, targetDevice, axes, fuseMultiply, shift) = this->GetParam();

    function = ngraph::builder::subgraph::NormalizeL2Function::getOriginal(
        precision,
        shapes,
        ngraph::element::u8,
        axes,
        fuseMultiply,
        shift);
}

TEST_P(NormalizeL2Transformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
