// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_convert_transformation.hpp"

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
#include "lpt_ngraph_functions/fuse_convert_function.hpp"

namespace LayerTestsDefinitions {

std::string FuseConvertTransformation::getTestCaseName(testing::TestParamInfo<FuseConvertTransformationParams> obj) {
    std::string targetDevice;
    ngraph::PartialShape shape;
    ngraph::element::Type precision;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    ngraph::builder::subgraph::DequantizationOperations deqOperations;
    bool constInput;
    std::tie(precision, shape, targetDevice, deqOperations, constInput) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, shape, targetDevice, params) <<
           "_" << deqOperations << "_" << constInput;
    return result.str();
}

void FuseConvertTransformation::SetUp() {
    ngraph::PartialShape shape;
    ngraph::element::Type precision;
    ngraph::builder::subgraph::DequantizationOperations deqOperations;
    bool constInput;
    std::tie(precision, shape, targetDevice, deqOperations, constInput) = this->GetParam();

    function = ngraph::builder::subgraph::FuseConvertFunction::getWithFQ(
        shape,
        precision,
        deqOperations,
        constInput);
}

TEST_P(FuseConvertTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
