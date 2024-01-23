// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_convert_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ov_models/pass/convert_prc.hpp"
#include "ov_lpt_models/fuse_convert.hpp"

namespace LayerTestsDefinitions {

std::string FuseConvertTransformation::getTestCaseName(const testing::TestParamInfo<FuseConvertTransformationParams>& obj) {
    std::string targetDevice;
    ov::PartialShape shape;
    ov::element::Type precision;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    ngraph::builder::subgraph::DequantizationOperations deqOperations;
    bool constInput;
    std::tie(precision, shape, targetDevice, deqOperations, constInput) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(precision, shape, targetDevice, params) <<
           "_" << deqOperations << "_" << constInput;
    return result.str();
}

void FuseConvertTransformation::SetUp() {
    abs_threshold = 0.01;
    ov::PartialShape shape;
    ov::element::Type precision;
    ngraph::builder::subgraph::DequantizationOperations deqOperations;
    bool constInput;
    std::tie(precision, shape, targetDevice, deqOperations, constInput) = this->GetParam();

    init_input_shapes(constInput ? std::vector<ov::PartialShape>{ shape } : std::vector<ov::PartialShape>{ shape, shape });

    function = ngraph::builder::subgraph::FuseConvertFunction::getWithFQ(
        shape,
        precision,
        deqOperations,
        constInput);
}

TEST_P(FuseConvertTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
