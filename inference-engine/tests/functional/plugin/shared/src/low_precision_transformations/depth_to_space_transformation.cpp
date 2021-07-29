// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/depth_to_space_transformation.hpp"

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
#include "ngraph_functions/builders.hpp"

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <ngraph/op/depth_to_space.hpp>

#include "lpt_ngraph_functions/depth_to_space_function.hpp"

using namespace ngraph::opset1;

namespace LayerTestsDefinitions {

std::string DepthToSpaceTransformation::getTestCaseName(testing::TestParamInfo<DepthToSpaceTransformationParams> obj) {
    static std::map<DepthToSpace::DepthToSpaceMode, std::string> names = {
        {DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, "BLOCKS_FIRST"},
        {DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, "DEPTH_FIRST"},
    };

    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    DepthToSpace::DepthToSpaceMode mode;
    size_t blockSize;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    std::tie(precision, inputShape, targetDevice, mode, blockSize) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, inputShape, targetDevice, params) <<
        "_" << names[mode] << "_" << blockSize;
    return result.str();
}

void DepthToSpaceTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    DepthToSpace::DepthToSpaceMode mode;
    size_t blockSize;
    std::tie(precision, inputShape, targetDevice, mode, blockSize) = this->GetParam();

    if (inputShape.rank().is_dynamic() || inputShape.rank().get_length() != 4) {
        IE_THROW() << "not supported input shape size " << inputShape.rank();
    }

    function = ngraph::builder::subgraph::DepthToSpaceFunction::getOriginal(precision, inputShape, mode, blockSize);
}

TEST_P(DepthToSpaceTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
