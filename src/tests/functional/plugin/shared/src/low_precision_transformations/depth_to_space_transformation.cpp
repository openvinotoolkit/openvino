// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "low_precision_transformations/depth_to_space_transformation.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "ov_lpt_models/depth_to_space.hpp"
#include "transformations/common_optimizations/depth_to_space_fusion.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

namespace LayerTestsDefinitions {

std::string DepthToSpaceTransformation::getTestCaseName(const testing::TestParamInfo<DepthToSpaceTransformationParams>& obj) {
    static std::map<ov::op::v0::DepthToSpace::DepthToSpaceMode, std::string> names = {
        {ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, "BLOCKS_FIRST"},
        {ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, "DEPTH_FIRST"},
    };

    ov::element::Type precision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ov::op::v0::DepthToSpace::DepthToSpaceMode mode;
    size_t blockSize;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    std::tie(precision, inputShape, targetDevice, mode, blockSize) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(precision, inputShape, targetDevice, params) <<
           "_" << names[mode] << "_" << blockSize;
    return result.str();
}

void DepthToSpaceTransformation::SetUp() {
    ov::element::Type precision;
    ov::PartialShape inputShape;
    ov::op::v0::DepthToSpace::DepthToSpaceMode mode;
    size_t blockSize;
    std::tie(precision, inputShape, targetDevice, mode, blockSize) = this->GetParam();

    init_input_shapes(inputShape);

    if (inputShape.rank().is_dynamic() || inputShape.rank().get_length() != 4) {
        OPENVINO_THROW("not supported input shape size ", inputShape.rank());
    }

    function = ov::builder::subgraph::DepthToSpaceFunction::getOriginal(precision, inputShape, mode, blockSize);
}

TEST_P(DepthToSpaceTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
