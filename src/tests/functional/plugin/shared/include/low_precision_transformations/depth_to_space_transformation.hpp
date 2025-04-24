// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "openvino/op/depth_to_space.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ov::op::v0::DepthToSpace::DepthToSpaceMode,
    size_t> DepthToSpaceTransformationParams;

class DepthToSpaceTransformation :
    public testing::WithParamInterface<DepthToSpaceTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DepthToSpaceTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
