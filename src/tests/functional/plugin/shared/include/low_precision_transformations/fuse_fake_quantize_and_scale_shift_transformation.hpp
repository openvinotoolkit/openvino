// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "ov_lpt_models/fuse_fake_quantize_and_scale_shift.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    ov::builder::subgraph::FakeQuantizeOnData> FuseFakeQuantizeAndScaleShiftTransformationParams;

class FuseFakeQuantizeAndScaleShiftTransformation :
    public testing::WithParamInterface<FuseFakeQuantizeAndScaleShiftTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FuseFakeQuantizeAndScaleShiftTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
