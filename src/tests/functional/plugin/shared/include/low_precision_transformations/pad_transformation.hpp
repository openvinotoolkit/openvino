// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {
class PadTransformationParam {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::vector<int64_t> padsBegin;
    std::vector<int64_t> padsEnd;
    float padValue;
    std::string layerName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    ov::op::PadMode,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    PadTransformationParam
> PadTransformationParams;

class PadTransformation :
    public testing::WithParamInterface<PadTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PadTransformationParams>& obj);

protected:
    void SetUp() override;
    void run() override;
};
}  // namespace LayerTestsDefinitions
