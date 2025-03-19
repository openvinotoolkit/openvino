// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class ShuffleChannelsTransformationParam {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::int64_t axis;
    std::int64_t group;
    std::string layerName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    ShuffleChannelsTransformationParam
> ShuffleChannelsTransformationParams;

class ShuffleChannelsTransformation :
    public testing::WithParamInterface<ShuffleChannelsTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ShuffleChannelsTransformationParams>& obj);

protected:
    void SetUp() override;
    void run() override;
};

}  // namespace LayerTestsDefinitions
