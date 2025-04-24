// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class AddTestValues{
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    bool broadcast;
    std::vector<ov::element::Type> precisionOnActivations;
    std::vector<ov::element::Type> expectedPrecisions;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    AddTestValues
> AddTransformationParams;

class AddTransformation :
    public testing::WithParamInterface<AddTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<AddTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
