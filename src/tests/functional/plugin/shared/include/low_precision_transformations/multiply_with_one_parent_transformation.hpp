// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class MultiplyWithOneParentTransformationValues {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    MultiplyWithOneParentTransformationValues
> MultiplyWithOneParentTransformationParams;

class MultiplyWithOneParentTransformation :
    public testing::WithParamInterface<MultiplyWithOneParentTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MultiplyWithOneParentTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
