// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    ov::builder::subgraph::FakeQuantizeOnData> FakeQuantizeAndMaxPoolTransformationParams;

class FakeQuantizeAndMaxPoolTransformation :
    public testing::WithParamInterface<FakeQuantizeAndMaxPoolTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FakeQuantizeAndMaxPoolTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
