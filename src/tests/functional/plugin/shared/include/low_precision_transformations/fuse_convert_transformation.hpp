// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

typedef std::
    tuple<ov::element::Type, ov::PartialShape, std::string, ov::builder::subgraph::DequantizationOperations, bool>
        FuseConvertTransformationParams;

class FuseConvertTransformation : public testing::WithParamInterface<FuseConvertTransformationParams>,
                                  public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FuseConvertTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
