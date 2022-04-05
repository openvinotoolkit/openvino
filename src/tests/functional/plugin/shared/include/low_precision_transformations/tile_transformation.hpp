// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {
class TileTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
};

typedef std::tuple<
        ngraph::element::Type,
        ngraph::PartialShape,
        std::string,
        ngraph::pass::low_precision::LayerTransformation::Params,
        TileTransformationParam
> TileTransformationParams;

class TileTransformation :
        public testing::WithParamInterface<TileTransformationParams>,
        public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TileTransformationParams>& obj);
protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
