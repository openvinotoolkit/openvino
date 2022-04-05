// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {
class BroadcastTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    BroadcastTransformationParam,
    size_t,
    std::string
> BroadcastTransformationParams;

class BroadcastTransformation :
        public testing::WithParamInterface<BroadcastTransformationParams>,
        public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BroadcastTransformationParams>& obj);
protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
