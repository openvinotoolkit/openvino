// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {
class ConcatWithSplitTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData1;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData2;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    std::string,
    ConcatWithSplitTransformationParam,
    ngraph::pass::low_precision::LayerTransformation::Params> ConcatWithSplitTransformationParams;

class ConcatWithSplitTransformation :
    public testing::WithParamInterface<ConcatWithSplitTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConcatWithSplitTransformationParams> obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
