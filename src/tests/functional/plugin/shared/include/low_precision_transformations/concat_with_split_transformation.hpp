// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {
class ConcatWithSplitTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData1;
    ngraph::builder::subgraph::FakeQuantizeOnData fqOnData2;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    ConcatWithSplitTransformationParam,
    ov::pass::low_precision::LayerTransformation::Params> ConcatWithSplitTransformationParams;

class ConcatWithSplitTransformation :
    public testing::WithParamInterface<ConcatWithSplitTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatWithSplitTransformationParams>& obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
