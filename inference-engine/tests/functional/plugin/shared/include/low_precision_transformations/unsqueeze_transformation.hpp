// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class UnsqueezeTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::vector<float> unsqueezeAxes;
    ngraph::PartialShape shape;
};

typedef std::tuple<
    ngraph::element::Type,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    UnsqueezeTransformationParam
> UnsqueezeTransformationParams;

class UnsqueezeTransformation :
    public testing::WithParamInterface<UnsqueezeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;
    static std::string getTestCaseName(testing::TestParamInfo<UnsqueezeTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
