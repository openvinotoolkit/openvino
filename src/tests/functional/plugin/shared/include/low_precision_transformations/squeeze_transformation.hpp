// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class SqueezeTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::vector<float> squeezeAxes;
    ngraph::PartialShape shape;
};

std::string stringifySqueezeArgs(const std::vector<float>& axes);

typedef std::tuple<
    ngraph::element::Type,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    SqueezeTransformationParam
> SqueezeTransformationParams;

class SqueezeTransformation :
    public testing::WithParamInterface<SqueezeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;
    static std::string getTestCaseName(const testing::TestParamInfo<SqueezeTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
