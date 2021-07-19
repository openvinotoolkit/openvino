// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string, // target device: CPU, GPU
    ngraph::pass::low_precision::LayerTransformation::Params, // transformation parameters
    bool, // transparent intermediate
    // multichannel
    bool> ConcatWithIntermediateTransformationParams;

class ConcatWithIntermediateTransformation :
    public testing::WithParamInterface<ConcatWithIntermediateTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConcatWithIntermediateTransformationParams> obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
