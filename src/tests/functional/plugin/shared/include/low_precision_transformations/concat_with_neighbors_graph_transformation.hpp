// Copyright (C) 2018-2023 Intel Corporation
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
    std::string,
    ov::pass::low_precision::LayerTransformation::Params> ConcatNeighboringGraphTransformationParams;

class ConcatWithNeighborsGraphTransformation :
    public testing::WithParamInterface<ConcatNeighboringGraphTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatNeighboringGraphTransformationParams>& obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
