// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class PReluTestValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    bool isSubtract;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    PReluTestValues> PReluTransformationParams;

class PReluTransformation :
    public testing::WithParamInterface<PReluTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PReluTransformationParams> obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
