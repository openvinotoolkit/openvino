// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "lpt_ngraph_functions/fake_quantize_function.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {
class FakeQuantizeTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakequantize;

    std::string layerName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    FakeQuantizeTransformationParam> FakeQuantizeTransformationParams;

class FakeQuantizeTransformation :
    public testing::WithParamInterface<FakeQuantizeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeTransformationParams> obj);

protected:
    void SetUp() override;
    void Run() override;
};

}  // namespace LayerTestsDefinitions
