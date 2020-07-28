// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    ngraph::element::Type,
    std::pair<ngraph::Shape, ngraph::Shape>,
    std::string,
    ngraph::op::InterpolateAttrs,
    LayerTestsUtils::LayerTransformation::LptVersion> InterpolateTransformationParams;

class InterpolateTransformation :
    public testing::WithParamInterface<InterpolateTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateTransformationParams> obj);

protected:
    void SetUp() override;

private:
    void validate();
};

}  // namespace LayerTestsDefinitions
