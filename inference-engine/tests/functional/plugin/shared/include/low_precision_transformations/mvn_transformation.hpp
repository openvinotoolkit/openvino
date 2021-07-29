// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

using namespace ngraph;

namespace LayerTestsDefinitions {

typedef std::tuple <
    element::Type,
    PartialShape,
    std::string,
    AxisSet,
    bool> MVNTransformationParams;

class MVNTransformation :
    public testing::WithParamInterface<MVNTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MVNTransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
