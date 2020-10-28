// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple <
    ngraph::element::Type,
    std::pair<ngraph::Shape, ngraph::Shape>,
    std::string,
    std::vector<uint64_t>,
    bool,
    bool> NormalizeL2TransformationParams;

class NormalizeL2Transformation :
    public testing::WithParamInterface<NormalizeL2TransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<NormalizeL2TransformationParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
