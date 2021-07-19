// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple <
    ngraph::element::Type,
    std::pair<ngraph::PartialShape, ngraph::Shape>,
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
