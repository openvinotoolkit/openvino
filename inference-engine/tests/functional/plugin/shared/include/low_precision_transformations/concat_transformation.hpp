// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class ConcatTransformation : public LayerTestsUtils::LayerTransformation<LayerTestsUtils::LayerTransformationParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj);

protected:
    void SetUp() override;

private:
    void validate();
};

}  // namespace LayerTestsDefinitions
