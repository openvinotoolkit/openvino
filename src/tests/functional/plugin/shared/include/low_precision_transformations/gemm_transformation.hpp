// Copyright (C) 2018-2022 Intel Corporation
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
    ngraph::pass::low_precision::LayerTransformation::Params> GemmTransformationParams;

// TODO: use MatMulTransformation
class GemmTransformation :
    public testing::WithParamInterface<GemmTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GemmTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
