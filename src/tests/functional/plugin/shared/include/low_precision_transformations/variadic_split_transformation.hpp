// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {
class VariadicSplitTransformationParam {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    int64_t splitedAxis;
    std::vector<size_t> splitLengths;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    VariadicSplitTransformationParam
> VariadicSplitTransformationParams;

class VariadicSplitTransformation :
    public testing::WithParamInterface<VariadicSplitTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<VariadicSplitTransformationParams>& obj);
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
