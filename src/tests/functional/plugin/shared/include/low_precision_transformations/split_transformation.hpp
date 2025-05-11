// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {
class SplitTransformationParam {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    int64_t splitedAxis;
    size_t numSplit;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    SplitTransformationParam
> SplitTransformationParams;

class SplitTransformation :
    public testing::WithParamInterface<SplitTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SplitTransformationParams>& obj);
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
