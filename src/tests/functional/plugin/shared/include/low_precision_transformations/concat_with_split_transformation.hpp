// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {
class ConcatWithSplitTransformationParam {
public:
    ov::builder::subgraph::FakeQuantizeOnData fqOnData1;
    ov::builder::subgraph::FakeQuantizeOnData fqOnData2;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ConcatWithSplitTransformationParam,
    ov::pass::low_precision::LayerTransformation::Params> ConcatWithSplitTransformationParams;

class ConcatWithSplitTransformation :
    public testing::WithParamInterface<ConcatWithSplitTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatWithSplitTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
