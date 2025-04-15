// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace LayerTestsDefinitions {
class StridedSliceTransformationParam {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> strides;
    std::vector<int64_t> beginMask;
    std::vector<int64_t> endMask;
    std::vector<int64_t> newAxisMask;
    std::vector<int64_t> shrinkAxisMask;
    std::vector<int64_t> elipsisMask;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    StridedSliceTransformationParam
> StridedSliceTransformationParams;

class StridedSliceTransformation :
    public testing::WithParamInterface<StridedSliceTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StridedSliceTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
