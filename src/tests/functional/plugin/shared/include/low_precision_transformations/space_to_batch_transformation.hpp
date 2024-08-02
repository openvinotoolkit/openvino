// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"


namespace LayerTestsDefinitions {
class SpaceToBatchTransformationParam {
public:
    ov::PartialShape input_shape;
    std::vector<size_t> block_shape;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;
    ov::builder::subgraph::FakeQuantizeOnData fake_quantize;
    std::string layer_type;
    std::string expected_kernel_type;
};

typedef std::tuple<
    ov::element::Type,
    std::string,
    SpaceToBatchTransformationParam
> SpaceToBatchTransformationParams;

class SpaceToBatchTransformation :
    public testing::WithParamInterface<SpaceToBatchTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SpaceToBatchTransformationParams>& obj);

protected:
    void SetUp() override;
    void run() override;
};

}  // namespace LayerTestsDefinitions
