// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/convolution.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class ElementwiseBranchSelectionTestValues{
public:
    class Branch {
    public:
        ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeBefore;
        ov::builder::subgraph::Convolution convolution;
        ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeAfter;
    };

    Branch branch1;
    Branch branch2;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeAfter;
    std::vector<std::pair<std::string, std::string>> expectedReorders;
    // expected operation name + expected operation precision
    std::vector<std::pair<std::string, std::string>> expectedPrecisions;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ElementwiseBranchSelectionTestValues,
    std::string
> ElementwiseBranchSelectionTransformationParams;

class ElementwiseBranchSelectionTransformation :
    public testing::WithParamInterface<ElementwiseBranchSelectionTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ElementwiseBranchSelectionTransformationParams>& obj);

protected:
    void SetUp() override;
    void run() override;
};

}  // namespace LayerTestsDefinitions
