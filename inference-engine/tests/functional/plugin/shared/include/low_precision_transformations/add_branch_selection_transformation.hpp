// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/convolution.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class AddBranchSelectionTestValues{
public:
    class Branch {
    public:
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeBefore;
        ngraph::builder::subgraph::Convolution convolution;
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeAfter;
    };

    Branch branch1;
    Branch branch2;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeAfter;
    std::vector<std::pair<std::string, std::string>> expectedReorders;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    AddBranchSelectionTestValues
> AddBranchSelectionTransformationParams;

class AddBranchSelectionTransformation :
    public testing::WithParamInterface<AddBranchSelectionTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<AddBranchSelectionTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
