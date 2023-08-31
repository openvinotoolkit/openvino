// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class ReluTestValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    bool isSubtract;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    ReluTestValues> ReluTransformationParams;

class ReluTransformation :
    public testing::WithParamInterface<ReluTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReluTransformationParams>& obj);
    ov::test::utils::InputsMap get_input_map() override;

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
