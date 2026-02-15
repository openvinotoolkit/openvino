// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <set>

#include "ov_lpt_models/common/add.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class EliminateFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBefore;
        ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData1;
        ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData2;
    };

    class Expected {
    public:
        std::set<std::string> exist;
        std::set<std::string> absent;
        size_t int8_convolutions;
    };

    ov::PartialShape inputShape;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    std::string,
    EliminateFakeQuantizeTransformationTestValues> EliminateFakeQuantizeTransformationParams;

class EliminateFakeQuantizeTransformation
    : public testing::WithParamInterface<EliminateFakeQuantizeTransformationParams>,
      public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<EliminateFakeQuantizeTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
