// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "ov_lpt_models/fake_quantize.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"

namespace LayerTestsDefinitions {

class FakeQuantizePrecisionSelectionTransformationActualValues {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizePrecisionSelectionTransformationActualValues& values) {
    return out << values.fakeQuantizeOnData << "_" << values.fakeQuantizeOnWeights;
}

class FakeQuantizePrecisionSelectionTransformationExpectedValues {
public:
    ov::element::Type fakeQuantizeOnDataOutPrecision;
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizePrecisionSelectionTransformationExpectedValues& values) {
    return out << values.fakeQuantizeOnDataOutPrecision << "_" << values.fakeQuantizeOnData << "_" << values.fakeQuantizeOnWeights;
}

class FakeQuantizePrecisionSelectionTransformationTestValues {
public:
    std::vector<ov::element::Type> precisionsOnActivations;
    std::vector<ov::element::Type> precisionsOnActivationForLimitedOperation;
    bool operationBeforeLimitedOperationIsPrecisionTransparent;
    FakeQuantizePrecisionSelectionTransformationActualValues actual;
    FakeQuantizePrecisionSelectionTransformationExpectedValues expected;
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizePrecisionSelectionTransformationTestValues& values) {
    return out << values.operationBeforeLimitedOperationIsPrecisionTransparent << "_" << values.actual << "_" << values.expected;
}

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    FakeQuantizePrecisionSelectionTransformationTestValues> FakeQuantizeTransformationParams;

class FakeQuantizePrecisionSelectionTransformation :
    public testing::WithParamInterface<FakeQuantizeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FakeQuantizeTransformationParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
