// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "ov_lpt_models/fake_quantize.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

#include "ov_lpt_models/fake_quantize_and_convolution.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/constant.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"

namespace LayerTestsDefinitions {

class FakeQuantizeWithNotOptimalTransformationTestValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fqOnData;
    ngraph::builder::subgraph::DequantizationOperations::Convert convertOnData;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOnData;

    ngraph::builder::subgraph::Constant constantOnWeights;
    ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights;
    ngraph::builder::subgraph::DequantizationOperations::Convert convertOnWeights;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOnWeights;

    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    std::string expectedPrecision;
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeWithNotOptimalTransformationTestValues& data) {
    return out <<  "_" <<
        data.fqOnData << "_" <<
        data.convertOnData << "_" <<
        data.dequantizationOnData << "_" <<

        data.constantOnWeights << "_" <<
        data.fqOnWeights << "_" <<
        data.convertOnWeights << "_" <<
        data.dequantizationOnWeights <<

        data.dequantizationAfter << "_" <<
        data.expectedPrecision;
}

// ngraph::builder::subgraph::FakeQuantizeOnData
typedef std::tuple<
    ngraph::element::Type,
    ngraph::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    FakeQuantizeWithNotOptimalTransformationTestValues> FakeQuantizeTransformationParams;

class FakeQuantizeWithNotOptimalTransformation :
    public testing::WithParamInterface<FakeQuantizeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FakeQuantizeTransformationParams>& obj);

protected:
    void SetUp() override;
    void Run() override;
};

}  // namespace LayerTestsDefinitions
