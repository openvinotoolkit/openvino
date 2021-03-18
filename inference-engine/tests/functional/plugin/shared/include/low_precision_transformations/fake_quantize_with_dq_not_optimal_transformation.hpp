// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "lpt_ngraph_functions/fake_quantize_function.hpp"
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

#include "lpt_ngraph_functions/fake_quantize_and_convolution_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/constant.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"

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
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    FakeQuantizeWithNotOptimalTransformationTestValues> FakeQuantizeTransformationParams;

class FakeQuantizeWithNotOptimalTransformation :
    public testing::WithParamInterface<FakeQuantizeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeTransformationParams> obj);

protected:
    void SetUp() override;
    void Run() override;
};

}  // namespace LayerTestsDefinitions
