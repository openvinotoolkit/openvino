// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <utility>
#include <transformations/utils/utils.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

#include "lpt_ngraph_functions/reduce_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/constant.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class ReduceTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type preicsionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    std::vector<int64_t> constantValues;
    bool keepDims;
    Actual actual;
    Expected expected;
};

typedef std::tuple <
    ngraph::PartialShape,
    ReduceTransformationTestValues
> ReduceTransformationParams;

template <typename ReduceType>
class ReduceTransformation : public LayerTransformation, public testing::WithParamInterface<ReduceTransformationParams> {
public:
    void SetUp() override {
        const ngraph::PartialShape inputShape = std::get<0>(GetParam());
        const ReduceTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::ReduceFunction::getOriginal<ReduceType>(
            testValues.actual.inputPrecision,
            inputShape,
            testValues.actual.dequantization,
            testValues.constantValues,
            testValues.keepDims);

        referenceFunction = ngraph::builder::subgraph::ReduceFunction::getReference<ReduceType>(
            testValues.expected.inputPrecision,
            inputShape,
            testValues.expected.dequantizationBefore,
            testValues.constantValues,
            testValues.keepDims,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ReduceTransformationParams> obj) {
        const ngraph::PartialShape inputShape = std::get<0>(obj.param);
        const ReduceTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
            testValues.actual.inputPrecision << "_" <<
            LayerTransformation::getTestCaseNameByParams(testValues.actual.inputPrecision, inputShape, testValues.params) << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore << "_" <<
            testValues.expected.preicsionAfterOperation << "_" <<
            testValues.expected.dequantizationAfter << "_" <<
            (testValues.keepDims ? "_keep_dims_" : "_") <<
            "reduction_axes_";
        for (const auto& elem : testValues.constantValues) {
            result << "_" << elem << "_";
        }

        return result.str();
    }
};
