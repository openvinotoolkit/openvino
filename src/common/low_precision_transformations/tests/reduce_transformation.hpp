// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <utility>
#include "transformations/utils/utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"

#include "ov_lpt_models/reduce.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/constant.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class ReduceTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type inputPrecision;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ov::element::Type inputPrecision;
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        ov::element::Type preicsionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    std::vector<int64_t> constantValues;
    bool keepDims;
    Actual actual;
    Expected expected;
};

typedef std::tuple <
    ov::PartialShape,
    ReduceTransformationTestValues
> ReduceTransformationParams;

template <typename ReduceType>
class ReduceTransformation : public LayerTransformation, public testing::WithParamInterface<ReduceTransformationParams> {
public:
    void SetUp() override {
        const ov::PartialShape inputShape = std::get<0>(GetParam());
        const ReduceTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = ov::builder::subgraph::ReduceFunction::getOriginal<ReduceType>(
            testValues.actual.inputPrecision,
            inputShape,
            testValues.actual.dequantization,
            testValues.constantValues,
            testValues.keepDims);

        referenceFunction = ov::builder::subgraph::ReduceFunction::getReference<ReduceType>(
            testValues.expected.inputPrecision,
            inputShape,
            testValues.expected.dequantizationBefore,
            testValues.constantValues,
            testValues.keepDims,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ReduceTransformationParams> obj) {
        const ov::PartialShape inputShape = std::get<0>(obj.param);
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
