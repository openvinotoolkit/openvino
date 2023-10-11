// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <low_precision/batch_to_space.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/batch_to_space.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"


using namespace testing;
using namespace ov::pass;
using namespace ov;

class BatchToSpaceTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type input_type;
        ngraph::builder::subgraph::DequantizationOperations dequantization_before;
        ov::element::Type preicsionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantization_after;
    };

    class Expected {
    public:
        ov::element::Type input_type;
        ngraph::builder::subgraph::DequantizationOperations dequantization_before;
        ov::element::Type preicsionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantization_after;
    };

    TestTransformationParams params;
    std::vector<size_t> block_shape;
    std::vector<size_t> crops_begin;
    std::vector<size_t> crops_end;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::PartialShape,
    BatchToSpaceTransformationTestValues> BatchToSpaceTransformationParams;

class BatchToSpaceTransformation : public LayerTransformation,
                                   public testing::WithParamInterface<BatchToSpaceTransformationParams> {
public:
    void SetUp() override {
        const ngraph::PartialShape input_shape = std::get<0>(GetParam());
        const BatchToSpaceTransformationTestValues test_values = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::BatchToSpaceFunction::get(
            input_shape,
            test_values.actual.input_type,
            test_values.actual.dequantization_before,
            test_values.block_shape,
            test_values.crops_begin,
            test_values.crops_end,
            test_values.actual.dequantization_after);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::BatchToSpaceTransformation>(test_values.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::BatchToSpaceFunction::get(
            input_shape,
            test_values.expected.input_type,
            test_values.expected.dequantization_before,
            test_values.block_shape,
            test_values.crops_begin,
            test_values.crops_end,
            test_values.expected.dequantization_after);
    }

    static std::string getTestCaseName(testing::TestParamInfo<BatchToSpaceTransformationParams> obj) {
        const ngraph::PartialShape shape = std::get<0>(obj.param);
        const BatchToSpaceTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result << testValues.actual.input_type << "_"<<
            shape << "_" << toString(testValues.params) << "_" <<
            testValues.actual.dequantization_before << "_" <<
            testValues.actual.dequantization_after << "_" <<
            testValues.expected.dequantization_before << "_" <<
            testValues.expected.dequantization_after << "_";
        return result.str();
    }
};

TEST_P(BatchToSpaceTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(actualFunction, referenceFunction, true, false, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues {
const std::vector<ngraph::PartialShape> input_shapes = {
    {4, 3, 50, 86}
};

const std::vector<BatchToSpaceTransformationTestValues> test_values = {
    // per-tensor dequantization
    {
        LayerTransformation::createParamsU8I8(),
        {1, 1, 2, 2},
        {0, 0, 0, 0},
        {0, 0, 0, 1},
        {
            ov::element::u8,
            { ov::element::f32, {128.f}, {0.01f}},
            ov::element::f32,
            {}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            { ov::element::f32, {128.f}, {0.01f}}
        }
    },
    // per-channel dequantization
    {
        LayerTransformation::createParamsU8I8(),
        {1, 1, 2, 2},
        {0, 0, 0, 0},
        {0, 0, 0, 1},
        {
            ov::element::u8,
            {ov::element::f32, {{128.f, 64.f, 32.f}}, {{0.02f, 0.01f, 0.03f}}},
            ov::element::f32,
            {}
        },
        {
            ov::element::u8,
            {ov::element::f32, {{128.f, 64.f, 32.f}}, { {0.02f, 0.01f, 0.03f} }},
            ov::element::f32,
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    BatchToSpaceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(input_shapes),
        ::testing::ValuesIn(test_values)),
    BatchToSpaceTransformation::getTestCaseName);
} // namespace testValues
