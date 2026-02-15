// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/space_to_batch.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/space_to_batch.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"


using namespace testing;
using namespace ov::pass;
using namespace ov;

class SpaceToBatchTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type input_type;
        ov::builder::subgraph::DequantizationOperations dequantization_before;
        ov::element::Type preicsionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantization_after;
    };

    class Expected {
    public:
        ov::element::Type input_type;
        ov::builder::subgraph::DequantizationOperations dequantization_before;
        ov::element::Type preicsionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantization_after;
    };

    TestTransformationParams params;
    std::vector<size_t> block_shape;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ov::PartialShape,
    SpaceToBatchTransformationTestValues> SpaceToBatchTransformationParams;

class SpaceToBatchTransformation : public LayerTransformation, public testing::WithParamInterface<SpaceToBatchTransformationParams> {
public:
    void SetUp() override {
        const ov::PartialShape input_shape = std::get<0>(GetParam());
        const SpaceToBatchTransformationTestValues test_values = std::get<1>(GetParam());

        actualFunction = ov::builder::subgraph::SpaceToBatchFunction::get(
            input_shape,
            test_values.actual.input_type,
            test_values.actual.dequantization_before,
            test_values.block_shape,
            test_values.pads_begin,
            test_values.pads_end,
            test_values.actual.dequantization_after);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::SpaceToBatchTransformation>(test_values.params);
        transform.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::SpaceToBatchFunction::get(
            input_shape,
            test_values.expected.input_type,
            test_values.expected.dequantization_before,
            test_values.block_shape,
            test_values.pads_begin,
            test_values.pads_end,
            test_values.expected.dequantization_after);
    }

    static std::string getTestCaseName(testing::TestParamInfo<SpaceToBatchTransformationParams> obj) {
        const ov::PartialShape shape = std::get<0>(obj.param);
        const SpaceToBatchTransformationTestValues testValues = std::get<1>(obj.param);

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

TEST_P(SpaceToBatchTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(actualFunction, referenceFunction, true, false, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues {
const std::vector<ov::PartialShape> shapes = {
    {1, 3, 100, 171},
};

const std::vector<SpaceToBatchTransformationTestValues> testValues = {
    // per-tensor dequantization
    {
        LayerTransformation::createParamsU8I8(),
        {1, 1, 2, 2},
        {0, 0, 2, 2},
        {0, 0, 2, 3},
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
        {0, 0, 2, 2},
        {0, 0, 2, 3},
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
    SpaceToBatchTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    SpaceToBatchTransformation::getTestCaseName);
} // namespace testValues
