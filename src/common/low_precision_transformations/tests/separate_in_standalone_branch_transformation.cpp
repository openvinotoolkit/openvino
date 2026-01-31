// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"
#include "low_precision/mat_mul.hpp"

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/mat_mul.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "openvino/op/relu.hpp"

namespace {

using namespace testing;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class SeparateInStandaloneBranchTransformationTestValues {
public:
    TestTransformationParams params;
    ov::element::Type precisionBefore;
    DequantizationOperations dequantization;
};

inline std::ostream& operator << (std::ostream& out, const SeparateInStandaloneBranchTransformationTestValues& testValues) {
    return out << "_" << testValues.dequantization;
}

typedef std::tuple<ov::Shape,  // input shape
                   SeparateInStandaloneBranchTransformationTestValues,
                   bool>  // is constant path
    SeparateInStandaloneBranchTransformationParams;

class SeparateInStandaloneBranchTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<SeparateInStandaloneBranchTransformationParams> {
public:
    void SetUp() override {
        ov::Shape inputShape;
        SeparateInStandaloneBranchTransformationTestValues testValues;
        bool isConstantPath;
        std::tie(inputShape, testValues, isConstantPath) = GetParam();
        const auto& precision = testValues.precisionBefore;

        const auto createActualFunction = [&](const DequantizationOperations& dequantizations) {
            ov::ParameterVector params;
            std::shared_ptr<ov::Node> input;
            if (isConstantPath) {
                input = ov::op::v0::Constant::create(precision, inputShape, std::vector<float>{137.f});
            } else {
                params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, inputShape));
                input = params.back();
            }
            const auto dequantizationsNode = makeDequantization(input, dequantizations);

            const auto reshape1 = std::make_shared<ov::op::v1::Reshape>(
                dequantizationsNode,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>({0, -1})),
                true);
            reshape1->set_friendly_name("reshape1");

            const auto reshape2 = std::make_shared<ov::op::v1::Reshape>(
                dequantizationsNode,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>({0, -1})),
                true);
            reshape2->set_friendly_name("reshape2");

            return std::make_shared<ov::Model>(ov::OutputVector{reshape1, reshape2},
                                               params,
                                               "SeparateInStandaloneBranchTransformation");
        };
        actualFunction = createActualFunction(testValues.dequantization);
        const auto result = actualFunction->get_results()[0];
        ov::pass::low_precision::NetworkHelper::separateInStandaloneBranch(result->get_input_node_shared_ptr(0));

        const auto createReferenceFunction = [&](DequantizationOperations dequantization) {
            // Note: separateInStandaloneBranch normalizes dequantization so constant indexes become equal to 1
            if (!dequantization.subtract.empty())
                dequantization.subtract.constantIndex = 1;
            if (!dequantization.multiply.empty())
                dequantization.multiply.constantIndex = 1;

            ov::ParameterVector params;
            std::shared_ptr<ov::Node> input;
            if (isConstantPath) {
                input = ov::op::v0::Constant::create(precision, inputShape, std::vector<float>{137.f});
            } else {
                params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, inputShape));
                input = params.back();
            }

            const auto reshape1 = std::make_shared<ov::op::v1::Reshape>(
                makeDequantization(input, dequantization),
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>({0, -1})),
                true);
            reshape1->set_friendly_name("reshape1");

            const auto reshape2 = std::make_shared<ov::op::v1::Reshape>(
                makeDequantization(input, dequantization),
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>({0, -1})),
                true);
            reshape2->set_friendly_name("reshape2");

            return std::make_shared<ov::Model>(ov::OutputVector{reshape1, reshape2},
                                               params,
                                               "SeparateInStandaloneBranchTransformation");
        };
        referenceFunction = createReferenceFunction(testValues.dequantization);
    }

    static std::string getTestCaseName(testing::TestParamInfo<SeparateInStandaloneBranchTransformationParams> obj) {
        const auto& [shapes, testValues, isConstantPath] = obj.param;
        std::stringstream ss;
        ss << shapes << "_" << testValues << "_is_constant_path_" << isConstantPath;
        return ss.str();
    }
};

TEST_P(SeparateInStandaloneBranchTransformation, CompareFunctions) {
    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
    auto comparator = FunctionsComparator::with_default()
                          .enable(FunctionsComparator::CmpValues::CONSUMERS_COUNT)
                          .enable(FunctionsComparator::CmpValues::CONST_VALUES)
                          .enable(FunctionsComparator::CmpValues::NAMES);
    auto res = comparator.compare(actualFunction, referenceFunction);
    ASSERT_TRUE(res.valid) << res.message;
}

const std::vector<ov::Shape> shapes = {
    { 1, 3, 9, 9 },
};

std::vector<SeparateInStandaloneBranchTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8U8().setSupportAsymmetricQuantization(true),
        ov::element::u8,
        { ov::element::f32, { 127.f }, { 0.02f } }
    },
    {
        LayerTransformation::createParamsU8U8(),
        ov::element::u8,
        { ov::element::f32, { 127.f }, {} }
    },
    {
        LayerTransformation::createParamsU8U8().setSupportAsymmetricQuantization(true),
        ov::element::u8,
        {
            ov::element::f32,
            { {127.f}, ov::element::f32, {}, true, 1ul, ov::element::u8, true},
            { 0.02f }
        }
    },
    {
        LayerTransformation::createParamsU8U8(),
        ov::element::u8,
        {
            ov::element::f32,
            { {127.f}, ov::element::f32, {}, false, 0ul},
            { {0.02f}, ov::element::f32, {}, false, 0ul }
        }
    },
    {
        LayerTransformation::createParamsU8U8(),
        ov::element::u8,
        {
            ov::element::f32,
            {},
            {{0.02f}, ov::element::f32, {1, 3, 1, 1}, false, 0ul}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    SeparateInStandaloneBranchTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues),
        ::testing::Values(false, true)),
    SeparateInStandaloneBranchTransformation::getTestCaseName);
} // namespace
