// Copyright (C) 2018-2024 Intel Corporation
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

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/mat_mul.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace {

using namespace testing;
using namespace ov::pass;

class SeparateInStandaloneBranchTransformationTestValues {
public:
    TestTransformationParams params;
    ov::element::Type precisionBefore;
    ov::builder::subgraph::DequantizationOperations dequantization;
};

inline std::ostream& operator << (std::ostream& out, const SeparateInStandaloneBranchTransformationTestValues& testValues) {
    return out << "_" << testValues.dequantization;
}

typedef std::tuple<
    ov::Shape,
    SeparateInStandaloneBranchTransformationTestValues> SeparateInStandaloneBranchTransformationParams;

class SeparateInStandaloneBranchTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<SeparateInStandaloneBranchTransformationParams> {
public:
    void SetUp() override {
        const ov::Shape shape = std::get<0>(GetParam());
        const SeparateInStandaloneBranchTransformationTestValues testValues = std::get<1>(GetParam());

        const auto createActualFunction = [](
            const ov::element::Type precision,
            const ov::Shape& inputShape,
            const ov::builder::subgraph::DequantizationOperations& dequantizations) -> std::shared_ptr<ov::Model> {
            const std::shared_ptr<ov::op::v0::Parameter> input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);
            const auto relu = std::make_shared<ov::op::v0::Relu>(input);
            const auto dequantizationsNode = ov::builder::subgraph::makeDequantization(relu, dequantizations);

            const std::shared_ptr<ov::Node> reshape1 = std::make_shared<ov::op::v1::Reshape>(
                dequantizationsNode,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{ 2 }, std::vector<double>({0, -1})),
                true);
            reshape1->set_friendly_name("reshape1");

            const std::shared_ptr<ov::Node> reshape2 = std::make_shared<ov::op::v1::Reshape>(
                dequantizationsNode,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{ 2 }, std::vector<double>({0, -1})),
                true);
            reshape2->set_friendly_name("reshape2");

            return std::make_shared<ov::Model>(
                ov::ResultVector{
                    std::make_shared<ov::op::v0::Result>(reshape1),
                    std::make_shared<ov::op::v0::Result>(reshape2)
                },
                std::vector<std::shared_ptr<ov::op::v0::Parameter>> { input },
                "SeparateInStandaloneBranchTransformation");
        };
        actualFunction = createActualFunction(testValues.precisionBefore, shape, testValues.dequantization);
        const auto result = actualFunction->get_results()[0];
        ov::pass::low_precision::NetworkHelper::separateInStandaloneBranch(result->get_input_node_shared_ptr(0));

        const auto createReferenceFunction = [](
            const ov::element::Type precision,
            const ov::Shape& inputShape,
            ov::builder::subgraph::DequantizationOperations dequantization) -> std::shared_ptr<ov::Model> {
            // Note: separateInStandaloneBranch normalizes dequantization so constant indexes become equal to 1
            if (!dequantization.subtract.empty())
                dequantization.subtract.constantIndex = 1;
            if (!dequantization.multiply.empty())
                dequantization.multiply.constantIndex = 1;

            const std::shared_ptr<ov::op::v0::Parameter> input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);
            const auto relu = std::make_shared<ov::op::v0::Relu>(input);

            const std::shared_ptr<ov::Node> reshape1 = std::make_shared<ov::op::v1::Reshape>(
                ov::builder::subgraph::makeDequantization(relu, dequantization),
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{ 2 }, std::vector<double>({0, -1})),
                true);
            reshape1->set_friendly_name("reshape1");

            const std::shared_ptr<ov::Node> reshape2 = std::make_shared<ov::op::v1::Reshape>(
                ov::builder::subgraph::makeDequantization(relu, dequantization),
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{ 2 }, std::vector<double>({0, -1})),
                true);
            reshape2->set_friendly_name("reshape2");

            return std::make_shared<ov::Model>(
                ov::ResultVector{
                    std::make_shared<ov::op::v0::Result>(reshape1),
                    std::make_shared<ov::op::v0::Result>(reshape2)
                },
                std::vector<std::shared_ptr<ov::op::v0::Parameter>> { input },
                "SeparateInStandaloneBranchTransformation");
        };
        referenceFunction = createReferenceFunction(testValues.precisionBefore, shape, testValues.dequantization);
    }

    static std::string getTestCaseName(testing::TestParamInfo<SeparateInStandaloneBranchTransformationParams> obj) {
        ov::Shape shapes;
        SeparateInStandaloneBranchTransformationTestValues testValues;
        std::tie(shapes, testValues) = obj.param;

        std::stringstream ss;
        ss << shapes << "_" << testValues;
        return ss.str();
    }
};

TEST_P(SeparateInStandaloneBranchTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
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
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    SeparateInStandaloneBranchTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    SeparateInStandaloneBranchTransformation::getTestCaseName);

} // namespace
