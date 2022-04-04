// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include "ngraph_functions/builders.hpp"
#include <transformations/init_node_info.hpp>
#include <ngraph_helpers/lpt_ngraph_functions/include/lpt_ngraph_functions/common/dequantization_operations.hpp>
#include <ngraph_helpers/lpt_ngraph_functions/include/lpt_ngraph_functions/common/builders.hpp>
#include "transformations/low_precision/pwl_transformation.hpp"
#include "ops/pwl.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"


namespace testing {

using namespace ov;

class PwlLPTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        std::vector<float> m;
        std::vector<float> b;
        std::vector<float> knots;
    };

    class Expected {
    public:
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        std::vector<float> m;
        std::vector<float> b;
        std::vector<float> knots;
        bool transformed;
    };

    Actual actual;
    Expected expected;
};

typedef std::tuple <
    Shape,         // input shape
    element::Type, // network precision
    element::Type, // input precision
    PwlLPTransformationTestValues> PwlLPTransformationParams;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "{";
    if (!v.empty()) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            os << v[i] << ", ";
        }
        os << v.back();
    }
    os << "}";
    return os;
}

class PwlLPTransformationTest: public CommonTestUtils::TestsCommon,
                             public ::testing::WithParamInterface<PwlLPTransformationParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PwlLPTransformationParams>& obj) {
        Shape inputShape;
        element::Type netPrecision;
        element::Type inputPrecision;
        PwlLPTransformationTestValues testValues;
        std::tie(inputShape, netPrecision, inputPrecision, testValues) = obj.param;

        std::ostringstream result;
        result << "inputShape=" << inputShape << "_";
        result << "netPrecision=" << netPrecision << "_";
        result << "inputPrecision=" << inputPrecision << "_";

        result << "actual:" << "_";
        result << "deqOps=" << testValues.actual.dequantization << "_";
        result << "m=" << testValues.actual.m << "_";
        result << "b=" << testValues.actual.b << "_";
        result << "knots=" << testValues.actual.knots << "_";


        result << "expected:" << "_";
        result << "deqOpsBefore=" << testValues.expected.dequantizationBefore << "_";

        return result.str();
    }
    void SetUp() override;
    virtual void Validate();
    virtual void Run();
public:
    std::shared_ptr<ngraph::Function> func_, ref_func_;
};

void PwlLPTransformationTest::Run() {
    SetUp();
    Validate();
}

void PwlLPTransformationTest::SetUp() {
    Shape inputShape;
    element::Type netPrecision;
    element::Type inputPrecision;
    PwlLPTransformationTestValues testValues;
    if (!testValues.actual.dequantization.multiply.empty()) {
        testValues.actual.dequantization.multiply.outPrecision = netPrecision;
    }
    if (!testValues.expected.dequantizationBefore.multiply.empty()) {
        testValues.expected.dequantizationBefore.multiply.outPrecision = netPrecision;
    }
    std::tie(inputShape, netPrecision, inputPrecision, testValues) = GetParam();

    // test function
    {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
        const auto dequantizationOp = makeDequantization(input, testValues.actual.dequantization);
        auto m = ngraph::builder::makeConstant(netPrecision, inputShape, testValues.actual.m);
        auto b = ngraph::builder::makeConstant(netPrecision, inputShape, testValues.actual.b);
        auto knots = ngraph::builder::makeConstant(netPrecision, inputShape, testValues.actual.knots);

        std::shared_ptr<ov::Node> pwl;
        if (!testValues.actual.dequantization.empty() || inputPrecision.is_real()) {
            pwl = std::make_shared<intel_gna::op::Pwl>(dequantizationOp, m, b, knots);
        } else {
            pwl = std::make_shared<ngraph::op::TypeRelaxed<intel_gna::op::Pwl>>(
                std::vector<element::Type>{ element::f32, element::f32, element::f32, element::f32 },
                std::vector<element::Type>{ element::f32 },
                ngraph::op::TemporaryReplaceOutputType(dequantizationOp, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(m, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(b, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(knots, element::f32).get());
            as_type_ptr<ngraph::op::TypeRelaxed<intel_gna::op::Pwl>>(pwl)->set_overridden_output_type(inputPrecision);
        }
        pwl->set_friendly_name("output");

        auto result = std::make_shared<ngraph::opset8::Result>(pwl);
        func_ = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{result},
            ngraph::ParameterVector{input},
            "PWLTransformation");
    }

    // ref function
    {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
        const auto dequantizationOpBefore = makeDequantization(input, testValues.expected.dequantizationBefore);

        element::Type constPrecision = netPrecision;
        if (testValues.expected.transformed) {
            constPrecision = element::f32;
        }
        auto m = ngraph::builder::makeConstant(constPrecision, inputShape, testValues.expected.m);
        auto b = ngraph::builder::makeConstant(constPrecision, inputShape, testValues.expected.b);
        auto knots = ngraph::builder::makeConstant(constPrecision, inputShape, testValues.expected.knots);

        std::shared_ptr<ov::Node> pwl;
        if (!testValues.expected.dequantizationBefore.empty() || inputPrecision.is_real()) {
            pwl = std::make_shared<intel_gna::op::Pwl>(dequantizationOpBefore, m, b, knots);
        } else {
            pwl = std::make_shared<ngraph::op::TypeRelaxed<intel_gna::op::Pwl>>(
                std::vector<element::Type>{ element::f32, element::f32, element::f32, element::f32 },
                std::vector<element::Type>{ element::f32 },
                ngraph::op::TemporaryReplaceOutputType(dequantizationOpBefore, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(m, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(b, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(knots, element::f32).get());
            as_type_ptr<ngraph::op::TypeRelaxed<intel_gna::op::Pwl>>(pwl)->set_overridden_output_type(inputPrecision);
        }
        pwl->set_friendly_name("output");

        auto result = std::make_shared<ngraph::opset8::Result>(pwl);
        ref_func_ = std::make_shared<ngraph::Function>(
                ngraph::ResultVector{result},
                ngraph::ParameterVector{input},
                "PWLTransformation");
    }
}

void PwlLPTransformationTest::Validate() {
    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<ngraph::pass::low_precision::PWLTransformation>(
        ngraph::pass::low_precision::LayerTransformation::Params(
            true,
            element::f32,
            ngraph::pass::low_precision::precision_set::int8_int16_int32_support));
    m.run_passes(func_);

    auto res = compare_functions(func_, ref_func_, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

std::vector<Shape> inputShapes = {
    {1, 3}
};

element::TypeVector netPrecisions = {
    element::f16,
    element::f32
};

element::TypeVector inputPrecisions = {
    element::i8, element::u8,
    element::i16, element::u16,
    element::i32, element::u32,
};

std::vector<PwlLPTransformationTestValues> testValues = {
    // general case
    {
        {
            {{element::f32}, {0.5f}, {2.f}},
            {1.f, 2.f, 4.f},
            {4.f, 2.f, 1.f},
            {-8.f, 0.5f, 2.f}
        },
        {
            {},
            {2.f, 4.f, 8.f},
            {3.f, 0.f, -3.f},
            {-3.5f, 0.75f, 1.5f},
            true
        },
    },
        // general case
    {
        {
            {{element::f32}, {}, {2.f}},
            {1.f, 2.f, 4.f},
            {4.f, 2.f, 1.f},
            {-8.f, 0.5f, 2.f}
        },
        {
            {},
            {2.f, 4.f, 8.f},
            {4.f, 2.f, 1.f},
            {-4.f, 0.25f, 1.f},
            true
        },
    },
    // no dequantizations
    {
        {
            {},
            {1.f, 2.f, 4.f},
            {4.f, 2.f, 1.f},
            {-8.f, 0.5f, 2.f}
        },
        {
            {},
            {1.f, 2.f, 4.f},
            {4.f, 2.f, 1.f},
            {-8.f, 0.5f, 2.f},
            false
        },
    },
    // per-channel dequantizations
    {
        {
            {{element::f32}, {{0.5f, 0.6f, 0.7f}}, {{2.f, 3.f, 4.f}}},
            {1.f, 2.f, 4.f},
            {4.f, 2.f, 1.f},
            {-8.f, 0.5f, 2.f}
        },
        {
            {{element::f32}, {{0.5f, 0.6f, 0.7f}}, {{2.f, 3.f, 4.f}}},
            {1.f, 2.f, 4.f},
            {4.f, 2.f, 1.f},
            {-8.f, 0.5f, 2.f},
            false
        },
    },
};

TEST_P(PwlLPTransformationTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests, PwlLPTransformationTest,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(testValues)),
    PwlLPTransformationTest::getTestCaseName);

} // namespace testing
