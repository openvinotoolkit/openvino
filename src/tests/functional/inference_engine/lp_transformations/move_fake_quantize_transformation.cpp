// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <low_precision/concat.hpp>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/relu.hpp>

#include <low_precision/low_precision.hpp>

#include "low_precision/move_fake_quantize.hpp"
#include <low_precision/fake_quantize_decomposition.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/move_fake_quantize_function.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/relu_function.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

namespace {

class MoveFakeQuantizeTransformationActualValues {
public:
    size_t number_of_operations;
    std::vector<ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant> fakeQuantizeBefore;
    ngraph::builder::subgraph::DequantizationOperations::Convert convertBefore;
    ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
    std::string operation;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeAfter;
    ngraph::builder::subgraph::DequantizationOperations::Convert convertAfter;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
};

inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeTransformationActualValues& values) {
    return out << "_" <<
        values.number_of_operations << "_" <<
        values.convertBefore.outPrecision << "_" <<
        values.dequantizationBefore << "_" <<
        values.operation << "_" <<
        values.fakeQuantizeAfter << "_" <<
        values.convertAfter.outPrecision << "_" <<
        values.dequantizationAfter;
}

class MoveFakeQuantizeTransformationResultValues {
public:
    size_t number_of_operations;
    std::vector<ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant> fakeQuantizeBefore;
    ngraph::builder::subgraph::DequantizationOperations::Convert convertBefore;
    ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
    std::string operation;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeAfter;
    ngraph::builder::subgraph::DequantizationOperations::Convert convertAfter;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    ngraph::element::Type precisionAfterOperation;
};

inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeTransformationResultValues& values) {
    return out << "_" <<
        values.convertBefore.outPrecision << "_" <<
        values.dequantizationBefore << "_" <<
        values.operation << "_" <<
        values.fakeQuantizeAfter << "_" <<
        values.convertAfter << "_" <<
        values.dequantizationAfter;
}

class MoveFakeQuantizeTransformationTestValues {
public:
    MoveFakeQuantizeTransformationTestValues() = default;
    MoveFakeQuantizeTransformationTestValues(
        const TestTransformationParams& params,
        const bool multiChannels,
        const  std::int64_t axis,
        const MoveFakeQuantizeTransformationActualValues& actual,
        const MoveFakeQuantizeTransformationResultValues& result,
        const bool addNotPrecisionPreservedOperation = false,
        const bool checkIntervalsAlignmentAttributes = true) :
        params(params),
        multiChannels(multiChannels),
        axis(axis),
        actual(actual),
        result(result) {}

    TestTransformationParams params;
    bool multiChannels;
    std::int64_t axis;
    MoveFakeQuantizeTransformationActualValues actual;
    MoveFakeQuantizeTransformationResultValues result;
    // add not precision preserved operation to set output precision for FakeQuantize
    // don't set to 'true' by default to keep test cases with tested operation as output
};

inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeTransformationTestValues& values) {
    return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ngraph::element::Type,
    std::vector<ngraph::PartialShape>,
    MoveFakeQuantizeTransformationTestValues,
    bool
> MoveFakeQuantizeTransformationParams;

class MoveFakeQuantizeTransformation : public LayerTransformation, public testing::WithParamInterface<MoveFakeQuantizeTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        std::vector<ngraph::PartialShape> inputShapes = std::get<1>(GetParam());
        //const auto shape = std::get<1>(GetParam());
        MoveFakeQuantizeTransformationTestValues testValues = std::get<2>(GetParam());
        const bool oneInputWithSplit = std::get<3>(GetParam());
        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.actual.dequantizationBefore.multiply.empty()) {
            testValues.actual.dequantizationBefore.multiply.outPrecision = precision;
        }

        IntervalsAlignmentSharedValue::Interval interval{ -1.28f, 2.55f };

        actualFunction = ngraph::builder::subgraph::MoveFakeQuantize::get(
            precision,
            inputShapes,
            testValues.actual.number_of_operations,
            testValues.actual.fakeQuantizeBefore,
            testValues.actual.convertBefore,
            testValues.actual.dequantizationBefore,
            testValues.actual.operation,
            testValues.actual.fakeQuantizeAfter,
            testValues.actual.convertAfter,
            testValues.actual.dequantizationAfter,
            {
                PrecisionPreservedAttribute(true),
                IntervalsAlignmentAttribute(interval, 256),
                QuantizationAlignmentAttribute(false)
            },
            ngraph::element::undefined,
            testValues.axis,
            oneInputWithSplit);

        auto supportedPrecisionsOnActivation = std::vector<ngraph::pass::low_precision::PrecisionsRestriction>({
                ngraph::pass::low_precision::PrecisionsRestriction::create<ngraph::opset1::AvgPool>({{0, testValues.params.precisionsOnActivations}})
            });

        auto quantizationRestrictions = testValues.multiChannels ?
            std::vector<ngraph::pass::low_precision::QuantizationGranularityRestriction>() :
            std::vector<ngraph::pass::low_precision::QuantizationGranularityRestriction>({
                ngraph::pass::low_precision::QuantizationGranularityRestriction::create<ngraph::opset1::AvgPool>()
                });

        const auto params = TestTransformationParams::toParams(testValues.params);
        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::low_precision::MoveFakeQuantize>(params);
        manager.run_passes(actualFunction);

        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.result.dequantizationAfter.multiply.empty()) {
            testValues.result.dequantizationAfter.multiply.outPrecision = precision;
        }

        if (!testValues.params.updatePrecisions &&
            (precision == ngraph::element::f32) &&
            !testValues.result.dequantizationAfter.convert.empty()) {
            testValues.result.dequantizationAfter.convert = {};
        }

        referenceFunction = ngraph::builder::subgraph::MoveFakeQuantize::get(
            precision,
            inputShapes,
            testValues.result.number_of_operations,
            testValues.result.fakeQuantizeBefore,
            testValues.result.convertBefore,
            testValues.result.dequantizationBefore,
            testValues.result.operation,
            testValues.result.fakeQuantizeAfter,
            testValues.result.convertAfter,
            testValues.result.dequantizationAfter,
            {
                PrecisionPreservedAttribute(true),
                IntervalsAlignmentAttribute(interval, 256),
                QuantizationAlignmentAttribute(false)
            },
            testValues.result.precisionAfterOperation,
            testValues.axis,
            oneInputWithSplit);
    }
    static std::string getTestCaseName(testing::TestParamInfo<MoveFakeQuantizeTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const std::vector<ngraph::PartialShape> shape = std::get<1>(obj.param);
        const MoveFakeQuantizeTransformationTestValues testValues = std::get<2>(obj.param);
        const bool oneInputWithSplit = std::get<3>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape[0], testValues.params) << "_" <<
            (testValues.multiChannels ? "multiChannels_" : "notMultiChannels_") <<
            "axis_" << testValues.axis << "_" <<
            testValues.actual << "_" <<
            testValues.result << "_" <<
            oneInputWithSplit;
        return result.str();
    }
};

TEST_P(MoveFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, true, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";

    const auto actualFakeQuantizes = LayerTransformation::get<opset1::FakeQuantize>(actualFunction);
    ASSERT_TRUE(checkIfOutputAttributesSharedValuesAreTheSame<PrecisionsAttribute>(actualFakeQuantizes)) <<
        "PrecisionsAttribute are not the same";
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

namespace perTensorValues {
const std::vector<std::vector<ngraph::PartialShape>> shapes = {
    {{ 1, 1, 9, 9 }, { 1, 1, 9, 9 }},
    {{ 4, 3, 9, 9 }, { 4, 3, 9, 9 }},
    {{ -1, -1, -1, -1 }, { -1, -1, -1, -1 }}
};

const std::vector<MoveFakeQuantizeTransformationTestValues> testValues = {
     // without operation
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            2,
            {},
            {},
            {},
            "",
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
            {},
            {}
        },
        {
            2,
            {{ 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
            {},
            {},
            "",
            {},
            {},
            {},
        }
    },
    // with ReLU
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            2,
            {},
            {},
            {},
            "relu",
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
            {},
            {}
        },
        {
            2,
            {{ 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
            {},
            {},
            "relu",
            {},
            {},
            {},
        }
    },
    // concat by batch
    {
        LayerTransformation::createParamsU8I8(),
        false,
        0,
        {
            2,
            {},
            {},
            {},
            "",
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
            {},
            {}
        },
        {
            2,
            {{ 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
            {},
            {},
            "",
            {},
            {},
            {}
        }
    },
    // Q/DQ
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            2,
            {},
            {},
            {},
            "",
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
            { ngraph::element::u8 },
            {
                { element::f32 },
                {},
                { 0.01f }
            },
        },
        {
            2,
            {{ 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
            { ngraph::element::u8 },
            {
                { element::f32 },
                {},
                { 0.01f }
            },
            "",
            {},
            {},
            {},
        }
    },
    // Q/DQ with ReLU
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            2,
            {},
            {},
            {},
            "relu",
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
            { ngraph::element::u8 },
            {
                { element::f32 },
                {},
                { 0.01f }
            },
        },
        {
            2,
            {{ 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
            { ngraph::element::u8 },
            {
                { element::f32 },
                {},
                { 0.01f }
            },
            "relu",
            {},
            {},
            {},
        }
    },
    // Q/DQ with subtract
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            2,
            {},
            {},
            {},
            "",
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
            { ngraph::element::u8 },
            {
                { element::f32 },
                { 0.01f },
                { 0.01f }
            },
        },
        {
            2,
            {{ 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}},
            { ngraph::element::u8 },
            {
                { element::f32 },
                { 0.01f },
                { 0.01f }
            },
            "",
            {},
            {},
            {},
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MoveFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn({ false, true })),
    MoveFakeQuantizeTransformation::getTestCaseName);
} // namespace perTensorValues

namespace perChannelValues {
const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<std::vector<ngraph::PartialShape>> shapes = {
    {{ 1, 1, 224, 224 }, { 1, 2, 224, 224 }},
    {{ 4, 1, 9, 9 }, { 4, 2, 9, 9 }},
    {{ -1, 1, -1, -1 }, { -1, 2, -1, -1 }},
};

const std::vector<MoveFakeQuantizeTransformationTestValues> testValues = {
    // multi-chanels
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {
            2,
            {},
            {},
            {},
            "",
            {
                256ul,
                {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
                {-2.66068696975708f}, {2.6399004459381104f},
                {-31.695816040039062f, -35.69844055175781f, -49.126914978027344f},
                {277.8320007324219f, 267.07110595703125f, 254.99429321289062f}
            },
            {},
            {}
        },
        {
            2,
            {
                {256ul,
                {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}},
                {-2.66068696975708f}, {2.6399004459381104f}, {-31.695816040039062f}, {277.8320007324219f}},
                {256ul,
                {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 2, 1, 1}, {1, 2, 1, 1}},
                {-2.66068696975708f}, {2.6399004459381104f},
                {-35.69844055175781f, -49.126914978027344f},
                {267.07110595703125f, 254.99429321289062f}}
            },
            {},
            {},
            "",
            {},
            {},
            {},
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {
            2,
            {},
            {},
            {},
            "",
            {
                256ul,
                {{}, {}, {1, 3, 1, 1}, {1, 3, 1, 1}},
                {-2.6f}, {2.6f},
                {-31.7f, -35.7f, -49.1f},
                {277.8f, 267.f, 254.9f}
            },
            {},
            {}
        },
        {
            2,
            {
                {256ul,
                {{}, {}, {1, 1, 1, 1}, {1, 1, 1, 1}},
                {-2.6}, {2.6f}, {-31.7f}, {277.8f}},
                {256ul,
                {{}, {}, {1, 2, 1, 1}, {1, 2, 1, 1}},
                {-2.6f}, {2.6f},
                {-35.7f, -49.1f},
                {267.f, 254.9f}}
            },
            {},
            {},
            "",
            {},
            {},
            {},
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {
            2,
            {},
            {},
            {},
            "",
            {
                256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
                {-31.7f, -35.7f, -49.1f},
                {277.8f, 267.f, 254.9f},
                {-2.6f}, {2.6f},
            },
            {},
            {}
        },
        {
            2,
            {
                {256ul,
                {{1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}},
                {-31.7f}, {277.8f}, {-2.6}, {2.6f}},
                {256ul,
                {{1, 2, 1, 1}, {1, 2, 1, 1}, {}, {}},
                {-35.7f, -49.1f},
                {267.f, 254.9f},
                {-2.6f}, {2.6f}}
            },
            {},
            {},
            "",
            {},
            {},
            {},
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MoveFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn({ false })),
    MoveFakeQuantizeTransformation::getTestCaseName);
} // namespace perChannelValues

namespace testValues3 {
    const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
    };

    const std::vector<std::vector<ngraph::PartialShape>> shapes = {
        {{ 1, 1}, { 1, 2}},
        {{ 4, 1}, { 4, 2}}
    };
    const std::vector<MoveFakeQuantizeTransformationTestValues> testValues = {
        // 2D shape
        {
            LayerTransformation::createParamsU8I8(),
            true,
            1,
            {
                2,
                {},
                {},
                {},
                "",
                {
                    256ul,
                    {{1, 3}, {1, 3}, {}, {}},
                    {-31.7f, -35.7f, -49.1f},
                    {277.8f, 267.f, 254.9f},
                    {-2.6f}, {2.6f},
                },
                {},
                {}
            },
            {
                2,
                {
                    {256ul,
                    {{1, 1}, {1, 1}, {}, {}},
                    {-31.7f}, {277.8f}, {-2.6}, {2.6f}},
                    {256ul,
                    {{1, 2}, {1, 2}, {}, {}},
                    {-35.7f, -49.1f},
                    {267.f, 254.9f},
                    {-2.6f}, {2.6f}}
                },
                {},
                {},
                "",
                {},
                {},
                {},
            }
        },
    };
    INSTANTIATE_TEST_SUITE_P(
        smoke_LPT,
        MoveFakeQuantizeTransformation,
        ::testing::Combine(
            ::testing::ValuesIn(precisions),
            ::testing::ValuesIn(shapes),
            ::testing::ValuesIn(testValues),
            ::testing::ValuesIn({ false })),
        MoveFakeQuantizeTransformation::getTestCaseName);
} // namespace testValues3

namespace NegativeTestValues {
const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32
};

const std::vector<std::vector<ngraph::PartialShape>> shapes = {
    {{-1, -1, -1, -1}, {-1, -1, -1, -1}},
};
const std::vector<MoveFakeQuantizeTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        true,
        1,
        {
            2,
            {},
            {},
            {},
            "",
            {
                256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
                {-31.7f, -35.7f, -49.1f},
                {277.8f, 267.f, 254.9f},
                {-2.6f}, {2.6f},
            },
            {},
            {}
        },
        {
            2,
            {},
            {},
            {},
            "",
            {
                256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
                {-31.7f, -35.7f, -49.1f},
                {277.8f, 267.f, 254.9f},
                {-2.6f}, {2.6f},
            },
            {},
            {}
        },
    },
};
INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MoveFakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn({ false })),
    MoveFakeQuantizeTransformation::getTestCaseName);
} // namespace NegativeTestValues
} // namespace
