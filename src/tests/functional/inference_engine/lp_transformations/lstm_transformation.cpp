// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <low_precision/common/operation_per_tensor_quantization_restriction.hpp>
#include <low_precision/common/operation_precision_restriction.hpp>
#include <low_precision/lstm.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/fold_convert.hpp>
#include <low_precision/fuse_convert.hpp>
#include <low_precision/fuse_multiply_to_fake_quantize.hpp>
#include <low_precision/fuse_subtract_to_fake_quantize.hpp>
#include <low_precision/rt_info/intervals_alignment_attribute.hpp>
#include <low_precision/rt_info/precision_preserved_attribute.hpp>
#include <low_precision/rt_info/quantization_alignment_attribute.hpp>
#include <memory>
#include <sstream>
#include <vector>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "layer_transformation.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/lstm_function.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

namespace {

class LSTMTransformationValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize_X;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert_X;
    ngraph::builder::subgraph::DequantizationOperations dequantization_X;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize_H;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert_H;
    ngraph::builder::subgraph::DequantizationOperations dequantization_H;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize_W;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert_W;
    ngraph::builder::subgraph::DequantizationOperations dequantization_W;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize_R;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert_R;
    ngraph::builder::subgraph::DequantizationOperations dequantization_R;
    ngraph::element::Type precisionAfterOperation;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
};

inline std::ostream& operator<<(std::ostream& out, const LSTMTransformationValues& values) {
    return out << "_" << values.fakeQuantize_X << "_" << values.convert_X << "_" << values.dequantization_X <<
                  "_" << values.fakeQuantize_H << "_" << values.convert_H << "_" << values.dequantization_H <<
                  "_" << values.fakeQuantize_W << "_" << values.convert_W << "_" << values.dequantization_W <<
                  "_" << values.fakeQuantize_R << "_" << values.convert_R << "_" << values.dequantization_R;
}

class LSTMTransformationTestValues {
public:
    LSTMTransformationTestValues() = default;
    LSTMTransformationTestValues(const TestTransformationParams& params,
                                 const LSTMFunction::RNNType type,
                                 const LSTMTransformationValues& actual,
                                 const LSTMTransformationValues& result,
                                 const bool addNotPrecisionPreservedOperation = false,
                                 const bool checkIntervalsAlignmentAttributes = true)
        : params(params),
          type(type),
          actual(actual),
          result(result),
          addNotPrecisionPreservedOperation(addNotPrecisionPreservedOperation),
          checkIntervalsAlignmentAttributes(checkIntervalsAlignmentAttributes) {}

    TestTransformationParams params;
    LSTMFunction::RNNType type;
    LSTMTransformationValues actual;
    LSTMTransformationValues result;
    // add not precision preserved operation to set output precision for FakeQuantize
    // don't set to 'true' by default to keep test cases with tested operation as output
    bool addNotPrecisionPreservedOperation;
    bool checkIntervalsAlignmentAttributes;
};

inline std::ostream& operator<<(std::ostream& out, const LSTMTransformationTestValues& values) {
    return out << "_" << values.actual << "_" << values.result;
}

typedef std::tuple<ngraph::element::Type, std::vector<ngraph::PartialShape>, std::vector<ngraph::Shape>, LSTMTransformationTestValues>
    LSTMTransformationParams;

class LSTMTransformation : public LayerTransformation, public testing::WithParamInterface<LSTMTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const std::vector<ngraph::PartialShape> activations_shapes = std::get<1>(GetParam());
        const std::vector<ngraph::Shape> weights_shapes = std::get<2>(GetParam());
        LSTMTransformationTestValues testValues = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::LSTMFunction::get(precision,
                                                                      activations_shapes,
                                                                      weights_shapes,
                                                                      testValues.type,
                                                                      {
                                                                          testValues.actual.fakeQuantize_X,
                                                                          testValues.actual.fakeQuantize_H,
                                                                          testValues.actual.fakeQuantize_W,
                                                                          testValues.actual.fakeQuantize_R
                                                                      },
                                                                      {
                                                                          testValues.actual.convert_X,
                                                                          testValues.actual.convert_H,
                                                                          testValues.actual.convert_W,
                                                                          testValues.actual.convert_R
                                                                      },
                                                                      {
                                                                          testValues.actual.dequantization_X,
                                                                          testValues.actual.dequantization_H,
                                                                          testValues.actual.dequantization_W,
                                                                          testValues.actual.dequantization_R
                                                                      },
                                                                      {},
                                                                      ngraph::element::undefined,
                                                                      {});

        const auto params = TestTransformationParams::toParams(testValues.params);

        SimpleLowPrecisionTransformer transformer;
        transformer.commonGraphRewrite->add_matcher<ngraph::pass::low_precision::LSTMTransformation>(params);
        transformer.transform(actualFunction);

        SimpleLowPrecisionTransformer clenup_transformer;
        clenup_transformer.commonGraphRewrite->add_matcher<ngraph::pass::low_precision::FoldConvertTransformation>(params);
        clenup_transformer.commonGraphRewrite->add_matcher<ngraph::pass::low_precision::FuseConvertTransformation>(params);
        clenup_transformer.commonGraphRewrite->add_matcher<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>(params);
        clenup_transformer.commonGraphRewrite->add_matcher<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>(params);
        clenup_transformer.transform(actualFunction);

        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.result.dequantizationAfter.multiply.empty()) {
            testValues.result.dequantizationAfter.multiply.outPrecision = precision;
        }

        if (!testValues.params.updatePrecisions && (precision == ngraph::element::f32) &&
            !testValues.result.dequantizationAfter.convert.empty()) {
            testValues.result.dequantizationAfter.convert = {};
        }

        IntervalsAlignmentSharedValue::Interval interval{-1.28f, 2.55f};

        referenceFunction = ngraph::builder::subgraph::LSTMFunction::get(precision,
                                                                         activations_shapes,
                                                                         weights_shapes,
                                                                         testValues.type,
                                                                         {
                                                                            testValues.result.fakeQuantize_X,
                                                                            testValues.result.fakeQuantize_H,
                                                                            testValues.result.fakeQuantize_W,
                                                                            testValues.result.fakeQuantize_R
                                                                         },
                                                                         {
                                                                            testValues.result.convert_X,
                                                                            testValues.result.convert_H,
                                                                            testValues.result.convert_W,
                                                                            testValues.result.convert_R
                                                                         },
                                                                         {
                                                                            testValues.result.dequantization_X,
                                                                            testValues.result.dequantization_H,
                                                                            testValues.result.dequantization_W,
                                                                            testValues.result.dequantization_R
                                                                         },
                                                                         {},
                                                                         testValues.result.precisionAfterOperation,
                                                                         testValues.result.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<LSTMTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const std::vector<ngraph::PartialShape> activations_shapes = std::get<1>(obj.param);
        const std::vector<ngraph::Shape> weights_shapes = std::get<2>(obj.param);
        const LSTMTransformationTestValues testValues = std::get<3>(obj.param);

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, activations_shapes[0], testValues.params)
               << "_" << testValues.actual << "_" << testValues.result << "_";
        return result.str();
    }
};

TEST_P(LSTMTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

namespace testValues1 {
const std::vector<std::vector<ngraph::PartialShape>> activations_shapes = {{{1, 16}, {1, 128}, {1, 128}}};

const std::vector<std::vector<ngraph::Shape>> weights_shapes = {{{512, 16}, {512, 128}, {512}}};

const std::vector<LSTMTransformationTestValues> testValues = {
    // LSTM Cell
    {LayerTransformation::createParamsU8I8(),
     LSTMFunction::RNNType::LSTMCell,
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
        {
             {element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
        {
             {element::f32},
             {},
             {0.01f},
        },
        // W
        {255ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
        {},
        {{}, {}, {}},
        // R
        {255ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
        {},
        {{}, {}, {}},
    },
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
         {
             {element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
         {
             {element::f32},
             {},
             {0.01f},
        },
        // W
        {},
        {},
        {
            {element::f32},
            {},
            {0.01f}
        },
        // R
        {},
        {},
        {
            {element::f32},
            {},
            {0.01f}
        },
    }
    },
};
INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    LSTMTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(activations_shapes),
        ::testing::ValuesIn(weights_shapes),
        ::testing::ValuesIn(testValues)),
    LSTMTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<std::vector<ngraph::PartialShape>> activations_shapes = {{{1, 1, 16}, {1, 1, 128}, {1, 1, 128}}};

const std::vector<std::vector<ngraph::Shape>> weights_shapes = {{{1, 512, 16}, {1, 512, 128}, {1, 512}}};

const std::vector<LSTMTransformationTestValues> testValues = {
    // LSTM Sequence
    {LayerTransformation::createParamsU8I8(),
     LSTMFunction::RNNType::LSTMSequence,
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
        {
             {element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
        {
             {element::f32},
             {},
             {0.01f},
        },
        // W
        {255ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
        {},
        {{}, {}, {}},
        // R
        {255ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
        {},
        {{}, {}, {}},
    },
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
         {
             {element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
         {
             {element::f32},
             {},
             {0.01f},
        },
        // W
        {},
        {},
        {
            {element::f32},
            {},
            {0.01f}
        },
        // R
        {},
        {},
        {
            {element::f32},
            {},
            {0.01f}
        },
    }
    },
};
INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    LSTMTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(activations_shapes),
        ::testing::ValuesIn(weights_shapes),
        ::testing::ValuesIn(testValues)),
    LSTMTransformation::getTestCaseName);
} // namespace testValues2

namespace testValues3 {
const std::vector<std::vector<ngraph::PartialShape>> activations_shapes = {{{2, 3}, {2, 3}, {}}};

const std::vector<std::vector<ngraph::Shape>> weights_shapes = {{{9, 3}, {9, 3}, {9}}};

const std::vector<LSTMTransformationTestValues> testValues = {
    // GRU
    {LayerTransformation::createParamsU8I8(),
    LSTMFunction::RNNType::GRU,
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
        {
             {element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
        {
             {element::f32},
             {},
             {0.01f},
        },
        // W
        {255ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
        {},
        {{}, {}, {}},
        // R
        {255ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
        {},
        {{}, {}, {}},
    },
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
         {
             {element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
         {
             {element::f32},
             {},
             {0.01f},
        },
        // W
        {},
        {},
        {
            {element::f32},
            {},
            {0.01f}
        },
        // R
        {},
        {},
        {
            {element::f32},
            {},
            {0.01f}
        },
    }
    }
};
INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    LSTMTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(activations_shapes),
        ::testing::ValuesIn(weights_shapes),
        ::testing::ValuesIn(testValues)),
    LSTMTransformation::getTestCaseName);
} // namespace testValues3

namespace testValues4 {
const std::vector<std::vector<ngraph::PartialShape>> activations_shapes = {{{2, 3}, {2, 3}, {}}};

const std::vector<std::vector<ngraph::Shape>> weights_shapes = {{{3, 3}, {3, 3}, {9}}};

const std::vector<LSTMTransformationTestValues> testValues = {
    // RNNCell
    {LayerTransformation::createParamsU8I8(),
    LSTMFunction::RNNType::RNNCell,
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
        {
             {element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
        {
             {element::f32},
             {},
             {0.01f},
        },
        // W
        {255ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
        {},
        {{}, {}, {}},
        // R
        {255ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}},
        {},
        {{}, {}, {}},
    },
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
         {
             {element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ngraph::element::u8},
         {
             {element::f32},
             {},
             {0.01f},
        },
        // W
        {},
        {},
        {
            {element::f32},
            {},
            {0.01f}
        },
        // R
        {},
        {},
        {
            {element::f32},
            {},
            {0.01f}
        },
    }
    }
};
INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    LSTMTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(activations_shapes),
        ::testing::ValuesIn(weights_shapes),
        ::testing::ValuesIn(testValues)),
    LSTMTransformation::getTestCaseName);
} // namespace testValues4
} // namespace
