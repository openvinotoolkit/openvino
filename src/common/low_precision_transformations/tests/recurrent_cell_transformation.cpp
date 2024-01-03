// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/common/precisions_restriction.hpp"
#include "low_precision/recurrent_cell.hpp"
#include "low_precision/fold_convert.hpp"
#include "low_precision/fuse_convert.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"
#include <memory>
#include <sstream>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/recurrent_cell.hpp"
#include "simple_low_precision_transformer.hpp"
#include "openvino/opsets/opset5.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ngraph::builder::subgraph;

namespace {

class RecurrentCellTransformationValues {
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
    ov::element::Type precisionAfterOperation;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
};

inline std::ostream& operator<<(std::ostream& out, const RecurrentCellTransformationValues& values) {
    return out << "_" << values.fakeQuantize_X << "_" << values.convert_X << "_" << values.dequantization_X <<
                  "_" << values.fakeQuantize_H << "_" << values.convert_H << "_" << values.dequantization_H <<
                  "_" << values.fakeQuantize_W << "_" << values.convert_W << "_" << values.dequantization_W <<
                  "_" << values.fakeQuantize_R << "_" << values.convert_R << "_" << values.dequantization_R;
}

class RecurrentCellTransformationTestValues {
public:
    RecurrentCellTransformationTestValues() = default;
    RecurrentCellTransformationTestValues(const TestTransformationParams& params,
                                 const RecurrentCellFunction::RNNType type,
                                 const RecurrentCellTransformationValues& actual,
                                 const RecurrentCellTransformationValues& result,
                                 const bool addNotPrecisionPreservedOperation = false,
                                 const bool checkIntervalsAlignmentAttributes = true)
        : params(params),
          type(type),
          actual(actual),
          result(result) {}

    TestTransformationParams params;
    RecurrentCellFunction::RNNType type;
    RecurrentCellTransformationValues actual;
    RecurrentCellTransformationValues result;
};

inline std::ostream& operator<<(std::ostream& out, const RecurrentCellTransformationTestValues& values) {
    return out << "_" << values.actual << "_" << values.result;
}

typedef std::tuple<ov::element::Type, std::vector<ov::PartialShape>, std::vector<ov::Shape>, RecurrentCellTransformationTestValues>
    RecurrentCellTransformationParams;

class RecurrentCellTransformation : public LayerTransformation, public testing::WithParamInterface<RecurrentCellTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const std::vector<ov::PartialShape> activations_shapes = std::get<1>(GetParam());
        const std::vector<ov::Shape> weights_shapes = std::get<2>(GetParam());
        RecurrentCellTransformationTestValues testValues = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::RecurrentCellFunction::get(precision,
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
                                                                      });

        const auto params = TestTransformationParams::toParams(testValues.params);

        SimpleLowPrecisionTransformer transformer;
        transformer.commonGraphRewrite->add_matcher<ov::pass::low_precision::RecurrentCellTransformation>(params);
        transformer.transform(actualFunction);

        SimpleLowPrecisionTransformer clenup_transformer;
        clenup_transformer.commonGraphRewrite->add_matcher<ov::pass::low_precision::FoldConvertTransformation>(params);
        clenup_transformer.commonGraphRewrite->add_matcher<ov::pass::low_precision::FuseConvertTransformation>(params);
        clenup_transformer.commonGraphRewrite->add_matcher<ov::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>(params);
        clenup_transformer.commonGraphRewrite->add_matcher<ov::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>(params);
        clenup_transformer.transform(actualFunction);

        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.result.dequantizationAfter.multiply.empty()) {
            testValues.result.dequantizationAfter.multiply.outPrecision = precision;
        }

        referenceFunction =
            ngraph::builder::subgraph::RecurrentCellFunction::get(precision,
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
                                                                         });
    }

    static std::string getTestCaseName(testing::TestParamInfo<RecurrentCellTransformationParams> obj) {
        const ov::element::Type precision = std::get<0>(obj.param);
        const std::vector<ov::PartialShape> activations_shapes = std::get<1>(obj.param);
        const std::vector<ov::Shape> weights_shapes = std::get<2>(obj.param);
        const RecurrentCellTransformationTestValues testValues = std::get<3>(obj.param);

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, activations_shapes[0], testValues.params)
               << "_" << testValues.actual << "_" << testValues.result << "_";
        return result.str();
    }
};

TEST_P(RecurrentCellTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    // ov::element::f16
};

namespace testValues2 {
const std::vector<std::vector<ov::PartialShape>> activations_shapes = {{{1, 1, 16}, {1, 1, 128}, {1, 1, 128}}};

const std::vector<std::vector<ov::Shape>> weights_shapes = {{{1, 512, 16}, {1, 512, 128}, {1, 512}}};

const std::vector<RecurrentCellTransformationTestValues> testValues = {
    // LSTM Sequence
    {LayerTransformation::createParamsU8I8(),
     RecurrentCellFunction::RNNType::LSTMSequence,
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
        {
             {element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
        {
             {element::f32},
             {},
             {0.01f},
        },
        // W
        {255ul, {}, {-1.27f}, {1.27f}, {-1.27f}, {1.27f}},
        {},
        {{}, {}, {}},
        // R
        {255ul, {}, {-1.27f}, {1.27f}, {-1.27f}, {1.27f}},
        {},
        {{}, {}, {}},
    },
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
         {
             {element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
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
    RecurrentCellTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(activations_shapes),
        ::testing::ValuesIn(weights_shapes),
        ::testing::ValuesIn(testValues)),
    RecurrentCellTransformation::getTestCaseName);
} // namespace testValues2

namespace testValues3 {
const std::vector<std::vector<ov::PartialShape>> activations_shapes = {{{1, 2, 3}, {1, 1, 3}, {}}};

const std::vector<std::vector<ov::Shape>> weights_shapes = {{{1, 9, 3}, {1, 9, 3}, {1, 9}}};

const std::vector<RecurrentCellTransformationTestValues> testValues = {
    // GRU
    {LayerTransformation::createParamsU8I8(),
    RecurrentCellFunction::RNNType::GRUSequence,
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
        {
             {element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
        {
             {element::f32},
             {},
             {0.01f},
        },
        // W
        {255ul, {}, {-1.27f}, {1.27f}, {-1.27f}, {1.27f}},
        {},
        {{}, {}, {}},
        // R
        {255ul, {}, {-1.27f}, {1.27f}, {-1.27f}, {1.27f}},
        {},
        {{}, {}, {}},
    },
    {
        // X
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
         {
             {element::f32},
             {},
             {0.01f},
        },
        // H
        {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}},
        {ov::element::u8},
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
    RecurrentCellTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(activations_shapes),
        ::testing::ValuesIn(weights_shapes),
        ::testing::ValuesIn(testValues)),
    RecurrentCellTransformation::getTestCaseName);
} // namespace testValues3

} // namespace
