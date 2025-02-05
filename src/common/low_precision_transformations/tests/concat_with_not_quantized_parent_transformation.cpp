// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <sstream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"

#include "low_precision/low_precision.hpp"

#include "low_precision/concat.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/align_quantization_parameters.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/markup_can_be_quantized.hpp"
#include "low_precision/markup_quantization_granularity.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/concat.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

namespace {

class ConcatWithNotQuantizedParentTransformationActualValues {
public:
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ov::builder::subgraph::DequantizationOperations::Convert convert1;
    ov::builder::subgraph::DequantizationOperations dequantization1;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ov::builder::subgraph::DequantizationOperations::Convert convert2;
    ov::builder::subgraph::DequantizationOperations dequantization2;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatWithNotQuantizedParentTransformationActualValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.convert1.outPrecision << "_" <<
        values.dequantization1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.convert2.outPrecision << "_" <<
        values.dequantization2;
}

class ConcatWithNotQuantizedParentTransformationResultValues {
public:
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ov::builder::subgraph::DequantizationOperations::Convert convert1;
    ov::builder::subgraph::DequantizationOperations dequantization1;
    ov::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ov::builder::subgraph::DequantizationOperations::Convert convert2;
    ov::builder::subgraph::DequantizationOperations dequantization2;
    ov::element::Type precisionAfterOperation;
    ov::builder::subgraph::DequantizationOperations dequantizationAfter;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatWithNotQuantizedParentTransformationResultValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.convert1.outPrecision << "_" <<
        values.dequantization1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.convert2.outPrecision << "_" <<
        values.dequantization2 << "_" <<
        values.dequantizationAfter;
}

class ConcatWithNotQuantizedParentTransformationTestValues {
public:
    ConcatWithNotQuantizedParentTransformationTestValues() = default;
    ConcatWithNotQuantizedParentTransformationTestValues(
        const TestTransformationParams& params,
        const bool multiChannels,
        const  std::int64_t axis,
        const ConcatWithNotQuantizedParentTransformationActualValues& actual,
        const ConcatWithNotQuantizedParentTransformationResultValues& result,
        const bool addNotPrecisionPreservedOperation = false,
        const bool checkIntervalsAlignmentAttributes = true) :
        params(params),
        multiChannels(multiChannels),
        axis(axis),
        actual(actual),
        result(result),
        addNotPrecisionPreservedOperation(addNotPrecisionPreservedOperation),
        checkIntervalsAlignmentAttributes(checkIntervalsAlignmentAttributes) {}

    TestTransformationParams params;
    bool multiChannels;
    std::int64_t axis;
    ConcatWithNotQuantizedParentTransformationActualValues actual;
    ConcatWithNotQuantizedParentTransformationResultValues result;
    // add not precision preserved operation to set output precision for FakeQuantize
    // don't set to 'true' by default to keep test cases with tested operation as output
    bool addNotPrecisionPreservedOperation;
    bool checkIntervalsAlignmentAttributes;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatWithNotQuantizedParentTransformationTestValues& values) {
    return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ov::element::Type,
    std::pair<ov::Shape, ov::Shape>,
    ConcatWithNotQuantizedParentTransformationTestValues
> ConcatWithNotQuantizedParentTransformationParams;

class ConcatWithNotQuantizedParentTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<ConcatWithNotQuantizedParentTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const std::pair<ov::Shape, ov::Shape> shapes = std::get<1>(GetParam());
        ConcatWithNotQuantizedParentTransformationTestValues testValues = std::get<2>(GetParam());

        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.actual.dequantization1.multiply.empty()) {
            testValues.actual.dequantization1.multiply.outPrecision = precision;
        }
        if (!testValues.actual.dequantization2.multiply.empty()) {
            testValues.actual.dequantization2.multiply.outPrecision = precision;
        }

        actualFunction = ov::builder::subgraph::ConcatFunction::get(
            precision,
            shapes.first,
            testValues.actual.fakeQuantize1,
            testValues.actual.convert1,
            testValues.actual.dequantization1,
            false,
            shapes.second,
            testValues.actual.fakeQuantize2,
            testValues.actual.convert2,
            testValues.actual.dequantization2,
            true,
            {},
            ov::element::dynamic,
            {},
            testValues.axis,
            testValues.addNotPrecisionPreservedOperation);

        auto precisionsRestrictions = std::vector<ov::pass::low_precision::PrecisionsRestriction>({
            ov::pass::low_precision::PrecisionsRestriction::create<ov::op::v1::Convolution>({
                {{0}, {ov::element::u8}},
                {{1}, {ov::element::i8}}
            }),
            ov::pass::low_precision::PrecisionsRestriction::create<ov::op::v1::AvgPool>({{{0}, testValues.params.precisionsOnActivations}})
        });

        auto quantizationRestrictions = std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>({
            ov::pass::low_precision::QuantizationGranularityRestriction::create<ov::op::v1::Convolution>({0})
        });

        const auto params = TestTransformationParams(testValues.params.updatePrecisions);
        const auto legacyParams = TestTransformationParams::toParams(params);

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::low_precision::MarkupPrecisions>(precisionsRestrictions);
        manager.register_pass<ov::pass::low_precision::MarkupQuantizationGranularity>(quantizationRestrictions);
        manager.register_pass<ov::pass::low_precision::MarkupAvgPoolPrecisionPreserved>(params.defaultPrecisions);
        manager.register_pass<ov::pass::low_precision::PropagatePrecisions>();
        manager.register_pass<ov::pass::low_precision::AlignQuantizationIntervals>(params.defaultPrecisions);
        manager.register_pass<ov::pass::low_precision::AlignQuantizationParameters>(params.defaultPrecisions);

        std::shared_ptr<ov::pass::GraphRewrite> common = manager.register_pass<ov::pass::GraphRewrite>();
        common->add_matcher<ov::pass::low_precision::ConcatTransformation>(legacyParams);
        common->add_matcher<ov::pass::low_precision::FakeQuantizeDecompositionTransformation>(legacyParams);
        manager.run_passes(actualFunction);

        {
            ov::pass::Manager standaloneCleanupManager;
            standaloneCleanupManager.register_pass<ov::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>();
            standaloneCleanupManager.run_passes(actualFunction);
        }

        {
            ov::pass::Manager standaloneCleanupManager;
            standaloneCleanupManager.register_pass<ov::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>();
            standaloneCleanupManager.run_passes(actualFunction);
        }

        if (!testValues.result.dequantizationAfter.multiply.empty()) {
            testValues.result.dequantizationAfter.multiply.outPrecision = precision;
        }

        if (!testValues.params.updatePrecisions &&
            (precision == ov::element::f32) &&
            !testValues.result.dequantizationAfter.convert.empty()) {
            testValues.result.dequantizationAfter.convert = {};
        }

        referenceFunction = ov::builder::subgraph::ConcatFunction::get(
            precision,
            shapes.first,
            testValues.result.fakeQuantize1,
            testValues.result.convert1,
            testValues.result.dequantization1,
            false,
            shapes.second,
            testValues.result.fakeQuantize2,
            testValues.result.convert2,
            testValues.result.dequantization2,
            true,
            {
                ov::PrecisionPreservedAttribute(true),
                ov::IntervalsAlignmentAttribute(ov::IntervalsAlignmentSharedValue::Interval{-1.28f, 2.55f}, 256ul),
                ov::QuantizationAlignmentAttribute(false)
            },
            testValues.result.precisionAfterOperation,
            testValues.result.dequantizationAfter,
            testValues.axis,
            testValues.addNotPrecisionPreservedOperation);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatWithNotQuantizedParentTransformationParams> obj) {
        const ov::element::Type precision = std::get<0>(obj.param);
        const std::pair<ov::Shape, ov::Shape> shapes = std::get<1>(obj.param);
        const ConcatWithNotQuantizedParentTransformationTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shapes.first, testValues.params) << "_" <<
            shapes.second <<
            (testValues.multiChannels ? "multiChannels_" : "notMultiChannels_") <<
            "axis_" << testValues.axis << "_" <<
            testValues.actual << "_" <<
            testValues.result << "_";
        return result.str();
    }
};

TEST_P(ConcatWithNotQuantizedParentTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false, true, false);
    ASSERT_TRUE(res.first) << res.second;

    auto actualFakeQuantizes = LayerTransformation::get<ov::op::v0::FakeQuantize>(actualFunction);
    for (auto it = actualFakeQuantizes.begin(); it != actualFakeQuantizes.end(); it++) {
        const auto actualFakeQuantize = *it;
        if (actualFakeQuantize->output(0).get_target_inputs().begin()->get_index() == 1ul) {
            actualFakeQuantizes.erase(it);
            break;
        }
    }
    ASSERT_TRUE(checkIfOutputAttributesSharedValuesAreTheSame<ov::PrecisionsAttribute>(actualFakeQuantizes)) <<
        "ov::PrecisionsAttribute are not the same";

    ConcatWithNotQuantizedParentTransformationTestValues testValues = std::get<2>(GetParam());
    if (testValues.checkIntervalsAlignmentAttributes) {
        auto operations = LayerTransformation::get<ov::op::v0::Concat>(actualFunction);
        operations.insert(operations.end(), actualFakeQuantizes.begin(), actualFakeQuantizes.end());
        ASSERT_TRUE(checkIfAttributesSharedValuesAreTheSame<ov::IntervalsAlignmentAttribute>(operations)) <<
            "ov::IntervalsAlignmentAttribute are not the same";
    }
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    //ov::element::f16
};

const std::vector<ConcatWithNotQuantizedParentTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            { 256ul, {}, {0.f}, {1.275f}, {0.f}, {1.275f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} }
        },
        {
            { 256ul, {}, {0.f}, {1.275f}, {0.f}, {1.28f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            ov::element::f32,
            {},
        }
    }
};

const std::vector<std::pair<ov::Shape, ov::Shape>> shapes = {
    {{ 1, 3, 9, 9 }, { 1, 3, 9, 9 }},
    {{ 4, 3, 9, 9 }, { 4, 3, 9, 9 }}
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConcatWithNotQuantizedParentTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ConcatWithNotQuantizedParentTransformation::getTestCaseName);
}  // namespace
