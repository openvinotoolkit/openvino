// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <sstream>
#include <memory>
#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>

#include <low_precision/align_quantization_intervals.hpp>
#include <low_precision/align_quantization_parameters.hpp>
#include <low_precision/markup_avg_pool_precision_preserved.hpp>
#include <low_precision/markup_can_be_quantized.hpp>
#include <low_precision/markup_precisions.hpp>
#include <low_precision/markup_quantization_granularity.hpp>
#include <low_precision/propagate_precisions.hpp>

#include <low_precision/concat.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/max_pool.hpp>
#include <low_precision/rt_info/quantization_alignment_attribute.hpp>
#include <low_precision/rt_info/quantization_granularity_attribute.hpp>

#include "lpt_ngraph_functions/precision_propagation_function.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

namespace {

class QuantizationGranularityActualValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize3;
};

inline std::ostream& operator<<(std::ostream& out, const QuantizationGranularityActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2 << "_" << values.fakeQuantize3;
}

class QuantizationGranularityTestValues {
public:
    TestTransformationParams params;
    QuantizationGranularityActualValues actual;
};

inline std::ostream& operator<<(std::ostream& out, const QuantizationGranularityTestValues& values) {
    return out << "_" << values.actual;
}

typedef std::tuple <
    ngraph::element::Type,
    ngraph::Shape,
    QuantizationGranularityTestValues,
    ngraph::QuantizationGranularityAttribute::Granularity
> QuantizationGranularityParams;

class QuantizationGranularityTest :
    public LayerTransformation,
    public testing::WithParamInterface<QuantizationGranularityParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        QuantizationGranularityTestValues testValues = std::get<2>(GetParam());
        QuantizationGranularityAttribute::Granularity granularity = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::PrecisionPropagationFunction::getOriginalWithNeighbors(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            {},
            {},
            testValues.actual.fakeQuantize2,
            {},
            {},
            testValues.actual.fakeQuantize3,
            {},
            {});

        auto supportedPrecisionsOnActivation = std::vector<ngraph::pass::low_precision::PrecisionsRestriction>({
            ngraph::pass::low_precision::PrecisionsRestriction::create<ngraph::opset1::Convolution>({
                {0, {ngraph::element::u8}},
                {1, {ngraph::element::i8}}
            })
        });

        auto quantizationRestrictions = std::vector<ngraph::pass::low_precision::QuantizationGranularityRestriction>({
            ngraph::pass::low_precision::QuantizationGranularityRestriction::create<ngraph::opset1::Convolution>({ { 0, granularity } }, false) });

        ngraph::pass::Manager markup;
        // markup by PrecisionPreservedAttribute attribute instances
        markup.register_pass<ngraph::pass::low_precision::MarkupPrecisions>();
        markup.register_pass<ngraph::pass::low_precision::MarkupQuantizationGranularity>(quantizationRestrictions);
        markup.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>();
        markup.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
        markup.register_pass<ngraph::pass::low_precision::AlignQuantizationIntervals>();
        markup.register_pass<ngraph::pass::low_precision::AlignQuantizationParameters>();
        markup.run_passes(actualFunction);

        ngraph::pass::Manager main;
        std::shared_ptr<ngraph::pass::GraphRewrite> common = main.register_pass<ngraph::pass::GraphRewrite>();
        common->add_matcher<ngraph::pass::low_precision::ConcatTransformation>();
        common->add_matcher<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>();
        common->add_matcher<ngraph::pass::low_precision::MaxPoolTransformation>();
        main.run_passes(actualFunction);
    }

    static std::string getTestCaseName(testing::TestParamInfo<QuantizationGranularityParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const QuantizationGranularityTestValues testValues = std::get<2>(obj.param);
        const ngraph::QuantizationGranularityAttribute::Granularity granularity = std::get<3>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_" <<
            (granularity == ngraph::QuantizationGranularityAttribute::Granularity::PerChannel ? "PerChannel" : "PerTensor") << "_" <<
            testValues.actual << "_";
        return result.str();
    }
};

TEST_P(QuantizationGranularityTest, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();

    auto concats = LayerTransformation::get<opset1::Concat>(actualFunction);
    ASSERT_EQ(2ul, concats.size()) << "unexpected operations count";

    auto getValue = [](const NodeVector& concats, const size_t index) {
        auto fakeQuantize = concats[index];
        auto attribute = ngraph::pass::low_precision::getAttribute<ngraph::QuantizationAlignmentAttribute>(fakeQuantize).
            as<ngraph::QuantizationAlignmentAttribute>();
        auto value = attribute.value();
        return value;
    };

    ASSERT_EQ("concat2", concats[0]->get_friendly_name());
    auto value2 = getValue(concats, 0);

    QuantizationGranularityAttribute::Granularity granularity = std::get<3>(GetParam());
    if (granularity == QuantizationGranularityAttribute::Granularity::PerTensor) {
        ASSERT_TRUE(value2);
    } else {
        ASSERT_FALSE(value2);
    }
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32
};

const std::vector<QuantizationGranularityTestValues> testValues = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            { 256ul, ngraph::Shape({}), {-1.28f / 3.f}, {1.27f / 3.f}, {-1.28f / 3.f}, {1.27f / 3.f} },
            { 256ul, ngraph::Shape({}), {-1.28f / 2.f}, {1.27f / 2.f}, {-1.28f / 2.f}, {1.27f / 2.f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
        }
    }
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 9, 9 }
};

const std::vector<ngraph::QuantizationGranularityAttribute::Granularity> granularities = {
    ngraph::QuantizationGranularityAttribute::Granularity::PerChannel,
    ngraph::QuantizationGranularityAttribute::Granularity::PerTensor
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    QuantizationGranularityTest,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(granularities)),
    QuantizationGranularityTest::getTestCaseName);
}  // namespace
