// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <low_precision/common/precisions_restriction.hpp>
#include <low_precision/low_precision.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/convolution_function.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class PrecisionsRestrictionTestValues {
public:
    low_precision::PrecisionsRestriction::PrecisionsByPort inputPrecisionsByPort;
    low_precision::PrecisionsRestriction::PrecisionsByPort outputPrecisionsByPort;
};

typedef std::tuple<
    PrecisionsRestrictionTestValues> PrecisionsRestrictionTestParams;

class PrecisionsRestrictionTest : public LayerTransformation, public testing::WithParamInterface<PrecisionsRestrictionTestParams> {
public:
    void SetUp() override {
        auto testValues = std::get<0>(GetParam());

        actualFunction = ngraph::builder::subgraph::ConvolutionFunction::get(
            Shape({ 1, 3, 16, 16 }),
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            std::vector<float>({ 1.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } });

        ngraph::pass::Manager manager;
        auto supportedPrecisions = std::vector<low_precision::PrecisionsRestriction>({
            low_precision::PrecisionsRestriction::create<ngraph::opset1::Convolution>(
                testValues.inputPrecisionsByPort,
                testValues.outputPrecisionsByPort) });
        manager.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(supportedPrecisions);
        manager.run_passes(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ConvolutionFunction::get(
            Shape({ 1, 3, 16, 16 }),
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            std::vector<float>({ 1.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            testValues.inputPrecisionsByPort,
            testValues.outputPrecisionsByPort);
    }

    static std::string getTestCaseName(testing::TestParamInfo<PrecisionsRestrictionTestParams> obj) {
        PrecisionsRestrictionTestValues testValues = std::get<0>(obj.param);

        std::ostringstream result;
        result << testValues.inputPrecisionsByPort.size() << "_" << testValues.outputPrecisionsByPort.size();
        return result.str();
    }
};

TEST_P(PrecisionsRestrictionTest, CompareFunctions) {
    auto res = compare_functions(actualFunction, referenceFunction, true, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

namespace testValues1 {
const std::vector<PrecisionsRestrictionTestValues> testValues = {
    {
        {},
        {}
    },
    {
        {
            {0, {ngraph::element::u8}},
            {1, {ngraph::element::i8}}
        },
        {}
    },
    {
        {},
        {
            {0, {ngraph::element::f16}}
        }
    },
    {
        {
            {0, {ngraph::element::u8}},
            {1, {ngraph::element::i8}}
        },
        {
            {0, {ngraph::element::f16}}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PrecisionsRestrictionTest,
    ::testing::Combine(::testing::ValuesIn(testValues)),
    PrecisionsRestrictionTest::getTestCaseName);
} // namespace testValues1
