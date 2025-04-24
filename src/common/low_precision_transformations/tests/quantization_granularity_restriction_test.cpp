// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include "low_precision/common/port_quantization_granularity_restriction.hpp"
#include "low_precision/common/quantization_granularity_restriction.hpp"
#include "low_precision/markup_quantization_granularity.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/convolution.hpp"
#include "openvino/op/convolution.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

class OperationQuantizationRestrictionTestValues {
public:
    std::vector<ov::pass::low_precision::PortQuantizationGranularityRestriction> restrictions;
};

typedef std::tuple<
    OperationQuantizationRestrictionTestValues,
    bool
> OperationQuantizationRestrictionParams;

class OperationQuantizationRestrictionTest : public LayerTransformation, public testing::WithParamInterface<OperationQuantizationRestrictionParams> {
public:
    void SetUp() override {
        const auto testValues = std::get<0>(GetParam());
        const auto explicitly = std::get<1>(GetParam());

        std::vector<size_t> ports;
        if (!explicitly) {
            for (size_t i = 0; i < testValues.restrictions.size(); ++i) {
                ports.push_back(testValues.restrictions[i].port);
            }
        }

        actualFunction = ov::builder::subgraph::ConvolutionFunction::get(
            Shape({ 1, 3, 16, 16 }),
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            std::vector<float>({ 1.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } });

        ov::pass::Manager manager;
        const auto quantizationRestrictions = std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>({
            explicitly ?
                ov::pass::low_precision::QuantizationGranularityRestriction::create<ov::opset1::Convolution>(testValues.restrictions, false) :
                ov::pass::low_precision::QuantizationGranularityRestriction::create<ov::opset1::Convolution>(ports)
        });
        manager.register_pass<ov::pass::low_precision::MarkupQuantizationGranularity>(quantizationRestrictions);
        manager.run_passes(actualFunction);

        referenceFunction = ov::builder::subgraph::ConvolutionFunction::get(
            Shape({ 1, 3, 16, 16 }),
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            std::vector<float>({ 1.f }),
            { 255ul, Shape({ 1, 1, 1, 1 }), { 0.f }, { 254.f }, { -1.27f }, { 1.27f } },
            quantizationRestrictions);
    }

    static std::string getTestCaseName(testing::TestParamInfo<OperationQuantizationRestrictionParams> obj) {
        const auto testValues = std::get<0>(obj.param);
        const auto explicitly = std::get<1>(obj.param);

        std::ostringstream result;
        result << testValues.restrictions.size() << "_" << explicitly;
        return result.str();
    }
};

TEST_P(OperationQuantizationRestrictionTest, CompareFunctions) {
    auto res = compare_functions(actualFunction, referenceFunction, true, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<OperationQuantizationRestrictionTestValues> testValues = {
    {
        {}
    },
    {
        {{0, ov::QuantizationGranularityAttribute::Granularity::PerTensor}}
    },
    {
        {{0, ov::QuantizationGranularityAttribute::Granularity::PerTensor}, {1, ov::QuantizationGranularityAttribute::Granularity::PerChannel}}
    }
};

const std::vector<bool> explicitly = { true, false };

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    OperationQuantizationRestrictionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(explicitly)),
    OperationQuantizationRestrictionTest::getTestCaseName);
