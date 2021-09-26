// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <low_precision/fake_quantize_decomposition.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/get_dequantization_function.hpp"
#include <low_precision/common/fake_quantize_dequantization.hpp>
#include "low_precision/network_helper.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class GetDequantizationBelowTestValues {
public:
    builder::subgraph::FakeQuantizeOnData fakeQuantize;
    builder::subgraph::DequantizationOperations dequantization;
};

inline std::ostream& operator<<(std::ostream& os, const std::vector<float>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

inline std::ostream& operator<<(std::ostream& out, const GetDequantizationBelowTestValues& testValue) {
    return out << "_" << testValue.fakeQuantize << "_" << testValue.dequantization;
}

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    GetDequantizationBelowTestValues> GetDequantizationBelowParams;

class GetDequantizationBelowTransformation : public LayerTransformation, public testing::WithParamInterface<GetDequantizationBelowParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const GetDequantizationBelowTestValues testValues = std::get<2>(GetParam());

        auto const function = ngraph::builder::subgraph::GetDequantizationFunction::get(
            precision,
            shape,
            testValues.fakeQuantize,
            testValues.dequantization);

        auto const fakeQuantize = function->get_parameters()[0]->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
        auto dequantization = ngraph::pass::low_precision::NetworkHelper::getDequantizationBelow(fakeQuantize);

        actualFunction = ngraph::builder::subgraph::GetDequantizationFunction::get(
            precision,
            shape,
            testValues.fakeQuantize,
            dequantization);

        referenceFunction = ngraph::builder::subgraph::GetDequantizationFunction::get(
            precision,
            shape,
            testValues.fakeQuantize,
            testValues.dequantization);
    }

    static std::string getTestCaseName(testing::TestParamInfo<GetDequantizationBelowParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        GetDequantizationBelowTestValues testValues;
        std::tie(precision, shape, testValues) = obj.param;

        std::ostringstream result;
        result << precision << "_" << shape << "_" << testValues;
        return result.str();
    }
};

TEST_P(GetDequantizationBelowTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
};

const std::vector<GetDequantizationBelowTestValues> testValues = {
    {
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f }, ngraph::element::u8 },
        { ngraph::element::f32, {}, { 0.01f } }
    },
    {
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f }, ngraph::element::u8 },
        { ngraph::element::f32, { 127.f }, { 0.01f } }
    },
    {
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f }, ngraph::element::u8 },
        {
            ngraph::element::f32,
            {{ 127.f }, ngraph::element::f32, {}, false, 1, ngraph::element::u8, true},
            { 0.01f }
        }
    }
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 },
    // TODO: 3D tensor
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    GetDequantizationBelowTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    GetDequantizationBelowTransformation::getTestCaseName);
