// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/transformer.hpp>
#include <low_precision/relu.hpp>
#include <low_precision/move_fake_quatize.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/move_fake_quantize_function.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/relu_function.hpp"
#include "simple_low_precision_transformer.hpp"
//nd/lpt/move_fake_quantize
using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

namespace {

class MoveFakeQuantizeActualValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert1;
    ngraph::builder::subgraph::DequantizationOperations dequantization1;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert2;
    ngraph::builder::subgraph::DequantizationOperations dequantization2;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize3;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert3;
    ngraph::builder::subgraph::DequantizationOperations dequantization3;
};

inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeActualValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.convert1.outPrecision << "_" <<
        values.dequantization1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.convert2.outPrecision << "_" <<
        values.dequantization2;
}

class MoveFakeQuantizeResultValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert1;
    ngraph::builder::subgraph::DequantizationOperations dequantization1;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert2;
    ngraph::builder::subgraph::DequantizationOperations dequantization2;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize3;
    ngraph::builder::subgraph::DequantizationOperations::Convert convert3;
    ngraph::builder::subgraph::DequantizationOperations dequantization3;
    ngraph::element::Type precisionAfterOperation;
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
};

inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeResultValues& values) {
    return out << "_" <<
        values.fakeQuantize1 << "_" <<
        values.convert1.outPrecision << "_" <<
        values.dequantization1 << "_" <<
        values.fakeQuantize2 << "_" <<
        values.convert2.outPrecision << "_" <<
        values.dequantization2 << "_" <<
        values.dequantizationAfter;
}

class MoveFakeQuantizeTestValues {
public:
    ngraph::pass::low_precision::LayerTransformation::Params params;
    bool multiChannels;
    std::int64_t axis;
    MoveFakeQuantizeActualValues actual;
    MoveFakeQuantizeResultValues result;
};

inline std::ostream& operator<<(std::ostream& out, const MoveFakeQuantizeTestValues& values) {
    return out << "_" << values.multiChannels << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ngraph::element::Type,
    ngraph::PartialShape,
    MoveFakeQuantizeTestValues
> MoveFakeQuantizeParams;

class MoveFakeQuantize : public LayerTransformation, public testing::WithParamInterface<MoveFakeQuantizeParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::PartialShape shape = std::get<1>(GetParam());
        MoveFakeQuantizeTestValues testValues = std::get<2>(GetParam());

        // dequantization output precision depends on input precision
        // to avoid huge amount of tests cases let's define dequantization output precision as input precision
        if (!testValues.actual.dequantization1.multiply.empty()) {
            testValues.actual.dequantization1.multiply.outPrecision = precision;
        }
        if (!testValues.actual.dequantization2.multiply.empty()) {
            testValues.actual.dequantization2.multiply.outPrecision = precision;
        }
        actualFunction = ngraph::builder::subgraph::MoveFakeQuantize::get(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.convert1,
            testValues.actual.dequantization1,
            testValues.actual.fakeQuantize2,
            testValues.actual.convert2,
            testValues.actual.dequantization2,
            testValues.actual.fakeQuantize3,
            testValues.actual.convert3,
            testValues.actual.dequantization3,
            ngraph::element::undefined,
            {},
            testValues.axis);
        /*
        actualFunction = ngraph::builder::subgraph::MoveFakeQuantize::get(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2,
            testValues.actual.fakeQuantize3,
            testValues.axis);
            */
        ngraph::pass::VisualizeTree("c:\\Users\\ndemasho\\rep\\Visual\\MFQtest.actual").run_on_function(actualFunction);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::MoveFakeQuantize, ngraph::opset1::FakeQuantize>(testValues.params);      
        transform.transform(actualFunction);
        ngraph::pass::VisualizeTree("c:\\Users\\ndemasho\\rep\\Visual\\MFQtest.transform").run_on_function(actualFunction);

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
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.convert1,
            testValues.actual.dequantization1,
            testValues.actual.fakeQuantize2,
            testValues.actual.convert2,
            testValues.actual.dequantization2,
            testValues.actual.fakeQuantize3,
            testValues.actual.convert3,
            testValues.actual.dequantization3,
            testValues.result.precisionAfterOperation,
            testValues.result.dequantizationAfter,
            testValues.axis);
        /*
        referenceFunction = ngraph::builder::subgraph::MoveFakeQuantize::get(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2,
            testValues.actual.fakeQuantize3,
            testValues.axis);
            */
        ngraph::pass::VisualizeTree("c:\\Users\\ndemasho\\rep\\Visual\\MFQtest.reference").run_on_function(referenceFunction);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MoveFakeQuantizeParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::PartialShape shape = std::get<1>(obj.param);
        const MoveFakeQuantizeTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_" <<
            (testValues.multiChannels ? "multiChannels_" : "notMultiChannels_") <<
            "axis_" << testValues.axis << "_" <<
            testValues.actual << "_" <<
            testValues.result << "_";
        return result.str();
    }
};

TEST_P(MoveFakeQuantize, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, referenceFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

namespace testValues1 {
const std::vector<ngraph::PartialShape> shapes = {
    { 1, 3, 9, 9 },
    //{ 4, 3, 9, 9 },
    //{ Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic() }
};

const std::vector<MoveFakeQuantizeTestValues> testValues = {
    // U8: concat
    {
        LayerTransformation::createParamsU8I8(),
        false,
        1,
        {
            {},
            {},
            {},
            {},
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { ngraph::element::u8 },
            {
                { element::f32 },
                {},
                { 0.01f }
            },
        },
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            {},
            {},
            {},
            {},
            {},
            ngraph::element::undefined,
            {},
        }
    },
};
INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MoveFakeQuantize,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    MoveFakeQuantize::getTestCaseName);
} // namespace testValues1
} // namespace
