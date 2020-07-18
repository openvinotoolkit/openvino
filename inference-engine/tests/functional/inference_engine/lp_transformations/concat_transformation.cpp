// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>
#include <transformations/low_precision/transformer.hpp>
#include <transformations/low_precision/concat.hpp>
#include <transformations/low_precision/fake_quantize.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/concat_function.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "simple_low_precision_transformer.hpp"

// TODO: debug only
// #include <ngraph/pass/visualize_tree.hpp>

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

namespace {

class ConcatTransformationActualValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2;
}

class ConcatTransformationResultValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOperations;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationResultValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2 << "_" << values.dequantizationOperations;
}

class ConcatTransformationTestValues {
public:
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ConcatTransformationActualValues actual;
    ConcatTransformationResultValues result;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationTestValues& values) {
    return out << "_" << values.actual << "_" << values.result;
}

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    ConcatTransformationTestValues
> ConcatTransformationParams;

class ConcatTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const ConcatTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::ConcatFunction::getOriginal(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2);

        // VisualizeTree("C:\\Projects\\temp\\test.actual").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ actualFunction });

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ConcatTransformation, ngraph::opset1::Concat>(testValues.params);
        transform.transform(actualFunction);

        // VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ actualFunction });

        referenceFunction = ngraph::builder::subgraph::ConcatFunction::getReference(
            precision,
            shape,
            testValues.result.fakeQuantize1,
            testValues.result.fakeQuantize2,
            testValues.result.dequantizationOperations);

        // VisualizeTree("C:\\Projects\\temp\\test.reference").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ referenceFunction });
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const ConcatTransformationTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_" <<
            testValues.actual << "_" <<
            testValues.result << "_";
        return result.str();
    }
};

TEST_P(ConcatTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

const std::vector<ConcatTransformationTestValues> testValues = {
    // U8
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} }
        },
        {
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {255.f}, ngraph::element::u8 },
            { ngraph::element::f32, {}, { 0.01f } }
        }
    },
    // I8
    {
        LayerTransformation::createParamsI8I8(),
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
        },
        {
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f}, ngraph::element::i8 },
            { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f}, ngraph::element::i8 },
            { ngraph::element::f32, {}, { 0.01f } }
        }
    },
    // mixed
    // {
    //    LayerTransformation::createParamsU8I8(),
    //    {
    //        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
    //        { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f} }
    //    },
    //    {
    //        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f}, ngraph::element::u8 },
    //        { 256ul, ngraph::Shape({}), {-1.28f}, {1.27f}, {-1.28f}, {1.27f}, ngraph::element::u8 },
    //        { ngraph::element::f32, {}, { 0.01f } }
    //    },
    // }
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    ConcatTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)
    ),
    ConcatTransformation::getTestCaseName);
}
