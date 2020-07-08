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
#include <transformations/low_precision/fake_quantize.hpp>
#include <transformations/low_precision/multiply.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/multiply_function.hpp"

using namespace testing;
using namespace ngraph::pass;

class MultiplyTransformationTestValues {
public:
    low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::vector<float> expectedSubtractValues;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    MultiplyTransformationTestValues> MultiplyTransformationParams;

class MultiplyTransformation : public LayerTransformation, public testing::WithParamInterface<MultiplyTransformationParams> {
public:
    void SetUp() override {
        //const ngraph::element::Type precision = std::get<0>(GetParam());
        //const ngraph::Shape shape = std::get<1>(GetParam());
        //const FakeQuantizeOnDataTestValues fakeQuantizeOnData = std::get<2>(GetParam());

        //actualFunction = ngraph::builder::subgraph::MultiplyFunction::getOriginal(
        //    precision,
        //    shape,
        //    fakeQuantizeOnData.params,
        //    fakeQuantizeOnData.actual);

        //ngraph::pass::low_precision::LowPrecisionTransformations transformations(
        //    {},
        //    { { "FakeQuantize", ngraph::pass::low_precision::LayerTransformationPtr(
        //        new ngraph::pass::low_precision::FakeQuantizeTransformation(fakeQuantizeOnData.params)) } },
        //    { { "Multiply", ngraph::pass::low_precision::LayerTransformationPtr(
        //        new ngraph::pass::low_precision::MultiplyTransformation(fakeQuantizeOnData.params)) } }
        //);
        //ngraph::pass::low_precision::LowPrecisionTransformer transformer(transformations);
        //transformer.transform(actualFunction);

        //referenceFunction = ngraph::builder::subgraph::MultiplyFunction::getReference(precision, shape);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MultiplyTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const MultiplyTransformationTestValues fakeQuantizeOnData = std::get<2>(obj.param);

        return LayerTransformation::getTestCaseNameByParams(precision, shape, fakeQuantizeOnData.params);
    }
};

TEST_P(MultiplyTransformation, CompareFunctions) {
    // InitNodeInfo().run_on_function(actualFunction);
    // ConvFusion().run_on_function(actualFunction);

    // actualFunction->validate_nodes_and_infer_types();

    // auto res = compare_functions(referenceFunction, actualFunction);
    // ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

const std::vector<MultiplyTransformationTestValues> fakeQuantizeOnDataTestValues = {
    // U8
    {
        LayerTransformation::createParamsU8I8(),
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        {}
    },
    // {
    //    LayerTransformation::createParamsU8I8(),
    //    { 256ul, {}, { -1.28f} , { 1.27f }, { -1.28f} , { 1.27f } },
    //    { 1.28f }
    // },

    //// I8
    // {
    //    LayerTransformation::createParamsI8I8(),
    //    { 256ul, {}, { -1.28f}, { 1.27f }, { -1.28f}, { 1.27f } },
    //    {}
    // },
    // {
    //    LayerTransformation::createParamsI8I8(),
    //    { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
    //    { 1.28f }
    // }
};

INSTANTIATE_TEST_CASE_P(
    DISABLED_LPT,
    MultiplyTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(fakeQuantizeOnDataTestValues)),
    MultiplyTransformation::getTestCaseName);
