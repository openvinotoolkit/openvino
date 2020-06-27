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
#include "../transformations/ngraph_test_utils.hpp"
#include <transformations/low_precision/transformer.hpp>
#include "transformations/low_precision/fake_quantize.hpp"
#include "ngraph_functions/low_precision_transformations/fake_quantize_function.hpp"

// TODO: debug only
#include <ngraph/pass/visualize_tree.hpp>

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

// typedef std::pair<ngraph::Shape, std::vector<builder::subgraph::FakeQuantizeOnData>> FakeQuantizeShapes;

class FakeQuantizeOnDataTestValues {
public:
    low_precision::LayerTransformation::Params params;
    builder::subgraph::FakeQuantizeOnData actual;
    builder::subgraph::FakeQuantizeOnData expected;
    std::vector<float> expectedSubtractValues;
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

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeOnDataTestValues& testValue) {
    return out << "_" <<
        testValue.actual.constantShape << "_" << testValue.actual.outputLowValues << "_" << testValue.actual.outputHighValues << "_" <<
        testValue.expected.constantShape << "_" << testValue.expected.outputLowValues << "_" << testValue.expected.outputHighValues;;
}


typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    ngraph::pass::low_precision::LayerTransformation::Params,
    FakeQuantizeOnDataTestValues> FakeQuantizeTransformationParams;

class FakeQuantizeTransformation : public LayerTransformation, public testing::WithParamInterface<FakeQuantizeTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const FakeQuantizeOnDataTestValues fakeQuantizeOnData = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::FakeQuantizeFunction::getOriginal(
            precision,
            shape,
            fakeQuantizeOnData.params,
            fakeQuantizeOnData.actual);

        VisualizeTree("C:\\Projects\\temp\\test.original").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ actualFunction });

        // transform(actualFunction);

        ngraph::pass::low_precision::LowPrecisionTransformations transformations(
            {},
            { { "FakeQuantize", ngraph::pass::low_precision::LayerTransformationPtr(
                new ngraph::pass::low_precision::FakeQuantizeTransformation(fakeQuantizeOnData.params)) } },
            {});
        ngraph::pass::low_precision::LowPrecisionTransformer transformer(transformations);
        transformer.transform(actualFunction);

        VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ actualFunction });

        referenceFunction = ngraph::builder::subgraph::FakeQuantizeFunction::getReference(
            precision,
            shape,
            fakeQuantizeOnData.params,
            fakeQuantizeOnData.expected,
            fakeQuantizeOnData.expectedSubtractValues);

        VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ referenceFunction });
    }

    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        low_precision::LayerTransformation::Params params;
        FakeQuantizeOnDataTestValues fakeQuantizeOnData;
        std::tie(precision, shape, params, fakeQuantizeOnData) = obj.param;

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, shape, fakeQuantizeOnData.params) << fakeQuantizeOnData;
        return result.str();
    }
};

TEST_P(FakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<builder::subgraph::FakeQuantizeOnData> fakeQuantizeOnDataValues = {
    { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
    // { 256ul, {}, { -1.28f} , { 1.27f } },

    // { 256ul, { 1ul }, { 0.f }, { 2.55f } },
    // { 256ul, { 1ul }, { -1.28f} , { 1.27f } }
};

const std::vector<FakeQuantizeOnDataTestValues> fakeQuantizeOnDataTestValues = {
    // U8
    {
        LayerTransformation::createParamsU8I8(),
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        {}
    },
    {
        LayerTransformation::createParamsU8I8(),
        { 256ul, {}, { -1.28f} , { 1.27f }, { -1.28f} , { 1.27f } },
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        { 1.28f }
    },

    // I8
    {
        LayerTransformation::createParamsI8I8(),
        { 256ul, {}, { -1.28f}, { 1.27f }, { -1.28f}, { 1.27f } },
        { 256ul, {}, { -1.28f}, { 1.27f }, { -1.28f}, { 1.27f } },
        {}
    },
    {
        LayerTransformation::createParamsI8I8(),
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        { 256ul, {}, { -1.28f }, { 1.27f }, { -1.28f }, { 1.27f } },
        { 1.28f }
    }
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

const std::vector<low_precision::LayerTransformation::Params> trasformationParamValues = {
    // LayerTransformation::createParamsI8I8(),
    LayerTransformation::createParamsU8I8()
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    FakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(fakeQuantizeOnDataTestValues)),
    FakeQuantizeTransformation::getTestCaseName);
