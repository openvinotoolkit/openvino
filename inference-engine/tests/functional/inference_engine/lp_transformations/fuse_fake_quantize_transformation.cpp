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
#include "ngraph_functions/low_precision_transformations/fuse_fake_quantize_function.hpp"

// TODO: debug only
#include <ngraph/pass/visualize_tree.hpp>

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

// typedef std::pair<ngraph::Shape, std::vector<builder::subgraph::FakeQuantizeOnData>> FakeQuantizeShapes;

typedef std::tuple <
    ngraph::element::Type,
    ngraph::Shape,
    ngraph::element::Type,
    bool,
    std::vector<float>,
    std::vector<float>,
    ngraph::element::Type,
    > FuseFakeQuantizeTransformationParams;

class FuseFakeQuantizeTransformation : public LayerTransformation, public testing::WithParamInterface<FuseFakeQuantizeTransformationParams> {
public:
    void SetUp() override {
        ngraph::element::Type modelPrecision;
        ngraph::Shape inputShape;
        ngraph::element::Type inputPrecision;
        bool convertExists;
        std::vector<float> subtractValues;
        std::vector<float> multiplyValues;
        ngraph::element::Type fakeQuantizePrecision;
        std::tie(modelPrecision, inputShape, inputPrecision, convertExists, subtractValues, multiplyValues, fakeQuantizePrecision) = GetParam();

        // actualFunction = ngraph::builder::subgraph::FuseFakeQuantizeFunction::getOriginal(
        //    precision,
        //    shape,
        //    fakeQuantizeOnData.params,
        //    fakeQuantizeOnData.actual);

        // ngraph::pass::low_precision::LowPrecisionTransformations transformations(
        //    {},
        //    { { "FakeQuantize", ngraph::pass::low_precision::LayerTransformationPtr(
        //        new ngraph::pass::low_precision::FakeQuantizeTransformation(fakeQuantizeOnData.params)) } },
        //    {});
        // ngraph::pass::low_precision::LowPrecisionTransformer transformer(transformations);
        // transformer.transform(actualFunction);

        // referenceFunction = ngraph::builder::subgraph::FuseFakeQuantizeFunction::getReference(
        //    precision,
        //    shape,
        //    fakeQuantizeOnData.params,
        //    fakeQuantizeOnData.expected,
        //    fakeQuantizeOnData.expectedSubtractValues);

        // VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ referenceFunction });
    }

    static std::string getTestCaseName(testing::TestParamInfo<FuseFakeQuantizeTransformationParams> obj) {
        ngraph::element::Type modelPrecision;
        ngraph::Shape inputShape;
        ngraph::element::Type inputPrecision;
        bool convertExists;
        std::vector<float> subtractValues;
        std::vector<float> multiplyValues;
        ngraph::element::Type fakeQuantizePrecision;
        std::tie(modelPrecision, inputShape, inputPrecision, convertExists, subtractValues, multiplyValues, fakeQuantizePrecision) = obj.param;

        std::ostringstream result;
        result << modelPrecision << "_" << inputShape << "_" << (convertExists ? "convert" : "noConvert");
        return result.str();
        return result.str();
    }
};

TEST_P(FuseFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

// const std::vector<FuseFakeQuantizeTransformationParams> params = {
//    // U8
//    {
//        LayerTransformation::createParamsU8I8(),
//        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
//        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
//        {}
//    }
// };

// INSTANTIATE_TEST_CASE_P(
//    LPT,
//    FuseFakeQuantizeTransformation,
//    ::testing::Combine(
//        ::testing::ValuesIn(precisions),
//        ::testing::ValuesIn(shapes),
//        ::testing::ValuesIn(trasformationParamValues),
//        ::testing::ValuesIn(fakeQuantizeOnDataTestValues)),
//    FuseFakeQuantizeTransformation::getTestCaseName);
