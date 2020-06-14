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
#include "ngraph_functions/low_precision_transformations/normalize_l2_function.hpp"

using namespace testing;
using namespace ngraph::pass;

typedef std::tuple<
    ngraph::element::Type,
    std::pair<ngraph::Shape, ngraph::Shape>,
    low_precision::LayerTransformation::Params,
    bool,
    bool> NormalizeL2TransformationParams;

class NormalizeL2Transformation : public LayerTransformation, public testing::WithParamInterface<NormalizeL2TransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const std::pair<ngraph::Shape, ngraph::Shape> shapes = std::get<1>(GetParam());
        const low_precision::LayerTransformation::Params params = std::get<2>(GetParam());
        const bool fuseMultiply = std::get<3>(GetParam());
        const bool shift = std::get<4>(GetParam());

        actualFunction = ngraph::builder::subgraph::NormalizeL2Function::getOriginal(
            precision,
            shapes,
            params.precisionsOnActivations[0],
            fuseMultiply,
            shift);

        transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::NormalizeL2Function::getReference(
            precision,
            shapes,
            params.precisionsOnActivations[0],
            fuseMultiply,
            shift);
    }

    static std::string getTestCaseName(testing::TestParamInfo<NormalizeL2TransformationParams> obj) {
        ngraph::element::Type precision;
        std::pair<ngraph::Shape, ngraph::Shape> shapes;
        low_precision::LayerTransformation::Params params;
        bool fuseMultiply;
        bool shift;
        std::tie(precision, shapes, params, fuseMultiply, shift) = obj.param;

        std::ostringstream result;
        result << precision << "_" << shapes.first << "_" << shapes.second << "_" <<
            toString(params) <<
            (fuseMultiply ? "_multiply" : "") <<
            (shift ? "_shift" : "");

        return result.str();
    }

protected:
    std::shared_ptr<ngraph::Function> actualFunction;
    std::shared_ptr<ngraph::Function> referenceFunction;
};

TEST_P(NormalizeL2Transformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    ConvFusion().run_on_function(actualFunction);

    actualFunction->validate_nodes_and_infer_types();

    // auto res = compare_functions(referenceFunction, actualFunction);
    // ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<std::pair<ngraph::Shape, ngraph::Shape>> inputAndQuantizationShapes = {
    // { { 1ul, 4ul, 16ul, 16ul }, { 1ul } },
    { { 1ul, 4ul, 16ul, 16ul }, { 1ul, 4ul, 1ul, 1ul } },
    // { { 1, 4, 16 }, { 16, 1 } }
};

const std::vector<low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTransformation::createParamsI8I8(),
    LayerTransformation::createParamsU8I8()
};

const std::vector<bool> fuseMultiply = { true, false };
const std::vector<bool> shift = { true, false };

INSTANTIATE_TEST_CASE_P(
    LPT,
    NormalizeL2Transformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(inputAndQuantizationShapes),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(fuseMultiply),
        ::testing::ValuesIn(shift)),
    NormalizeL2Transformation::getTestCaseName);
