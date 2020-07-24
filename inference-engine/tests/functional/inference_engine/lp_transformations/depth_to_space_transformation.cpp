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
#include "transformations/low_precision/depth_to_space.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/depth_to_space_function.hpp"

// TODO: remove after debugging
#include <ngraph/pass/visualize_tree.hpp>

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;
using namespace ngraph::opset1;

class DepthToSpaceTransformationTestValues {
public:
    low_precision::LayerTransformation::Params transformationParams;
    DepthToSpaceActualValues actual;
    DepthToSpaceExpectedValues expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    DepthToSpace::DepthToSpaceMode,
    size_t,
    DepthToSpaceTransformationTestValues> DepthToSpaceTransformationParams;

class DepthToSpaceTransformation : public LayerTransformation, public testing::WithParamInterface<DepthToSpaceTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const DepthToSpace::DepthToSpaceMode mode = std::get<2>(GetParam());
        const size_t blockSize = std::get<3>(GetParam());
        const DepthToSpaceTransformationTestValues testParams = std::get<4>(GetParam());

        actualFunction = DepthToSpaceFunction::getOriginal(
            precision,
            shape,
            mode,
            blockSize,
            testParams.actual);

        SimpleLowPrecisionTransformer transform;
        transform.add<low_precision::DepthToSpaceTransformation, ngraph::opset1::DepthToSpace>(
            low_precision::LayerTransformation::Params(testParams.transformationParams));
        transform.transform(actualFunction);

        referenceFunction = DepthToSpaceFunction::getReference(
            precision,
            shape,
            mode,
            blockSize,
            testParams.expected);
    }

    static std::string getTestCaseName(testing::TestParamInfo<DepthToSpaceTransformationParams> obj) {
        static std::map<DepthToSpace::DepthToSpaceMode, std::string> names = {
            {DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, "BLOCKS_FIRST"},
            {DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, "DEPTH_FIRST"},
        };

        ngraph::element::Type precision;
        ngraph::Shape shape;
        DepthToSpace::DepthToSpaceMode mode;
        size_t blockSize;
        DepthToSpaceTransformationTestValues params;

        std::tie(precision, shape, mode, blockSize, params) = obj.param;

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, params.transformationParams) <<
            "_" << names[mode] << "_" << blockSize << "_" << params.actual << "_" << params.expected;
        return result.str();
    }
};

TEST_P(DepthToSpaceTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

const std::vector<DepthToSpace::DepthToSpaceMode> modes = {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        DepthToSpace::DepthToSpaceMode::DEPTH_FIRST
};

const std::vector<DepthToSpaceTransformationTestValues> depthToSpaceTransformationTestValues = {
    // I8

    {
        LayerTransformation::createParamsI8I8(),
        {ngraph::element::i8, { 2.3f }, { 4.5f }},
        {ngraph::element::i8, { 2.3f }, { 4.5f }}
    },

    // U8

    {
        LayerTransformation::createParamsU8I8(),
        {ngraph::element::u8, { 2.3f }, { 4.5f }},
        {ngraph::element::u8, { 2.3f }, { 4.5f }}
    }
};

const std::vector<ngraph::Shape> inputShapesBS2 = {
        {1, 4, 3, 3}, {2, 16, 5, 4}
};

const auto DepthToSpaceBS2 = ::testing::Combine(
    ::testing::ValuesIn(precisions),
    ::testing::ValuesIn(inputShapesBS2),
    ::testing::ValuesIn(modes),
    ::testing::Values(2),
    ::testing::ValuesIn(depthToSpaceTransformationTestValues)
);

INSTANTIATE_TEST_CASE_P(LPT_BS2, DepthToSpaceTransformation, DepthToSpaceBS2, DepthToSpaceTransformation::getTestCaseName);

const std::vector<ngraph::Shape> inputShapesBS3 = {
        {1, 9, 3, 3}, {2, 27, 5, 4}
};

const auto DepthToSpaceBS3 = ::testing::Combine(
    ::testing::ValuesIn(precisions),
    ::testing::ValuesIn(inputShapesBS3),
    ::testing::ValuesIn(modes),
    ::testing::Values(3),
    ::testing::ValuesIn(depthToSpaceTransformationTestValues)
);

INSTANTIATE_TEST_CASE_P(LPT_BS3, DepthToSpaceTransformation, DepthToSpaceBS3, DepthToSpaceTransformation::getTestCaseName);
