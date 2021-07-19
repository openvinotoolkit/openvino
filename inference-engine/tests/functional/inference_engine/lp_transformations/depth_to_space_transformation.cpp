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
#include "low_precision/depth_to_space.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/depth_to_space_function.hpp"

namespace {
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;
using namespace ngraph::opset1;
using namespace ngraph;

class DepthToSpaceTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    DepthToSpace::DepthToSpaceMode mode;
    size_t blockSize;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::PartialShape,
    DepthToSpaceTransformationTestValues> DepthToSpaceTransformationParams;

class DepthToSpaceTransformation : public LayerTransformation, public testing::WithParamInterface<DepthToSpaceTransformationParams> {
public:
    void SetUp() override {
        const ngraph::PartialShape inputShape = std::get<0>(GetParam());
        const DepthToSpaceTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = DepthToSpaceFunction::getOriginal(
            inputShape,
            testValues.mode,
            testValues.blockSize,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transform;
        transform.add<low_precision::DepthToSpaceTransformation, ngraph::opset1::DepthToSpace>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = DepthToSpaceFunction::getReference(
            inputShape,
            testValues.mode,
            testValues.blockSize,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<DepthToSpaceTransformationParams> obj) {
        static std::map<DepthToSpace::DepthToSpaceMode, std::string> names = {
            {DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, "BLOCKS_FIRST"},
            {DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, "DEPTH_FIRST"},
        };

        const ngraph::PartialShape inputShape = std::get<0>(obj.param);
        const DepthToSpaceTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
            inputShape << "_" <<
            names[testValues.mode] << "_" <<
            testValues.blockSize << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization;
        return result.str();
    }
};

TEST_P(DepthToSpaceTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

namespace testValues1 {
const std::vector<ngraph::PartialShape> inputShapesForBlockSize2 = {
    { 1, 4, 3, 3 },
    {Dimension::dynamic(), 4, Dimension::dynamic(), Dimension::dynamic()}
};

const std::vector<DepthToSpaceTransformationTestValues> testValues = {
    // blockSize = 2
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        2,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {0.32f}, {0.45f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {0.32f}, {0.45f}}
        }
    },
    // blockSize = 2
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        2,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{0.32f}, ngraph::element::f32, {}, false, 1, ngraph::element::u8, true},
                {0.45f}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{0.32f}, ngraph::element::f32, {}, false, 1, ngraph::element::u8, true},
                {0.45f}
            }
        }
    },
    // not scalar-like dequantizations with different values
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        2,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{0.32f, 0.5f, 0.6f, 0.77f}},
                {{0.1f, 0.55f, 0.3f, 0.8f}}
            }
        },
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{0.32f, 0.5f, 0.6f, 0.77f}},
                {{0.1f, 0.55f, 0.3f, 0.8f}}
            },
            ngraph::element::f32,
            { {}, {}, {}}
        }
    },
    // not scalar-like dequantizations with the same values
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        2,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{0.32f, 0.32f, 0.32f, 0.32f}},
                {{0.1f, 0.1f, 0.1f, 0.1f}}
            }
        },
        {
            ngraph::element::u8,
            { {}, {}, {}},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{0.32f, 0.32f, 0.32f, 0.32f}},
                {{0.1f, 0.1f, 0.1f, 0.1f}}
            }
        }
    },
    // without dequantization
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        2,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {}
        },
        {
            ngraph::element::u8,
            { {}, {}, {}},
            ngraph::element::u8,
            { {}, {}, {}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    DepthToSpaceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesForBlockSize2),
        ::testing::ValuesIn(testValues)),
    DepthToSpaceTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ngraph::PartialShape> inputShapesForBlockSize3 = {
    { 1, 9, 3, 3 },
    {Dimension::dynamic(), 9, Dimension::dynamic(), Dimension::dynamic()}
};

const std::vector<DepthToSpaceTransformationTestValues> testValues = {
    // blockSize = 3
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        3,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {0.32f}, {0.45f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {0.32f}, {0.45f}}
        }
    },
    // DEPTH_FIRST
    {
        DepthToSpace::DepthToSpaceMode::DEPTH_FIRST,
        3,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {0.32f}, {0.45f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {0.32f}, {0.45f}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    DepthToSpaceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesForBlockSize3),
        ::testing::ValuesIn(testValues)),
    DepthToSpaceTransformation::getTestCaseName);
} // namespace testValues2

namespace testValues3 {
const std::vector<ngraph::PartialShape> inputShapesWithDynamicChannel = {
    { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
};

const std::vector<DepthToSpaceTransformationTestValues> testValues = {
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
            2,
            LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {0.32f}, {0.45f}}
        },
            {
                ngraph::element::u8,
                {{}, {}, {}},
                ngraph::element::u8,
                {{ngraph::element::f32}, {0.32f}, {0.45f}}
            }
    },
    // per-channel dequantizations with the same values
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        2,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{0.32f, 0.32f, 0.32f, 0.32f}},
                {{0.1f, 0.1f, 0.1f, 0.1f}}
            }
        },
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{0.32f, 0.32f, 0.32f, 0.32f}},
                {{0.1f, 0.1f, 0.1f, 0.1f}}
            },
            ngraph::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    DepthToSpaceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesWithDynamicChannel),
        ::testing::ValuesIn(testValues)),
    DepthToSpaceTransformation::getTestCaseName);
} // namespace testValues3

namespace testValues4 {
const std::vector<ngraph::PartialShape> inputShapesWithDynamicRank = {
    PartialShape::dynamic(),
};

const std::vector<DepthToSpaceTransformationTestValues> testValues = {
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        2,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {0.32f}, {0.45f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {0.32f}, {0.45f}},
            ngraph::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    DepthToSpaceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesWithDynamicRank),
        ::testing::ValuesIn(testValues)),
    DepthToSpaceTransformation::getTestCaseName);
} // namespace testValues4
} // namespace
