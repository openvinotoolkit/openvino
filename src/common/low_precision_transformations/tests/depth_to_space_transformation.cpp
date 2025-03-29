// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"
#include "low_precision/depth_to_space.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/depth_to_space.hpp"

namespace {
using namespace ov::pass;
using namespace ov::builder::subgraph;
using namespace ov::opset1;
using namespace ov;

class DepthToSpaceTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        ov::element::Type precisionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    DepthToSpace::DepthToSpaceMode mode;
    size_t blockSize;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ov::PartialShape,
    DepthToSpaceTransformationTestValues> DepthToSpaceTransformationParams;

class DepthToSpaceTransformation : public LayerTransformation, public testing::WithParamInterface<DepthToSpaceTransformationParams> {
public:
    void SetUp() override {
        const ov::PartialShape inputShape = std::get<0>(GetParam());
        const DepthToSpaceTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = DepthToSpaceFunction::getOriginal(
            inputShape,
            testValues.mode,
            testValues.blockSize,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::DepthToSpaceTransformation, ov::opset1::DepthToSpace>(testValues.params);
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

        const ov::PartialShape inputShape = std::get<0>(obj.param);
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
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ov::PartialShape> inputShapesForBlockSize2 = {
    { 1, 4, 3, 3 },
    {-1, -1, -1, -1}
};

const std::vector<DepthToSpaceTransformationTestValues> testValues = {
    // blockSize = 2
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        2,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {0.32f}, {0.45f}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {{ov::element::f32}, {0.32f}, {0.45f}}
        }
    },
    // blockSize = 2
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        2,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{0.32f}, ov::element::f32, {}, false, 1, ov::element::u8, true},
                {0.45f}
            }
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {
                {ov::element::f32},
                {{0.32f}, ov::element::f32, {}, false, 1, ov::element::u8, true},
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
            ov::element::u8,
            {
                {ov::element::f32},
                {{0.32f, 0.5f, 0.6f, 0.77f}},
                {{0.1f, 0.55f, 0.3f, 0.8f}}
            }
        },
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{0.32f, 0.5f, 0.6f, 0.77f}},
                {{0.1f, 0.55f, 0.3f, 0.8f}}
            },
            ov::element::f32,
            { {}, {}, {}}
        }
    },
    // not scalar-like dequantizations with the same values
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        2,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{0.32f, 0.32f, 0.32f, 0.32f}},
                {{0.1f, 0.1f, 0.1f, 0.1f}}
            }
        },
        {
            ov::element::u8,
            { {}, {}, {}},
            ov::element::u8,
            {
                {ov::element::f32},
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
            ov::element::u8,
            {}
        },
        {
            ov::element::u8,
            { {}, {}, {}},
            ov::element::u8,
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
const std::vector<ov::PartialShape> inputShapesForBlockSize3 = {
    { 1, 9, 3, 3 },
    {-1, -1, -1, -1}
};

const std::vector<DepthToSpaceTransformationTestValues> testValues = {
    // blockSize = 3
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        3,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {0.32f}, {0.45f}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {{ov::element::f32}, {0.32f}, {0.45f}}
        }
    },
    // DEPTH_FIRST
    {
        DepthToSpace::DepthToSpaceMode::DEPTH_FIRST,
        3,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {0.32f}, {0.45f}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {{ov::element::f32}, {0.32f}, {0.45f}}
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
const std::vector<ov::PartialShape> inputShapesWithDynamicRank = {
    PartialShape::dynamic(),
};

const std::vector<DepthToSpaceTransformationTestValues> testValues = {
    {
        DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        2,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {0.32f}, {0.45f}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {0.32f}, {0.45f}},
            ov::element::f32,
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
} // namespace testValues3
} // namespace
