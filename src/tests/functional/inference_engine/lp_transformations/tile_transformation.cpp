// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <gtest/gtest.h>

#include <transformations/init_node_info.hpp>
#include <low_precision/tile.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/tile_function.hpp"
#include "simple_low_precision_transformer.hpp"


namespace {
using namespace testing;
using namespace ngraph::pass;
using namespace ngraph;

class TileTransformationTestValues {
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
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple <
    ngraph::PartialShape,
    TileTransformationTestValues> TileTransformationParams;

class TileTransformation : public LayerTransformation, public testing::WithParamInterface<TileTransformationParams> {
public:
    void SetUp() override {
        const ngraph::PartialShape inputShape = std::get<0>(GetParam());
        const TileTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::TileFunction::getOriginal(
                                 inputShape,
                                 testValues.actual.precisionBeforeDequantization,
                                 testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::TileTransformation, ngraph::opset1::Clamp>(testValues.params);
        transformer.transform(actualFunction);
        referenceFunction = ngraph::builder::subgraph::TileFunction::getReference(
                                    inputShape,
                                    testValues.expected.precisionBeforeDequantization,
                                    testValues.expected.dequantizationBefore,
                                    testValues.expected.dequantizationAfter);
    }

static std::string getTestCaseName(testing::TestParamInfo<TileTransformationParams> obj) {
    const ngraph::PartialShape inputShape = std::get<0>(obj.param);
    const TileTransformationTestValues testValues = std::get<1>(obj.param);

    std::ostringstream result;
    result << toString(testValues.params) << "_" <<
           inputShape << "_" <<
           testValues.actual.precisionBeforeDequantization << "_" <<
           testValues.actual.dequantization << "_" <<
           testValues.expected.dequantizationBefore;
    return result.str();
}
};

TEST_P(TileTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ngraph::PartialShape> inputShapes = {
    ngraph::PartialShape({ 1, 3, 50, 50 }),
    ngraph::PartialShape({ -1, -1, -1, -1 }),
};

const std::vector<TileTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            element::u8,
            {{element::f32}, {128.f}, {3.f}}
        },
        {
            element::u8,
            {{}, {}, {}},
            {{element::f32}, {128.f}, {3.f}}
        }
    },
    // FP32 without convert
    {
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {{}, {128.f}, {3.f}}
        },
        {
            element::f32,
            {{}, {}, {}},
            {{}, {128.f}, {3.f}}
        }
    },
    // U8 without subtract
    {
        LayerTransformation::createParamsU8I8(),
        {
            element::u8,
            {{element::f32}, {}, {3.f}}
        },
        {
            element::u8,
            {{}, {}, {}},
            {{element::f32}, {}, {3.f}}
        }
    },
    // U8 per channel quantization with different values
    {
        LayerTransformation::createParamsU8I8(),
        {
            element::u8,
            {
                {element::f32},
                {{128.f, 0.f, 128.f / 2}},
                {{3.f, 1.f, 2.f}}
            }
        },
        {
            element::u8,
            {
                {element::f32},
                {{128.f, 0.f, 128.f / 2}},
                {{3.f, 1.f, 2.f}}
            },
            {{}, {}, {}}
        }
    },
    // U8 per channel quantization with the same values
    {
        LayerTransformation::createParamsU8I8(),
        {
            element::u8,
            {
                {element::f32},
                {{128.f, 128.f, 128.f}},
                {{3.f, 3.f, 3.f}}
            }
        },
        {
            element::u8,
            {{}, {}, {}},
            { {element::f32}, {128.f}, {3.f} },
        }
    },
    // without dequantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {{}, {}, {}}
        },
        {
            element::f32,
            {{}, {}, {}},
            {{}, {}, {}}
        },
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    TileTransformation,
    ::testing::Combine(
    ::testing::ValuesIn(inputShapes),
    ::testing::ValuesIn(testValues)),
    TileTransformation::getTestCaseName);
} // namespace testValues1
} // namespace
