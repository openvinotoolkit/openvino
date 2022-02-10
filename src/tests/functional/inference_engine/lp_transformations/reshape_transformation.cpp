// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/reshape.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/reshape_function.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class ReshapeTransformationTestValues {
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

    ngraph::PartialShape inputShape;
    std::vector<int> reshapeConstValues; // if empty then create shapeOf
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

class ReshapeTransformation : public LayerTransformation, public testing::WithParamInterface<ReshapeTransformationTestValues> {
public:
    void SetUp() override {
        const ReshapeTransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::ReshapeFunction::getOriginal(
            testValues.inputShape,
            testValues.reshapeConstValues,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::ReshapeTransformation, ngraph::opset1::Reshape>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ReshapeFunction::getReference(
            testValues.inputShape,
            testValues.reshapeConstValues,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ReshapeTransformationTestValues> obj) {
        const ReshapeTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.inputShape << "_" <<
            testValues.reshapeConstValues << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.precisionAfterOperation << "_" <<
            testValues.expected.dequantizationAfter << "_" <<
            testValues.expected.dequantizationBefore;
        return result.str();
    }
};

TEST_P(ReshapeTransformation, CompareFunctions) {
    InitNodeInfo().run_on_model(actualFunction);
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<ReshapeTransformationTestValues> testValues = {
    // U8: no subtract 3D -> 4D: channels are not affected
    {
        { 1, 384, 1024 },
        { 1, 384, 16, 64 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        }
    },
    // U8: 3D -> 4D: dynamic shape
    {
        { -1, -1, -1 },
        { 0, 384, 16, 64 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        }
    },
    // U8: 3D -> 4D: dynamic rank: per tensor quantization
    {
        PartialShape::dynamic(),
        { 0, 384, 16, 64 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        }
    },
    // U8: 3D -> 4D: dynamic rank: per tensor quantization
    {
        PartialShape::dynamic(),
        { 0, 384, 16, 64 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {128}, {0.1f}}
        }
    },
    // U8: 3D -> 4D: dynamic rank
    {
        PartialShape::dynamic(),
        { 0, 3, 16, 64 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, element::f32, {1, 3, 1, 1}}},
            ngraph::element::f32,
            {}
        }
    },
    // U8: no subtract 3D -> 4D: channels are not affected
    {
        { 4, 384, 1024 },
        { 4, 384, 16, 64},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        }
    },
    // U8: no subtract 3D -> 4D: channels are not affected: no subtract
    {
        { 1, 3, 20 },
        { 1, 3, 4, 5},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}}
        }
    },
    // U8: no subtract 3D -> 4D: channels are not affected: no subtract
    {
        ngraph::Shape({ 4, 3, 20 }),
        { 4, 3, 4, 5},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}}
        }
    },
    // U8: no subtract 3D -> 4D: channels are not affected: with subtract
    {
        { 1, 3, 20 },
        { 1, 3, 4, 5},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{32, 64, 128}, ngraph::element::f32, {1, 3, 1}},
                {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1}}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{32, 64, 128}, ngraph::element::f32, {1, 3, 1, 1}},
                {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}
            }
        }
    },
    // U8: with subtract 3D -> 4D: channels are not affected, dynamic batch
    {
        { -1, 3, 20 },
        { 0, 3, 4, 5},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{32, 64, 128}, ngraph::element::f32, {1, 3, 1}},
                {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1}}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{32, 64, 128}, ngraph::element::f32, {1, 3, 1, 1}},
                {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}
            }
        }
    },
    // U8: with subtract 3D -> 4D: channels are not affected, dynamic shape
    {
        { -1, -1, -1 },
        { 0, 3, 4, 5},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{32, 64, 128}, ngraph::element::f32, {1, 3, 1}},
                {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1}}
            }
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{32, 64, 128}, ngraph::element::f32, {1, 3, 1, 1}},
                {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}
            }
        }
    },
    // U8: no subtract 3D -> 4D: channels are not affected: with subtract
    {
        { 1, 3, 20 },
        { 1, -1, 4, 5},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{32, 64, 128}, ngraph::element::f32, {1, 3, 1}},
                {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1}}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{32, 64, 128}, ngraph::element::f32, {1, 3, 1, 1}},
                {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}
            }
        }
    },
    // U8: no subtract 4D -> 3D: rfcn-resnet101-coco
    {
        { 100, 4, 1, 1 },
        { -1, 1, 400},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32}, {}, {{0.1f, 0.1f, 0.1f, 0.1f}, ngraph::element::f32, {1, 4, 1, 1}}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                {ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {}}
            }
        }
    },
    // U8: no subtract 4D -> 6D: channels are not affected: no subtract
    {
        { 1, 3, 4, 5 },
        { 1, 3, 20, 1, 1, 1},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1, 1, 1}}},
        }
    },
    // U8: no subtract 4D -> 6D: channels are not affected: with subtract
    {
        { 1, 3, 4, 5 },
        { 1, 3, 20, 1, 1, 1},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{32, 64, 128}, ngraph::element::f32, {1, 3, 1, 1}},
                {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}
            }
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                { ngraph::element::f32 },
                {{32, 64, 128}, ngraph::element::f32, {1, 3, 1, 1, 1, 1}},
                {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1, 1, 1}}
            }
        }
    },
    // U8: no subtract 4D -> 2D: channels are affected: per tensor quantization
    {
        { 1, 16, 384, 384 },
        { 6144, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        }
    },
    // U8: no subtract 4D -> 2D: channels are affected: per channel quantization
    {
        { 1, 3, 4, 5 },
        { 12, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // U8: no subtract 2D -> 4D: channels are affected: per channel quantization
    {
        { 1, 3, 4, 8 },
        { 12, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}}, {{0.1f, 0.2f, 0.3f}}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}, ngraph::element::f32}, {{0.1f, 0.2f, 0.3f}}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // empty: FP32
    {
        { 1, 3, 4, 8 },
        { 12, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // empty: U8
    {
        { 1, 3, 4, 8 },
        { 12, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {}
        }
    },
    // U8: no subtract 4D -> 6D: channels are not affected: no subtract
    {
        { 1, 3, 1, 1 },
        { 1, 3, 1, 1, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1, 1, 1}}},
        }
    },
    // U8: no subtract 4D -> 5D: channels are not affected: no subtract
    {
        { 1, 3, 4, 5 },
        { 1, 3, 20, 1, 1},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1, 1}}},
        }
    },
    // U8: no subtract 4D -> 5D: channels are affected: no subtract
    {
        { 1, 3, 2, 3 },
        { 1, 18, 1, 1, 1},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {
                    {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f},
                    ngraph::element::f32,
                    {1, 18, 1, 1, 1}
                }
            },
        }
    },
    // U8: no subtract 4D -> 5D: channels are affected: no subtract
    {
        { 1, 3, 4, 5 },
        { 1, 12, 1, 1, 5},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {}}},
        }
    },
    // U8: no subtract 4D -> 5D: channels are affected: no subtract
    {
        { 1, 3, 4, 5 },
        { 1, 12, 1, 1, 5},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f, 0.2f, 0.3f, 0.3f, 0.3f, 0.3f}, ngraph::element::f32, {1, 12, 1, 1, 1}}
            }
        }
    },
    // U8: no subtract 4D -> 2D: channels are not affected: per tensor quantization
    {
        { 1, 3, 4, 5 },
        { 0, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f}, ngraph::element::f32, {}}, {{0.1f}, ngraph::element::f32, {}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f}, ngraph::element::f32, {}}, {{0.1f}, ngraph::element::f32, {}}}
        }
    },
    // U8: no subtract 4D -> 2D: channels are affected: per channel quantization
    {
        { 1, 3, 2, 2 },
        { 0, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}, ngraph::element::f32, {1, 3, 1, 1}}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{0.f, 0.f, 0.f, 0.f, 128.f, 128.f, 128.f, 128.f, 255.f, 255.f, 255.f, 255.f}, ngraph::element::f32, {1, 12}},
                {{0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f, 0.2f, 0.3f, 0.3f, 0.3f, 0.3f}, ngraph::element::f32, {1, 12}}
            }
        }
    },
    // U8: no subtract 4D -> 2D: channels are affected: per channel quantization
    {
        { 1, 3, 2, 2 },
        { 0, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}, ngraph::element::f32, {3, 1, 1}}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{0.f, 0.f, 0.f, 0.f, 128.f, 128.f, 128.f, 128.f, 255.f, 255.f, 255.f, 255.f}, ngraph::element::f32, {1, 12}},
                {{0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f, 0.2f, 0.3f, 0.3f, 0.3f, 0.3f}, ngraph::element::f32, {1, 12}}
            }
        }
    },
    // U8: 4D -> 2D: per channel dq and dynamic batch
    {
        { -1, 3, 2, 2 },
        { 0, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}, ngraph::element::f32, {1, 3, 1, 1}}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{0.f, 0.f, 0.f, 0.f, 128.f, 128.f, 128.f, 128.f, 255.f, 255.f, 255.f, 255.f}, ngraph::element::f32, {1, 12}},
                {{0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f, 0.2f, 0.3f, 0.3f, 0.3f, 0.3f}, ngraph::element::f32, {1, 12}}
            }
        }
    },
    // U8: no subtract 4D -> 2D: channels are not affected: per tensor quantization
    {
        { 4, 3, 2, 2 },
        { 0, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}, ngraph::element::f32, {1, 3, 1, 1}}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{0.f, 0.f, 0.f, 0.f, 128.f, 128.f, 128.f, 128.f, 255.f, 255.f, 255.f, 255.f}, ngraph::element::f32, {1, 12}},
                {{0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f, 0.2f, 0.3f, 0.3f, 0.3f, 0.3f}, ngraph::element::f32, {1, 12}}
            }
        }
    },
    // U8: no subtract 4D -> 2D: channels are not affected: per channel quantization: case #1: dequantization operation constant needs broadcast
    {
        { 1, 3, 1, 1 },
        { 0, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}, ngraph::element::f32, {3, 1, 1}}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}, ngraph::element::f32, {1, 3}}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3}}},
        }
    },
    // U8: no subtract 4D -> 2D: channels are not affected: per channel quantization: case #2: dequantization operation constant doesn't need broadcast
    {
        { 1, 3, 1, 1 },
        { 0, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}, ngraph::element::f32, {1, 3, 1, 1}}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}, ngraph::element::f32, {1, 3}}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3}}},
        }
    },
    // U8: no subtract 4D -> 3D: channels are affected: per tensor quantization: case #1: dequantization operation constant needs broadcast
    {
        { 1, 3, 4, 5 },
        { 0, 0, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}, ngraph::element::f32, {3, 1, 1}}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}, ngraph::element::f32, {1, 3, 1}}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1}}},
        }
    },
    // U8: no subtract 4D -> 3D: channels are affected: per tensor quantization: case #2: dequantization operation constant doesn't need broadcast
    {
        { 1, 3, 4, 5 },
        { 0, 0, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}, ngraph::element::f32, {1, 3, 1, 1}}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{0.f, 128.f, 255.f}, ngraph::element::f32, {1, 3, 1}}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1}}},
        }
    },
    // U8: no subtract 4D -> 2D
    {
        { 1, 2048, 1, 1 },
        { 1, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {}}}
        }
    },
    // U8: no subtract 4D -> 2D
    {
        { 2, 2048, 1, 1 },
        { 2, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {1ul}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {}}}
        }
    },
    // U8: no subtract 4D -> 2D
    {
        { 1, 2048, 1, 1 },
        { 1, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {1, 1, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {}}}
        }
    },
    // U8: no subtract 4D -> 2D: channels are not affected
    {
        { 2, 2048, 1, 1 },
        { 2, -1},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {1, 1, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {}}}
        }
    },
    // U8: no subtract 4D -> 2D: channels are not affected, dynamic batch
    {
        { -1, 2048, 1, 1 },
        { 0, -1},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {1, 1, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {}}}
        }
    },
    // U8: no subtract 4D -> 4D: channels are affected
    {
        { 1, 64, 320, 1 },
        { 0, 2, 3, 1},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {}}}
        }
    },
    // U8: with subtract 4D -> 4D: channels are affected
    {
        { 1, 64, 320, 1 },
        { 0, 2, 3, 1},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f}, ngraph::element::f32, {}}, {{0.1f}, ngraph::element::f32, {}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f}, ngraph::element::f32, {}}, {{0.1f}, ngraph::element::f32, {}}}
        }
    },
    // U8: with subtract 4D -> 3D, Dq after convolution: face-detection-0205 case
    {
        { 1, 3, 12, 12 },
        { 0, 3, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{128.f, 12.8f, 128.f}, ngraph::element::f32, {3, 1, 1}},
                {{0.1f, 0.01f, 0.1f}, ngraph::element::f32, {3, 1, 1}}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{128.f, 12.8f, 128.f}, ngraph::element::f32, {1, 3, 1}},
                {{0.1f, 0.01f, 0.1f}, ngraph::element::f32, {1, 3, 1}}
            }
        }
    },
    // U8: without subtract 4D -> 3D, Dq after convolution: face-detection-0205 case
    {
        { 1, 3, 12, 12 },
        { 0, 3, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{0.1f, 0.01f, 0.1f}, ngraph::element::f32, {3, 1, 1}}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{0.1f, 0.01f, 0.1f}, ngraph::element::f32, {1, 3, 1}}
            }
        }
    },
    // U8: without subtract 4D -> 3D, Dq after convolution
    {
        { 1, 3, 12, 12 },
        { 0, -1, 144 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{0.1f, 0.01f, 0.1f}, ngraph::element::f32, {3, 1, 1}}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{0.1f, 0.01f, 0.1f}, ngraph::element::f32, {1, 3, 1}}
            }
        }
    },
    // U8: without subtract 2D -> 2D
    {
        { 1, 3 },
        { 1, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{0.1f, 0.01f, 0.1f}, ngraph::element::f32, {1, 3}}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{0.1f, 0.01f, 0.1f}, ngraph::element::f32, {1, 3}}
            }
        }
    },
    // U8: without subtract 2D -> 2D
    {
        { -1, 2 },
        { -1, 6 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{0.1f, 0.02f}, ngraph::element::f32, {1, 2}}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {},
                {{0.1f,  0.02f, 0.1f, 0.02f, 0.1f, 0.02f}, ngraph::element::f32, {1, 6}}
            }
        }
    },
    // Nondequantization multiply (I32 precision)
    {
        { 1, 384, 1024 },
        { 1, 384, 16, 64 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::i32,
            {{}, {}, {2}}
        },
        {
            ngraph::element::i32,
            {{}, {}, {2}},
            ngraph::element::i32,
            {}
        }
    },
    // U8: non-const reshape pattern and per-tensor dequantization
    {
        { -1, -1, -1, -1 },
        {},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.1f}}
        }
    },
    // U8: non-const reshape pattern and per-channel dequantization
    {
        { -1, 3, -1, -1 },
        {},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 124.f, 120.f}}, {{0.1f, 1.f, 10.f}}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 124.f, 120.f}}, {{0.1f, 1.f, 10.f}}},
            ngraph::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ReshapeTransformation,
    ::testing::ValuesIn(testValues),
    ReshapeTransformation::getTestCaseName);

} // namespace
