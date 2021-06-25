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
#include <low_precision/reshape.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/reshape_function.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
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

    ngraph::Shape inputShape;
    std::vector<int> reshapeConstValues;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

inline std::ostream& operator<<(std::ostream& os, const std::vector<int>& values) {
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

const std::vector<ReshapeTransformationTestValues> testValues = {
    // U8: no subtract 3D -> 4D: channels are not affected
    {
        ngraph::Shape({ 1, 384, 1024 }),
        { 1, 384, 16, 64},
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
    // U8: no subtract 3D -> 4D: channels are not affected
    {
        ngraph::Shape({ 4, 384, 1024 }),
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
        ngraph::Shape({ 1, 3, 20 }),
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
        ngraph::Shape({ 1, 3, 20 }),
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
    // U8: no subtract 3D -> 4D: channels are not affected: with subtract
    {
        ngraph::Shape({ 1, 3, 20 }),
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
    // U8: no subtract 4D -> 6D: channels are not affected: no subtract
    {
        ngraph::Shape({ 1, 3, 4, 5 }),
        { 1, 3, 20, 1, 1, 1},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}},
            ngraph::element::f32,
            {}
        }
    },
    // U8: no subtract 4D -> 6D: channels are not affected: with subtract
    {
        ngraph::Shape({ 1, 3, 4, 5 }),
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
            {
                { ngraph::element::f32 },
                {{32, 64, 128}, ngraph::element::f32, {1, 3, 1, 1}},
                {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 3, 1, 1}}
            },
            ngraph::element::f32,
            {}
        }
    },
    // U8: no subtract 2D -> 4D: channels are affected: per tensor quantization
    // TODO: story 38439
    {
        ngraph::Shape({ 1, 16, 384, 384 }),
        { 6144, -1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}},
            ngraph::element::f32,
            {}
        }
    },
    // U8: no subtract 2D -> 4D: channels are affected: per channel quantization
    {
        ngraph::Shape({ 1, 3, 4, 5 }),
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
        ngraph::Shape({ 1, 3, 4, 8 }),
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
        ngraph::Shape({ 1, 3, 4, 8 }),
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
        ngraph::Shape({ 1, 3, 4, 8 }),
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
        ngraph::Shape({ 1, 3, 1, 1 }),
        { 1, 3, 1, 1, 1, 1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {3, 1, 1}}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}, ngraph::element::f32, {3, 1, 1}}},
            ngraph::element::f32,
            {}
        }
    },
    // U8: no subtract 4D -> 2D: channels are not affected: per tensor quantization
    // TODO: story 38439
    {
        ngraph::Shape({ 1, 3, 4, 5 }),
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
    // U8: no subtract 4D -> 2D: channels are not affected: per tensor quantization
    {
        ngraph::Shape({ 1, 3, 2, 2 }),
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
        ngraph::Shape({ 4, 3, 2, 2 }),
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
        ngraph::Shape({ 1, 3, 1, 1 }),
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
        ngraph::Shape({ 1, 3, 1, 1 }),
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
        ngraph::Shape({ 1, 3, 4, 5 }),
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
        ngraph::Shape({ 1, 3, 4, 5 }),
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
        ngraph::Shape({ 1, 2048, 1, 1 }),
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
        ngraph::Shape({ 2, 2048, 1, 1 }),
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
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {1ul}}}
        }
    },
    // U8: no subtract 4D -> 2D
    {
        ngraph::Shape({ 1, 2048, 1, 1 }),
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
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {1, 1}}}
        }
    },
    // U8: no subtract 4D -> 2D: channels are not affected
    {
        ngraph::Shape({ 2, 2048, 1, 1 }),
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
            {{ngraph::element::f32}, {}, {{0.1f}, ngraph::element::f32, {1, 1}}}
        }
    },
    // U8: no subtract 4D -> 4D: channels are affected
    {
        ngraph::Shape({ 1, 64, 320, 1 }),
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
        ngraph::Shape({ 1, 64, 320, 1 }),
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
        ngraph::Shape({ 1, 3, 12, 12 }),
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
        ngraph::Shape({ 1, 3, 12, 12 }),
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
        ngraph::Shape({ 1, 3, 12, 12 }),
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
    }
};

TEST_P(ReshapeTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ReshapeTransformation,
    ::testing::ValuesIn(testValues),
    ReshapeTransformation::getTestCaseName);

} // namespace
