// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <utility>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include "low_precision/multiply.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/multiply_function.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class MultiplyTransformationTestValues {
public:
    low_precision::LayerTransformation::Params transformationParams;
    MultiplyValues actual;
    MultiplyValues expected;

    MultiplyTransformationTestValues() = default;

    MultiplyTransformationTestValues(
        low_precision::LayerTransformation::Params transformationParams,
        MultiplyValues actual,
        MultiplyValues expected):
        transformationParams(std::move(transformationParams)),
        actual(std::move(actual)),
        expected(std::move(expected)) {}
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    bool,
    MultiplyTransformationTestValues> MultiplyTransformationParams;

class MultiplyTransformation : public LayerTransformation, public testing::WithParamInterface<MultiplyTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        //const bool broadcast = std::get<2>(GetParam());
        const MultiplyTransformationTestValues testParams = std::get<3>(GetParam());

        actualFunction = MultiplyFunction::get(precision, shape, testParams.actual);
        SimpleLowPrecisionTransformer transform;
        transform.add<low_precision::MultiplyTransformation, ngraph::opset1::Multiply>(
            low_precision::LayerTransformation::Params(testParams.transformationParams));
        transform.transform(actualFunction);

        referenceFunction = MultiplyFunction::get(precision, shape, testParams.expected);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MultiplyTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool broadcast;
        MultiplyTransformationTestValues params;
        std::tie(precision, shape, broadcast, params) = obj.param;

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, params.transformationParams) <<
            (broadcast ? "_broadcast_" : "") <<
            params.actual <<
            params.expected;
        return result.str();
    }
};

TEST_P(MultiplyTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

const std::vector<bool> broadcastValues = {
    true,
    false
};

const std::vector<MultiplyTransformationTestValues> multiplyTransformationTestValues = {
//    // U8
//    {
//        LayerTransformation::createParamsU8I8(),
//        {
//            {
//                { 1, 3, 8, 16 },
//                {},
//                ngraph::element::u8,
//                {ngraph::element::f32, { 2.f }, { 10.f }}
//            },
//            {
//                { 1, 3, 8, 16 },
//                {},
//                ngraph::element::u8,
//                {ngraph::element::f32, { 3.f }, { 7.f }}
//            },
//            false
//        },
//        {
//            {
//                { 1, 3, 8, 16 },
//                {},
//                ngraph::element::u8,
//                {ngraph::element::f32, { 2.f }, { 10.f }}
//            },
//            {
//                { 1, 3, 8, 16 },
//                {},
//                ngraph::element::u8,
//                {ngraph::element::f32, { 3.f }, { 7.f }}
//            },
//            false
//        }
//    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::u8,
                {ngraph::element::f32, { 2.f }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::u8,
                {ngraph::element::f32, { }, { 7.f }}
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::u8,
                {ngraph::element::f32, { 2.f }, { 70.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::u8,
                {}
            },
            false
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::u8,
                { ngraph::element::f32, {  }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::u8,
                { ngraph::element::f32, { }, { 7.f } }
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::u8,
                {ngraph::element::f32, {  }, { 70.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::u8,
                {}
            },
            false
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::u8,
                {ngraph::element::f32, { 2.f }, {  }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::u8,
                {ngraph::element::f32, { }, { 7.f } }
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::u8,
                {ngraph::element::f32, { 2.f }, { 7.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::u8,
                {}
            },
            false
        }
    },

    // I8
    {
        LayerTransformation::createParamsI8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { 2.f }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { 3.f }, { 7.f }}
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { 2.f }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { 3.f }, { 7.f } }
            },
            false
        }
    },

    // Actual:
    //
    // Parameter
    //  |I8
    //  |
    // Convert Constant    Parameter
    //  \FP32  /FP32          |I8
    //   \    /               |
    //  Subtract  Constant  Convert  Constant
    //     \FP32   /FP32      \FP32  /FP32
    //      \     /            \    /
    //      Multiply          Multiply
    //             \FP32      /FP32
    //              \        /
    //               Multiply
    // Transformed:
    //
    // Parameter
    //   |I8
    //   |
    // Convert  Constant
    //   \FP32   /FP32
    //    \     /
    //   Subtract    Constant
    //      \FP32    /FP32
    //       \      /
    //      Multiply   Parameter
    //          \FP32  /I8
    //           \    /
    //          Multiply
    {
        LayerTransformation::createParamsI8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { 2.f }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { }, { 7.f }}
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { 2.f }, { 70.f }},
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {}
            },
            false
        }
    },

    // Actual:
    //
    // Parameter Constant
    //  |I8      |I8
    //  |        |
    // Convert Convert      Parameter
    //  \FP32  /FP32         |I8
    //   \    /              |
    //  Subtract  Constant  Convert  Constant
    //     \FP32   /FP32      \FP32  /FP32
    //      \     /            \    /
    //      Multiply          Multiply
    //             \FP32      /FP32
    //              \        /
    //               Multiply
    // Transformed:
    //
    // Parameter
    //   |I8
    //   |
    // Convert  Constant
    //   \FP32   /FP32
    //    \     /
    //   Subtract    Constant
    //      \FP32    /FP32
    //       \      /
    //      Multiply   Parameter
    //          \FP32  /I8
    //           \    /
    //          Multiply
    {
        LayerTransformation::createParamsI8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {
                    ngraph::element::f32,
                    { {2.f}, ngraph::element::f32, {}, true, 1ul, ngraph::element::i8, true },
                    { 10.f }
                }
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { }, { 7.f }}
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { 2.f }, { 70.f }},
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {}
            },
            false
        }
    },

    {
        LayerTransformation::createParamsI8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { }, { 7.f } }
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                { ngraph::element::f32, {  }, { 70.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                { }
            },
            false
        }
    },

    {
        LayerTransformation::createParamsI8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { 2.f }, {  }},
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { }, { 7.f } },
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { 2.f }, { 7.f }},
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {}
            },
            false
        }
    },

    // Constant as input
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { }, { 10.f }},
            },
            {
                {},
                {{ 7.f }, ngraph::element::f32}, // Constant as input
                ngraph::element::f32,
                {}
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, {}, {}},
            },
            {
                {},
                {{ 70.f }, ngraph::element::f32},
                ngraph::element::f32,
                {}
            },
            true
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { 18.f }, { 10.f }},
            },
            {
                {},
                {{ 7.f }, ngraph::element::f32},
                ngraph::element::f32,
                {}
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { 18.f }, { }},
            },
            {
                {},
                {{ 70.f }, ngraph::element::f32},
                ngraph::element::f32,
                {}
            },
            true
        }
    },

    // Constant as input
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, { }, { 0.2f }},
            },
            {
                {},
                {{ 7.f }, ngraph::element::i8}, // Constant as input
                ngraph::element::i8,
                {ngraph::element::f32, { }, { 0.5f }},
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {ngraph::element::f32, {}, {}},
            },
            {
                {},
                {{ 0.7f }, ngraph::element::f32},
                ngraph::element::f32,
                {}
            },
            true
        }
    },

    // Actual:
    //
    // Parameter Constant   Constant  Constant
    //  |I8      |I8         |I8       |I8
    //  |        |           |         |
    // Convert Convert      Convert  Convert
    //  \FP32  /FP32         |I8    /FP32
    //   \    /              |     /
    //  Subtract  Constant  Subtract  Constant
    //     \FP32   /FP32      \FP32  /FP32
    //      \     /            \    /
    //      Multiply          Multiply
    //            \FP32      /FP32
    //             \        /
    //              Multiply
    // Transformed:
    //
    // Parameter Constant
    //   |I8      |I8
    //   |        |
    // Convert   Convert
    //   \FP32   /FP32
    //    \     /
    //   Subtract    Constant
    //      \FP32    /FP32
    //       \      /
    //      Multiply
    //
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {
                    ngraph::element::f32,
                    { {127.f}, ngraph::element::f32, {}, false, 1, ngraph::element::i8, true },
                    { 0.2f }
                },
            },
            {
                {},
                {{ 7.f }, ngraph::element::i8}, // Constant as input
                ngraph::element::i8,
                {
                    ngraph::element::f32,
                    { {127.f}, ngraph::element::f32, {}, false, 1, ngraph::element::i8, true },
                    { 0.5f }
                },
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {
                    ngraph::element::f32,
                    { {127.f}, ngraph::element::f32, {}, false, 1, ngraph::element::i8, true },
                    {}
                },
            },
            {
                {},
                {{ -12.f }, ngraph::element::f32},
                ngraph::element::f32,
                {}
            },
            true
        }
    },

    // Actual:
    //
    // Constant Constant   Parameter  Constant
    //  |I8      |I8         |I8       |I8
    //  |        |           |         |
    // Convert Convert      Convert  Convert
    //  \FP32  /FP32         |I8    /FP32
    //   \    /              |     /
    //  Subtract  Constant  Subtract  Constant
    //     \FP32   /FP32      \FP32  /FP32
    //      \     /            \    /
    //      Multiply          Multiply
    //            \FP32      /FP32
    //             \        /
    //              Multiply
    // Transformed:
    //
    // Parameter Constant
    //   |I8      |I8
    //   |        |
    // Convert   Convert
    //   \FP32   /FP32
    //    \     /
    //   Subtract    Constant
    //      \FP32    /FP32
    //       \      /
    //      Multiply
    //
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                {{ 7.f }, ngraph::element::i8}, // Constant as input
                ngraph::element::i8,
                {
                    ngraph::element::f32,
                    { {127.f}, ngraph::element::f32, {}, false, 1, ngraph::element::i8, true },
                    { 0.5f }
                },
            },
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {
                    ngraph::element::f32,
                    { {127.f}, ngraph::element::f32, {}, false, 1, ngraph::element::i8, true },
                    { 0.2f }
                },
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ngraph::element::i8,
                {
                    ngraph::element::f32,
                    { {127.f}, ngraph::element::f32, {}, false, 1, ngraph::element::i8, true },
                    {}
                },
            },
            {
                {},
                {{ -12.f }, ngraph::element::f32},
                ngraph::element::f32,
                {}
            },
            true
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    MultiplyTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(broadcastValues),
        ::testing::ValuesIn(multiplyTransformationTestValues)),
    MultiplyTransformation::getTestCaseName);
