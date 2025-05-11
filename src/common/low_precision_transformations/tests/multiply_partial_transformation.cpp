// Copyright (C) 2018-2025 Intel Corporation
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
#include "low_precision/multiply_partial.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/multiply_partial_function.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class MultiplyPartialTransformationTestValues {
public:
    TestTransformationParams transformationParams;
    MultiplyPartialValues actual;
    MultiplyPartialValues expected;

    MultiplyPartialTransformationTestValues() = default;

    MultiplyPartialTransformationTestValues(
        TestTransformationParams transformationParams,
        MultiplyPartialValues actual,
        MultiplyPartialValues expected):
        transformationParams(std::move(transformationParams)),
        actual(std::move(actual)),
        expected(std::move(expected)) {}
};

typedef std::tuple<
    ov::element::Type,
    MultiplyPartialTransformationTestValues> MultiplyPartialTransformationParams;

class MultiplyPartialTransformation : public LayerTransformation, public testing::WithParamInterface<MultiplyPartialTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const MultiplyPartialTransformationTestValues testParams = std::get<1>(GetParam());

        actualFunction = MultiplyPartialFunction::get(precision, testParams.actual);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::MultiplyPartialTransformation, ov::op::v1::Multiply>(testParams.transformationParams);
        transform.transform(actualFunction);

        referenceFunction = MultiplyPartialFunction::get(precision, testParams.expected);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MultiplyPartialTransformationParams> obj) {
        const ov::element::Type precision = std::get<0>(obj.param);
        const MultiplyPartialTransformationTestValues testParams = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, testParams.expected.branch1.inputShape, testParams.transformationParams) <<
            testParams.actual <<
            testParams.expected;
        return result.str();
    }
};

TEST_P(MultiplyPartialTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<MultiplyPartialTransformationTestValues> multiplyTransformationTestValues = {
    // U8
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::u8,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::u8,
                {ov::element::f32, { 3.f }, { 7.f }}
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::u8,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::u8,
                {ov::element::f32, { 3.f }, { 7.f }}
            },
            false
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::u8,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::u8,
                {ov::element::f32, { 3.f }, { 7.f }}
            },
            false
        },
        {
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::u8,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::u8,
                {ov::element::f32, { 3.f }, { 7.f }}
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
                ov::element::u8,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::u8,
                {ov::element::f32, { }, { 7.f }}
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::u8,
                {ov::element::f32, { 2.f }, { 70.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::u8,
                {}
            },
            false
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::u8,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::u8,
                {ov::element::f32, { }, { 7.f }}
            },
            false
        },
        {
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::u8,
                {ov::element::f32, { 2.f }, { 70.f }}
            },
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::u8,
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
                ov::element::u8,
                { ov::element::f32, {  }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::u8,
                { ov::element::f32, { }, { 7.f } }
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::u8,
                {ov::element::f32, {  }, { 70.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::u8,
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
                ov::element::u8,
                {ov::element::f32, { 2.f }, {  }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::u8,
                {ov::element::f32, { }, { 7.f } }
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::u8,
                {ov::element::f32, { 2.f }, { 7.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::u8,
                {}
            },
            false
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::u8,
                {ov::element::f32, { 2.f }, {  }}
            },
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::u8,
                {ov::element::f32, { }, { 7.f } }
            },
            false
        },
        {
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::u8,
                {ov::element::f32, { 2.f }, { 7.f }}
            },
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::u8,
                {}
            },
            false
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                PartialShape::dynamic(),
                {},
                ov::element::u8,
                {ov::element::f32, { 2.f }, {  }}
            },
            {
                PartialShape::dynamic(),
                {},
                ov::element::u8,
                {ov::element::f32, { }, { 7.f } }
            },
            false
        },
        {
            {
                PartialShape::dynamic(),
                {},
                ov::element::u8,
                {ov::element::f32, { 2.f }, {  }}
            },
            {
                PartialShape::dynamic(),
                {},
                ov::element::u8,
                {ov::element::f32, { }, { 7.f } }
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
                ov::element::i8,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, { 3.f }, { 7.f }}
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, { 3.f }, { 7.f } }
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
                ov::element::i8,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, { }, { 7.f }}
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, { 2.f }, { 70.f }},
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
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
                ov::element::i8,
                {
                    ov::element::f32,
                    { {2.f}, ov::element::f32, {}, true, 1ul, ov::element::i8, true },
                    { 10.f }
                }
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, { }, { 7.f }}
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, { 2.f }, { 70.f }},
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
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
                ov::element::i8,
                {ov::element::f32, { }, { 10.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, { }, { 7.f } }
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                { ov::element::f32, {  }, { 70.f }}
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
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
                ov::element::i8,
                {ov::element::f32, { 2.f }, {  }},
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, { }, { 7.f } },
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, { 2.f }, { 7.f }},
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
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
                ov::element::i8,
                {ov::element::f32, { }, { 10.f }},
            },
            {
                {},
                {{ 7.f }, ov::element::f32}, // Constant as input
                ov::element::f32,
                {}
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, {}, {}},
            },
            {
                {},
                {{ 70.f }, ov::element::f32},
                ov::element::f32,
                {}
            },
            true
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::i8,
                {ov::element::f32, { }, { 10.f }},
            },
            {
                {},
                {{ 7.f }, ov::element::f32}, // Constant as input
                ov::element::f32,
                {}
            },
            false
        },
        {
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::i8,
                {ov::element::f32, {}, {}},
            },
            {
                {},
                {{ 70.f }, ov::element::f32},
                ov::element::f32,
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
                ov::element::i8,
                {ov::element::f32, { 18.f }, { 10.f }},
            },
            {
                {},
                {{ 7.f }, ov::element::f32},
                ov::element::f32,
                {}
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, { 18.f }, { }},
            },
            {
                {},
                {{ 70.f }, ov::element::f32},
                ov::element::f32,
                {}
            },
            true
        }
    },

    // Constant as input with empty shape
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, { }, { 0.2f }},
            },
            {
                {},
                {{ 7.f }, ov::element::i8}, // Constant as input
                ov::element::i8,
                {ov::element::f32, { }, { 0.5f }},
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {ov::element::f32, {}, {}},
            },
            {
                {},
                {{ 0.7f }, ov::element::f32},
                ov::element::f32,
                {}
            },
            true
        }
    },

    // Constant as input with 1 dimension shape
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                {{ 7.f, 8.f, 9.f }, ov::element::i8, ov::Shape{3}}, // Constant as input
                ov::element::i8,
                {ov::element::f32, { }, { {0.1f, 0.2f, 0.3f}, element::f32, ov::Shape{3} }},
            },
            {
                { 1, 2, 3 },
                {},
                ov::element::f32,
                {{}, {}, {{0.2f, 0.3f, 0.4f}, element::f32, ov::Shape{3}}},
            },
            false
        },
        {
            {
                { 1, 2, 3 },
                {},
                ov::element::f32,
                {},
            },
            {
                {},
                { {0.14f, 0.48f, 1.08f}, ov::element::f32, ov::Shape{3}}, // Constant as input
                {},
                {},
            },
            true
        }
    },

    // Parameter as input with, Constant with 1 dimension shape
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                { 1, 2, 3 },
                {},
                ov::element::f32,
                {{}, {}, {{0.2f, 0.3f, 0.4f}, element::f32, ov::Shape{3}}},
            },
            {
                {},
                {{ 7.f, 8.f, 9.f }, ov::element::i8, ov::Shape{3}}, // Constant as input
                ov::element::i8,
                {ov::element::f32, { }, { {0.1f, 0.2f, 0.3f}, element::f32, ov::Shape{3} }},
            },
            false
        },
        {
            {
                { 1, 2, 3 },
                {},
                ov::element::f32,
                {},
            },
            {
                {},
                { {0.14f, 0.48f, 1.08f}, ov::element::f32, ov::Shape{3}}, // Constant as input
                {},
                {},
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
                ov::element::i8,
                {
                    ov::element::f32,
                    { {127.f}, ov::element::f32, {}, false, 1, ov::element::i8, true },
                    { 0.2f }
                },
            },
            {
                {},
                {{ 7.f }, ov::element::i8}, // Constant as input
                ov::element::i8,
                {
                    ov::element::f32,
                    { {127.f}, ov::element::f32, {}, false, 1, ov::element::i8, true },
                    { 0.5f }
                },
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {
                    ov::element::f32,
                    { {127.f}, ov::element::f32, {}, false, 1, ov::element::i8, true },
                    {}
                },
            },
            {
                {},
                {{ -12.f }, ov::element::f32},
                ov::element::f32,
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
                {{ 7.f }, ov::element::i8}, // Constant as input
                ov::element::i8,
                {
                    ov::element::f32,
                    { {127.f}, ov::element::f32, {}, false, 1, ov::element::i8, true },
                    { 0.5f }
                },
            },
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {
                    ov::element::f32,
                    { {127.f}, ov::element::f32, {}, false, 1, ov::element::i8, true },
                    { 0.2f }
                },
            },
            false
        },
        {
            {
                { 1, 3, 8, 16 },
                {},
                ov::element::i8,
                {
                    ov::element::f32,
                    { {127.f}, ov::element::f32, {}, false, 1, ov::element::i8, true },
                    {}
                },
            },
            {
                {},
                {{ -12.f }, ov::element::f32},
                ov::element::f32,
                {}
            },
            true
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                {{ 7.f }, ov::element::i8}, // Constant as input
                ov::element::i8,
                {
                    ov::element::f32,
                    { {127.f}, ov::element::f32, {}, false, 1, ov::element::i8, true },
                    { 0.5f }
                },
            },
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::i8,
                {
                    ov::element::f32,
                    { {127.f}, ov::element::f32, {}, false, 1, ov::element::i8, true },
                    { 0.2f }
                },
            },
            false
        },
        {
            {
                { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
                {},
                ov::element::i8,
                {
                    ov::element::f32,
                    { {127.f}, ov::element::f32, {}, false, 1, ov::element::i8, true },
                    {}
                },
            },
            {
                {},
                {{ -12.f }, ov::element::f32},
                ov::element::f32,
                {}
            },
            true
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MultiplyPartialTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(multiplyTransformationTestValues)),
    MultiplyPartialTransformation::getTestCaseName);
} // namespace
