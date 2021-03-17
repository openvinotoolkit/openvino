// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include "simple_low_precision_transformer.hpp"
#include <low_precision/normalize_l2.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

#include "lpt_ngraph_functions/normalize_l2_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class NormalizeL2TransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type inputPrecision;
        DequantizationOperations dequantization;
    };
    class Expected {
    public:
        ngraph::element::Type inputPrecision;
        DequantizationOperations dequantizationBefore;
        ngraph::element::Type precisionAfterOperation;
        DequantizationOperations dequantizationAfter;
    };
    low_precision::LayerTransformation::Params transformationParams;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    ngraph::op::EpsMode,
    std::vector<size_t>,
    NormalizeL2TransformationTestValues> NormalizeL2TransformationParams;

class NormalizeL2Transformation : public LayerTransformation, public testing::WithParamInterface<NormalizeL2TransformationParams> {
public:
    void SetUp() override {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        ngraph::op::EpsMode epsMode;
        std::vector<size_t> axes;
        NormalizeL2TransformationTestValues params;
        std::tie(precision, shape, epsMode, axes, params) = GetParam();

        actualFunction = ngraph::builder::subgraph::NormalizeL2Function::getOriginal(
            precision,
            params.actual.inputPrecision,
            shape,
            epsMode,
            axes,
            params.actual.dequantization);

        SimpleLowPrecisionTransformer transform;
        transform.add<low_precision::NormalizeL2Transformation, ngraph::opset1::NormalizeL2>(
            low_precision::LayerTransformation::Params(params.transformationParams));
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::NormalizeL2Function::getReference(
            precision,
            params.expected.inputPrecision,
            shape,
            epsMode,
            axes,
            params.expected.dequantizationBefore,
            params.expected.precisionAfterOperation,
            params.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<NormalizeL2TransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        ngraph::Shape axes;
        ngraph::op::EpsMode epsMode;
        NormalizeL2TransformationTestValues params;
        std::tie(precision, shape, epsMode, axes, params) = obj.param;

        std::ostringstream result;
        result <<
            precision << "_" <<
            toString(params.transformationParams) << shape << "_" <<
            axes << "_" << epsMode << "_" << params.actual.inputPrecision << "_" <<
            params.actual.dequantization;
        return result.str();
    }
};

TEST_P(NormalizeL2Transformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 16, 16 }
};

std::vector<ngraph::op::EpsMode> epsMode = {
    ngraph::op::EpsMode::ADD,
    ngraph::op::EpsMode::MAX
};

std::vector<std::vector<size_t>> axes = {
    { 1 },
    { 1, 2, 3 }
};

const std::vector<NormalizeL2TransformationTestValues> normalizeL2TransformationTestValues = {
//    // U8 per tensor quantization
//    {
//        LayerTransformation::createParamsU8I8(),
//        {
//            ngraph::element::u8,
//            {{ngraph::element::f32}, {2.f}, {-12.3f}}
//        },
//        {
//            ngraph::element::u8,
//            {{ngraph::element::f32}, {2.f}, {-12.3f}},
//            ngraph::element::f32,
//            {}
//        }
//    },
//    // U8 per tensor quantization without subtract
//    {
//        LayerTransformation::createParamsU8I8(),
//        {
//            ngraph::element::u8,
//            {{ngraph::element::f32}, {}, {-12.3f}}
//        },
//        {
//            ngraph::element::u8,
//            {},
//            ngraph::element::f32,
//            {{}, {}, {-1.f}}
//        }
//    },
//    // U8 per channel quantization with the same values
//    {
//        LayerTransformation::createParamsU8I8(),
//        {
//            ngraph::element::u8,
//            {{ngraph::element::f32}, {}, {{12.3f, 12.3f, 12.3f}}}
//        },
//        {
//            ngraph::element::u8,
//            {},
//            ngraph::element::f32,
//            {{}, {}, {{1.f, 1.f, 1.f}}}
//        }
//    },
//    // U8 per channel quantization with different values
//    {
//        LayerTransformation::createParamsU8I8(),
//        {
//            ngraph::element::u8,
//            {{ngraph::element::f32}, {}, {{12.3f, -12.3f, 12.3f}}}
//        },
//        {
//            ngraph::element::u8,
//            {{ngraph::element::f32}, {}, {{12.3f, -12.3f, 12.3f}}},
//            ngraph::element::f32,
//            {}
//        }
//    },
//    // U8 not update precisions
//    {
//        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
//        {
//            ngraph::element::f32,
//            {{}, {}, {12.3f}}
//        },
//        {
//            ngraph::element::f32,
//            {},
//            ngraph::element::f32,
//            {{}, {}, {1.f}}
//        }
//    },
    // U8 without dequantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {}
        }
    },
    // I8 per tensor quantization
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {2.f}, {-12.3f}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {2.f}, {-12.3f}},
            ngraph::element::f32,
            {}
        }
    },
    // I8 per tensor quantization without subtract
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {-12.3f}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::f32,
            {{}, {}, {-1.f}}
        }
    },
    // I8 per channel quantization with the same values
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {{12.3f, 12.3f, 12.3f}}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::f32,
            {{}, {}, {{1.f, 1.f, 1.f}}}
        }
    },
    // I8 per channel quantization with different values
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {{12.3f, -12.3f, 12.3f}}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {{12.3f, -12.3f, 12.3f}}},
            ngraph::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    NormalizeL2Transformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(epsMode),
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(normalizeL2TransformationTestValues)),
    NormalizeL2Transformation::getTestCaseName);
