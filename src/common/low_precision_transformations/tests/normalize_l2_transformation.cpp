// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "low_precision/normalize_l2.hpp"

#include "common_test_utils/ov_test_utils.hpp"

#include "ov_lpt_models/normalize_l2.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class NormalizeL2TransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type inputPrecision;
        DequantizationOperations dequantization;
    };
    class Expected {
    public:
        ov::element::Type inputPrecision;
        DequantizationOperations dequantizationBefore;
        ov::element::Type precisionAfterOperation;
        DequantizationOperations dequantizationAfter;
    };
    TestTransformationParams transformationParams;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    ov::op::EpsMode,
    std::vector<size_t>,
    NormalizeL2TransformationTestValues> NormalizeL2TransformationParams;

class NormalizeL2Transformation : public LayerTransformation, public testing::WithParamInterface<NormalizeL2TransformationParams> {
public:
    void SetUp() override {
        ov::element::Type precision;
        ov::PartialShape shape;
        ov::op::EpsMode epsMode;
        std::vector<size_t> axes;
        NormalizeL2TransformationTestValues params;
        std::tie(precision, shape, epsMode, axes, params) = GetParam();

        actualFunction = ov::builder::subgraph::NormalizeL2Function::getOriginal(
            precision,
            params.actual.inputPrecision,
            shape,
            epsMode,
            axes,
            params.actual.dequantization);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::NormalizeL2Transformation, ov::op::v0::NormalizeL2>(params.transformationParams);
        transform.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::NormalizeL2Function::getReference(
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
        ov::element::Type precision;
        ov::PartialShape shape;
        ov::Shape axes;
        ov::op::EpsMode epsMode;
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
    auto res = compare_functions(actualFunction, referenceFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    ov::element::f16
};

std::vector<ov::op::EpsMode> epsMode = {
    ov::op::EpsMode::ADD,
    ov::op::EpsMode::MAX
};

std::vector<std::vector<size_t>> axes = {
    { 1 },
    { 1, 2, 3 }
};

namespace testValues1 {
const std::vector<ov::PartialShape> shapes = {
    { 1, 3, 16, 16 },
    { -1, -1, -1, -1}
};

const std::vector<NormalizeL2TransformationTestValues> normalizeL2TransformationTestValues = {
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        {
            ov::element::f16,
            {{ov::element::f16}, {}, {{-12.3f}, ov::element::f16, {}, false, 1ul, ov::element::f16}}
        },
        {
            ov::element::f16,
            { },
            ov::element::f32,
            {{}, {}, {{-1.f}, ov::element::f16, {}, false, 1ul, ov::element::f16}},
        }
    },
    // U8 per tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {2.f}, {-12.3f}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {2.f}, {-12.3f}},
            ov::element::f32,
            {}
        }
    },
    // U8 per tensor quantization without subtract
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {-12.3f}}
        },
        {
            ov::element::u8,
            {},
            ov::element::f32,
            {{}, {}, {-1.f}}
        }
    },
    // U8 per channel quantization with the same values
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{12.3f, 12.3f, 12.3f}}}
        },
        {
            ov::element::u8,
            {},
            ov::element::f32,
            {{}, {}, {{1.f, 1.f, 1.f}}}
        }
    },
    // U8 per channel quantization with different values
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{12.3f, -12.3f, 12.3f}}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{12.3f, -12.3f, 12.3f}}},
            ov::element::f32,
            {}
        }
    },
    // U8 not update precisions
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            ov::element::f32,
            {{}, {}, {12.3f}}
        },
        {
            ov::element::f32,
            {},
            ov::element::f32,
            {{}, {}, {1.f}}
        }
    },
    // I8 per tensor quantization
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {2.f}, {-12.3f}}
        },
        {
            ov::element::i8,
            {{ov::element::f32}, {2.f}, {-12.3f}},
            ov::element::f32,
            {}
        }
    },
    // I8 per tensor quantization without subtract
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {-12.3f}}
        },
        {
            ov::element::i8,
            {},
            ov::element::f32,
            {{}, {}, {-1.f}}
        }
    },
    // I8 per channel quantization with the same values
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{12.3f, 12.3f, 12.3f}}}
        },
        {
            ov::element::i8,
            {},
            ov::element::f32,
            {{}, {}, {{1.f, 1.f, 1.f}}}
        }
    },
    // I8 per channel quantization with different values
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{12.3f, -12.3f, 12.3f}}}
        },
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{12.3f, -12.3f, 12.3f}}},
            ov::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    NormalizeL2Transformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(epsMode),
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(normalizeL2TransformationTestValues)),
    NormalizeL2Transformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ov::PartialShape> shapesWithDynamicChannels = {
    PartialShape::dynamic()
};

const std::vector<NormalizeL2TransformationTestValues> normalizeL2TransformationTestValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {-12.3f}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {-12.3f}},
            ov::element::f32,
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    NormalizeL2Transformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapesWithDynamicChannels),
        ::testing::ValuesIn(epsMode),
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(normalizeL2TransformationTestValues)),
    NormalizeL2Transformation::getTestCaseName);
} // namespace testValues2
} // namespace
