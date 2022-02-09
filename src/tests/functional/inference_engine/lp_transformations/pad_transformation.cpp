// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <gtest/gtest.h>
#include <ngraph/ngraph.hpp>

#include <low_precision/pad.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/pad_function.hpp"
#include "simple_low_precision_transformer.hpp"


namespace {
using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class PadTransformationTestValues {
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

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple <
    ngraph::PartialShape, // input Shape
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>>, // pads begin, pads end
    ngraph::op::PadMode, // pads mode
    float, // pads value (used if mode == CONSTANT)
    PadTransformationTestValues> PadTransformationParams;

class PadTransformation : public LayerTransformation, public testing::WithParamInterface<PadTransformationParams> {
public:
    void SetUp() override {
        const ngraph::PartialShape inputShape = std::get<0>(GetParam());
        const auto pads = std::get<1>(GetParam());
        const ngraph::op::PadMode padsMode = std::get<2>(GetParam());
        const float padsValue = std::get<3>(GetParam());
        const PadTransformationTestValues testValues = std::get<4>(GetParam());

        const auto precisionAfterActualOp = testValues.actual.dequantization.convert.empty() ?
            testValues.actual.precisionBeforeDequantization : testValues.actual.dequantization.convert.outPrecision;

        actualFunction = ngraph::builder::subgraph::PadFunction::get(
            inputShape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization,
            pads.first,
            pads.second,
            padsMode,
            padsValue,
            precisionAfterActualOp,
            { {}, {}, {} });

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::PadTransformation, ngraph::opset1::Pad>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::PadFunction::get(
            inputShape,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            pads.first,
            pads.second,
            padsMode,
            padsValue,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<PadTransformationParams> obj) {
        const ngraph::PartialShape inputShape = std::get<0>(obj.param);
        const auto pads = std::get<1>(obj.param);
        const ngraph::op::PadMode padsMode = std::get<2>(obj.param);
        const float padsValue = std::get<3>(obj.param);
        const PadTransformationTestValues testValues = std::get<4>(obj.param);

        std::ostringstream result;
        result << "mode_" << padsMode << "_";
        if (padsMode == ngraph::op::PadMode::CONSTANT) {
            result << "pad_value_{ " << padsValue << " }";
        }

        result << "_" <<
            toString(testValues.params) << "_" <<
            inputShape << "_" << pads.first << "_" << pads.second << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore;

        return result.str();
    }
};

TEST_P(PadTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<ngraph::PartialShape> inputShapes = {
    {1, 3, 6, 6},
    {4, 3, 6, 6},
    {-1, 3, 6, -1}
};

const std::pair<std::vector<uint64_t>, std::vector<uint64_t>> padsBySpatialDimensions = {
    {0, 0, 2, 1},
    {0, 0, 1, 2}
};

// test-cases with common logic for all modes
// (per-tensor & per-channel quantizations without subtracts, pads by spatial dimensions)
// and test-case without dequantization
namespace commonTestCases {
const std::vector<ngraph::PartialShape> commonInputShapes = {
    {4, 3, 6, 6},
    {-1, -1, -1, -1}
};

std::vector<ngraph::op::PadMode> allModes = {
    ngraph::op::PadMode::EDGE,
    ngraph::op::PadMode::REFLECT,
    ngraph::op::PadMode::SYMMETRIC,
    ngraph::op::PadMode::CONSTANT,
};

const std::vector<PadTransformationTestValues> deqWithoutSub = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {3.f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {3.f}}
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {3.f}}
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {3.f}}
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{}, {}, {}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{}, {}, {}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ngraph::element::f32,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {}, {{3.f, 1.f, 2.f}}}
        }
    },
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            ngraph::element::f32,
            {{}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ngraph::element::f32,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {}, {{3.f, 1.f, 2.f}}}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(commonInputShapes),
        ::testing::Values(padsBySpatialDimensions),
        ::testing::ValuesIn(allModes),
        ::testing::Values(0.f),
        ::testing::ValuesIn(deqWithoutSub)),
    PadTransformation::getTestCaseName);
} // namespace commonTestCases

// test-cases with common logic for "EDGE", "REFLECT", and "SYMMETRIC" modes:
// pads by spatial dimensions, dequantization with subtract
namespace dqWithSubtract {
std::vector<ngraph::op::PadMode> modesInWhichSubPropagated = {
    ngraph::op::PadMode::EDGE,
    ngraph::op::PadMode::REFLECT,
    ngraph::op::PadMode::SYMMETRIC,
};

const std::vector<PadTransformationTestValues> deqWithSub = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {3.f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {3.f}}
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {64.f}, {3.f}}
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::i8,
            {{ngraph::element::f32}, {64.f}, {3.f}}
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {{64.f, 32.f, 16.f}}, {{3.f, 1.f, 2.f}}}
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::i8,
            {{ngraph::element::f32}, {{64.f, 32.f, 16.f}}, {{3.f, 1.f, 2.f}}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{3.f, 1.f, 2.f}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{3.f, 1.f, 2.f}}}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsBySpatialDimensions),
        ::testing::ValuesIn(modesInWhichSubPropagated),
        ::testing::Values(0.f),
        ::testing::ValuesIn(deqWithSub)),
    PadTransformation::getTestCaseName);
} // namespace dqWithSubtract

// dequantization with subtract and "CONSTANT" mode, also dequantization and padding by the same dimension
namespace testCasesForConstantMode {
const std::vector<PadTransformationTestValues> testValuesForConstantMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {64.f}, {3.f}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {64.f}, {3.f}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {{64.f, 32.f, 16.f}}, {{3.f, 1.f, 2.f}}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {{64.f, 32.f, 16.f}}, {{3.f, 1.f, 2.f}}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{3.f, 1.f, 2.f}}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{128.f, 64.f, 32.f}}, {{3.f, 1.f, 2.f}}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ngraph::element::f32, {1, 1, 6, 1}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{1.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 1.f}, ngraph::element::f32, {1, 1, 9, 1}}}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsBySpatialDimensions),
        ::testing::Values(ngraph::op::PadMode::CONSTANT),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForConstantMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForConstantMode

// dequantization with "CONSTANT" mode (non zero value) and non unique pad dimension: dequantization isn't propagated
namespace testCasesForConstantModeWithNonZeroValues {
const std::vector<PadTransformationTestValues> testValuesForConstantMode2 = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {3.f}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {3.f}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {{3.f, 1.f, 2.f}}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{3.f, 1.f, 2.f}}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsBySpatialDimensions),
        ::testing::Values(ngraph::op::PadMode::CONSTANT),
        ::testing::Values(1.f),
        ::testing::ValuesIn(testValuesForConstantMode2)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForConstantModeWithNonZeroValues

namespace testCasesForConstantModeAndUniquePadDimension {
const std::pair<std::vector<uint64_t>, std::vector<uint64_t>> padsByUniqueDimension = {
    {0, 0, 2, 0},
    {0, 0, 1, 0}
};

const std::vector<PadTransformationTestValues> testValuesForConstantMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{64.f, 64.f, 64.f, 32.f, 32.f, 32.f}, ngraph::element::f32, {1, 1, 6, 1}},
                {{3.f, 3.f, 3.f, 2.f, 2.f, 2.f}, ngraph::element::f32, {1, 1, 6, 1}}
            }
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{0.f, 0.f, 64.f, 64.f, 64.f, 32.f, 32.f, 32.f, 0.f}, ngraph::element::f32, {1, 1, 9, 1}},
                {{1.f, 1.f, 3.f, 3.f, 3.f, 2.f, 2.f, 2.f, 1.f}, ngraph::element::f32, {1, 1, 9, 1}}
            }
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {64.f},
                {3.f}
            }
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{0.f, 0.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 0.f}, ngraph::element::f32, {1, 1, 9, 1}},
                {3.f}
            }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsByUniqueDimension),
        ::testing::Values(ngraph::op::PadMode::CONSTANT),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForConstantMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForConstantModeAndUniquePadDimension

namespace testCasesForNonZeroConstantModeAndUniquePadDimension {
const std::pair<std::vector<uint64_t>, std::vector<uint64_t>> padsByUniqueDimension = {
    {0, 0, 2, 0},
    {0, 0, 1, 0}
};

const std::vector<PadTransformationTestValues> testValuesForConstantMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {},
                {{3.f, 3.f, 3.f, 2.f, 2.f, 2.f}, ngraph::element::f32, {1, 1, 6, 1}}
            }
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {},
                {{1.f, 1.f, 3.f, 3.f, 3.f, 2.f, 2.f, 2.f, 1.f}, ngraph::element::f32, {1, 1, 9, 1}}
            }
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {64.f},
                {3.f}
            }
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{0.f, 0.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 0.f}, ngraph::element::f32, {1, 1, 9, 1}},
                {{1.f, 1.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 1.f}, ngraph::element::f32, {1, 1, 9, 1}}
            }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsByUniqueDimension),
        ::testing::Values(ngraph::op::PadMode::CONSTANT),
        ::testing::Values(2.f),
        ::testing::ValuesIn(testValuesForConstantMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForNonZeroConstantModeAndUniquePadDimension

namespace testCasesForEdgeMode {
const std::vector<PadTransformationTestValues> testValuesForEdgeMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ngraph::element::f32, {1, 1, 6, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ngraph::element::f32, {1, 1, 6, 1}}
            }
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{1.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 6.f}, ngraph::element::f32, {1, 1, 9, 1}},
                {{1.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 6.f}, ngraph::element::f32, {1, 1, 9, 1}}
            }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsBySpatialDimensions),
        ::testing::Values(ngraph::op::PadMode::EDGE),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForEdgeMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForEdgeMode

namespace testCasesForReflectMode {
const std::vector<PadTransformationTestValues> testValuesForReflectMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ngraph::element::f32, {1, 1, 6, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ngraph::element::f32, {1, 1, 6, 1}}
            }
        },
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ngraph::element::f32, {1, 1, 6, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ngraph::element::f32, {1, 1, 6, 1}}
            },
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsBySpatialDimensions),
        ::testing::Values(ngraph::op::PadMode::REFLECT),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForReflectMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForReflectMode

namespace testCasesForSymetricMode {
const std::vector<PadTransformationTestValues> testValuesForSymetricMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ngraph::element::f32, {1, 1, 6, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ngraph::element::f32, {1, 1, 6, 1}}
            }
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{2.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 6.f}, ngraph::element::f32, {1, 1, 9, 1}},
                {{2.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 6.f}, ngraph::element::f32, {1, 1, 9, 1}}
            }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsBySpatialDimensions),
        ::testing::Values(ngraph::op::PadMode::SYMMETRIC),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForSymetricMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForSymetricMode

namespace testCasesWithDynamicRank {
const std::vector<ngraph::PartialShape> inputShapesWithDynamicRank = {
    ngraph::PartialShape::dynamic()
};

std::vector<ngraph::op::PadMode> allModes = {
    ngraph::op::PadMode::EDGE,
    ngraph::op::PadMode::REFLECT,
    ngraph::op::PadMode::SYMMETRIC,
    ngraph::op::PadMode::CONSTANT,
};

const std::vector<PadTransformationTestValues> testValuesForDynamicRank = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {3.f}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {3.f}},
            ngraph::element::f32,
            {{}, {}, {}},
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {{3.f, 1.f, 2.f}}},
            ngraph::element::f32,
            {{}, {}, {}},
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesWithDynamicRank),
        ::testing::Values(padsBySpatialDimensions),
        ::testing::ValuesIn(allModes),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForDynamicRank)),
    PadTransformation::getTestCaseName);
} // namespace testCasesWithDynamicRank
} // namespace
