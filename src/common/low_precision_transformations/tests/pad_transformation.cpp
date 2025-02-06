// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <gtest/gtest.h>


#include "low_precision/pad.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/pad.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;

class PadTransformationTestValues {
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

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple <
    ov::PartialShape, // input Shape
    std::pair<std::vector<int64_t>, std::vector<int64_t>>, // pads begin, pads end
    ov::op::PadMode, // pads mode
    float, // pads value (used if mode == CONSTANT)
    PadTransformationTestValues> PadTransformationParams;

class PadTransformation : public LayerTransformation, public testing::WithParamInterface<PadTransformationParams> {
public:
    void SetUp() override {
        const ov::PartialShape inputShape = std::get<0>(GetParam());
        const auto pads = std::get<1>(GetParam());
        const ov::op::PadMode padsMode = std::get<2>(GetParam());
        const float padsValue = std::get<3>(GetParam());
        const PadTransformationTestValues testValues = std::get<4>(GetParam());

        const auto precisionAfterActualOp = testValues.actual.dequantization.convert.empty() ?
            testValues.actual.precisionBeforeDequantization : testValues.actual.dequantization.convert.outPrecision;

        actualFunction = ov::builder::subgraph::PadFunction::get(
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
        transformer.add<ov::pass::low_precision::PadTransformation, ov::op::v1::Pad>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::PadFunction::get(
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
        const ov::PartialShape inputShape = std::get<0>(obj.param);
        const auto pads = std::get<1>(obj.param);
        const ov::op::PadMode padsMode = std::get<2>(obj.param);
        const float padsValue = std::get<3>(obj.param);
        const PadTransformationTestValues testValues = std::get<4>(obj.param);

        std::ostringstream result;
        result << "mode_" << padsMode << "_";
        if (padsMode == ov::op::PadMode::CONSTANT) {
            result << "pad_value_{ " << padsValue << " }";
        }

        result << "_" <<
            toString(testValues.params) << "_" <<
            inputShape << "_" << pads.first << "_" << pads.second << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_expected_dequantizationBefore_" <<
            testValues.expected.dequantizationBefore << "_expected_dequantizationAfter_" <<
            testValues.expected.dequantizationAfter;

        return result.str();
    }
};

TEST_P(PadTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<ov::PartialShape> inputShapes = {
    {1, 3, 6, 6},
    {4, 3, 6, 6},
    {-1, 3, 6, -1}
};

const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> padsBySpatialDimensions = {
    {{0, 0, 2, 1}, {0, 0, 1, 2}},
    {{0, 0, -1, -1}, {0, 0, 2, 2}}
};

const std::pair<std::vector<int64_t>, std::vector<int64_t>> padsPositiveBySpatialDimensions = {
    {0, 0, 2, 1}, {0, 0, 1, 2}
};

const std::pair<std::vector<int64_t>, std::vector<int64_t>> padsNegativeBySpatialDimensions = {
    {0, 0, -2, -1}, {0, 0, -1, -2}
};

const std::pair<std::vector<int64_t>, std::vector<int64_t>> padsMixedBySpatialDimensions = {
    {0, 0, 2, -1}, {0, 0, -1, 2}
};


// test-cases with common logic for all modes
// (per-tensor & per-channel quantizations without subtracts, pads by spatial dimensions)
// and test-case without dequantization
namespace commonTestCases {
const std::vector<ov::PartialShape> commonInputShapes = {
    {4, 3, 6, 6},
    {-1, -1, -1, -1}
};

std::vector<ov::op::PadMode> allModes = {
    ov::op::PadMode::EDGE,
    ov::op::PadMode::REFLECT,
    ov::op::PadMode::SYMMETRIC,
    ov::op::PadMode::CONSTANT,
};

const std::vector<PadTransformationTestValues> deqWithoutSub = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {3.f}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {{ov::element::f32}, {}, {3.f}}
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {3.f}}
        },
        {
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            {{ov::element::f32}, {}, {3.f}}
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            {{ov::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {{ov::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{}, {}, {}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {{}, {}, {}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::f32,
            {{}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ov::element::f32,
            {{}, {}, {}},
            ov::element::f32,
            {{}, {}, {{3.f, 1.f, 2.f}}}
        }
    },
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            ov::element::f32,
            {{}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ov::element::f32,
            {{}, {}, {}},
            ov::element::f32,
            {{}, {}, {{3.f, 1.f, 2.f}}}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(commonInputShapes),
        ::testing::ValuesIn(padsBySpatialDimensions),
        ::testing::ValuesIn(allModes),
        ::testing::Values(0.f),
        ::testing::ValuesIn(deqWithoutSub)),
    PadTransformation::getTestCaseName);
} // namespace commonTestCases

// test-cases with common logic for "EDGE", "REFLECT", and "SYMMETRIC" modes:
// pads by spatial dimensions, dequantization with subtract
namespace dqWithSubtract {
std::vector<ov::op::PadMode> modesInWhichSubPropagated = {
    ov::op::PadMode::EDGE,
    ov::op::PadMode::REFLECT,
    ov::op::PadMode::SYMMETRIC,
};

const std::vector<PadTransformationTestValues> deqWithSub = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {3.f}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {3.f}}
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {64.f}, {3.f}}
        },
        {
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            {{ov::element::f32}, {64.f}, {3.f}}
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {{64.f, 32.f, 16.f}}, {{3.f, 1.f, 2.f}}}
        },
        {
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            {{ov::element::f32}, {{64.f, 32.f, 16.f}}, {{3.f, 1.f, 2.f}}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{3.f, 1.f, 2.f}}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{3.f, 1.f, 2.f}}}
        }
    },
    // int8 subtraction with Convert from u8 to fp32
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{128.f}, element::dynamic, {1, 3, 1, 1}, false, 1ul, element::u8, true},
                {3.f}
            }
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {
                {ov::element::f32},
                {{128.f}, element::dynamic, {1, 3, 1, 1}, false, 1ul, element::u8, true},
                {3.f}
            }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(padsBySpatialDimensions),
        ::testing::ValuesIn(modesInWhichSubPropagated),
        ::testing::Values(0.f),
        ::testing::ValuesIn(deqWithSub)),
    PadTransformation::getTestCaseName);
} // namespace dqWithSubtract

// dequantization with subtract and "CONSTANT" mode, also dequantization and padding by the same dimension
namespace testCasesForConstantMode {
namespace positive {
const std::vector<PadTransformationTestValues> testValuesForConstantMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {64.f}, {3.f}}
        },
        {
            ov::element::i8,
            {{ov::element::f32}, {64.f}, {3.f}},
            ov::element::f32,
            {{}, {}, {}}
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {{64.f, 32.f, 16.f}}, {{3.f, 1.f, 2.f}}}
        },
        {
            ov::element::i8,
            {{ov::element::f32}, {{64.f, 32.f, 16.f}}, {{3.f, 1.f, 2.f}}},
            ov::element::f32,
            {{}, {}, {}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{3.f, 1.f, 2.f}}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {{128.f, 64.f, 32.f}}, {{3.f, 1.f, 2.f}}},
            ov::element::f32,
            {{}, {}, {}}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsPositiveBySpatialDimensions),
        ::testing::Values(ov::op::PadMode::CONSTANT),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForConstantMode)),
    PadTransformation::getTestCaseName);
} // namespace positive

namespace negative {
    const std::vector<PadTransformationTestValues> testValuesForConstantMode = {
        {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {{ov::element::f32}, {}, {{3.f, 4.f, 5.f}, ov::element::f32, {1, 1, 3, 1}}}
        }
        }
    };

    INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsNegativeBySpatialDimensions),
        ::testing::Values(ov::op::PadMode::CONSTANT),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForConstantMode)),
    PadTransformation::getTestCaseName);
}

namespace mixed {
    const std::vector<PadTransformationTestValues> testValuesForConstantMode = {
        {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {{ov::element::f32}, {}, {{1.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f}, ov::element::f32, {1, 1, 7, 1}}}
        }
        }
    };

    INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsMixedBySpatialDimensions),
        ::testing::Values(ov::op::PadMode::CONSTANT),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForConstantMode)),
    PadTransformation::getTestCaseName);
}

} // namespace testCasesForConstantMode

// dequantization with "CONSTANT" mode (non zero value) and non unique pad dimension: dequantization isn't propagated
namespace testCasesForConstantModeWithNonZeroValues {
const std::vector<PadTransformationTestValues> testValuesForConstantMode2 = {
     {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {3.f}}
        },
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {3.f}},
            ov::element::f32,
            {{}, {}, {}}
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{3.f, 1.f, 2.f}}},
            ov::element::f32,
            {{}, {}, {}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{3.f, 1.f, 2.f}}},
            ov::element::f32,
            {{}, {}, {}}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(padsBySpatialDimensions),
        ::testing::Values(ov::op::PadMode::CONSTANT),
        ::testing::Values(1.f),
        ::testing::ValuesIn(testValuesForConstantMode2)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForConstantModeWithNonZeroValues

namespace testCasesForConstantModeAndUniquePadDimension {
namespace positive {
const std::pair<std::vector<int64_t>, std::vector<int64_t>> padsPositiveByUniqueDimension = {
    {0, 0, 2, 0}, {0, 0, 1, 0}
};

const std::vector<PadTransformationTestValues> testValuesForConstantMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {{64.f, 64.f, 64.f, 32.f, 32.f, 32.f}, ov::element::f32, {1, 1, 6, 1}},
                {{3.f, 3.f, 3.f, 2.f, 2.f, 2.f}, ov::element::f32, {1, 1, 6, 1}}
            }
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {
                {ov::element::f32},
                {{0.f, 0.f, 64.f, 64.f, 64.f, 32.f, 32.f, 32.f, 0.f}, ov::element::f32, {1, 1, 9, 1}},
                {{1.f, 1.f, 3.f, 3.f, 3.f, 2.f, 2.f, 2.f, 1.f}, ov::element::f32, {1, 1, 9, 1}}
            }
        }
    }
    ,
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {64.f},
                {3.f}
            }
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {
                {ov::element::f32},
                {{0.f, 0.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 0.f}, ov::element::f32, {1, 1, 9, 1}},
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
        ::testing::Values(padsPositiveByUniqueDimension),
        ::testing::Values(ov::op::PadMode::CONSTANT),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForConstantMode)),
    PadTransformation::getTestCaseName);
} // namespace positive

namespace negative {
const std::pair<std::vector<int64_t>, std::vector<int64_t>> padsNegativeByUniqueDimension = {
    {0, 0, -2, 0}, {0, 0, -1, 0}
};

const std::vector<PadTransformationTestValues> testValuesForConstantMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {{64.f, 64.f, 64.f, 32.f, 32.f, 32.f}, ov::element::f32, {1, 1, 6, 1}},
                {{3.f, 3.f, 3.f, 2.f, 2.f, 2.f}, ov::element::f32, {1, 1, 6, 1}}
            }
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {
                {ov::element::f32},
                {{64.f, 32.f, 32.f}, ov::element::f32, {1, 1, 3, 1}},
                {{3.f, 2.f, 2.f}, ov::element::f32, {1, 1, 3, 1}}
            }
        }
    }
    ,
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {64.f},
                {3.f}
            }
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {
                {ov::element::f32},
                {64.f},
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
        ::testing::Values(padsNegativeByUniqueDimension),
        ::testing::Values(ov::op::PadMode::CONSTANT),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForConstantMode)),
    PadTransformation::getTestCaseName);
} // namespace negative

namespace mixed {
const std::pair<std::vector<int64_t>, std::vector<int64_t>> padsMixedByUniqueDimension = {
    {0, 0, 2, 0}, {0, 0, -1, 0}
};

const std::vector<PadTransformationTestValues> testValuesForConstantMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {{64.f, 64.f, 64.f, 32.f, 32.f, 32.f}, ov::element::f32, {1, 1, 6, 1}},
                {{3.f, 3.f, 3.f, 2.f, 2.f, 2.f}, ov::element::f32, {1, 1, 6, 1}}
            }
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {
                {ov::element::f32},
                {{0.f, 0.f, 64.f, 64.f, 64.f, 32.f, 32.f}, ov::element::f32, {1, 1, 7, 1}},
                {{1.f, 1.f, 3.f, 3.f, 3.f, 2.f, 2.f}, ov::element::f32, {1, 1, 7, 1}}
            }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsMixedByUniqueDimension),
        ::testing::Values(ov::op::PadMode::CONSTANT),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForConstantMode)),
    PadTransformation::getTestCaseName);
} // namespace mixed

} // namespace testCasesForConstantModeAndUniquePadDimension

namespace testCasesForNonZeroConstantModeAndUniquePadDimension {
const std::pair<std::vector<int64_t>, std::vector<int64_t>> padsByUniqueDimension = {
    {0, 0, 2, 0},
    {0, 0, 1, 0}
};

const std::vector<PadTransformationTestValues> testValuesForConstantMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {},
                {{3.f, 3.f, 3.f, 2.f, 2.f, 2.f}, ov::element::f32, {1, 1, 6, 1}}
            }
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {
                {ov::element::f32},
                {},
                {{1.f, 1.f, 3.f, 3.f, 3.f, 2.f, 2.f, 2.f, 1.f}, ov::element::f32, {1, 1, 9, 1}}
            }
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {64.f},
                {3.f}
            }
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {
                {ov::element::f32},
                {{0.f, 0.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 0.f}, ov::element::f32, {1, 1, 9, 1}},
                {{1.f, 1.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 1.f}, ov::element::f32, {1, 1, 9, 1}}
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
        ::testing::Values(ov::op::PadMode::CONSTANT),
        ::testing::Values(2.f),
        ::testing::ValuesIn(testValuesForConstantMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForNonZeroConstantModeAndUniquePadDimension

namespace testCasesForEdgeModePostitive {
const std::vector<PadTransformationTestValues> testValuesForEdgeMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}}
            }
        },
        {
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            {
                {ov::element::f32},
                {{1.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 6.f}, ov::element::f32, {1, 1, 9, 1}},
                {{1.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 6.f}, ov::element::f32, {1, 1, 9, 1}}
            }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsPositiveBySpatialDimensions),
        ::testing::Values(ov::op::PadMode::EDGE),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForEdgeMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForEdgeModePostitive

namespace testCasesForEdgeModeNegative {
const std::vector<PadTransformationTestValues> testValuesForEdgeMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}}
            }
        },
        {
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            {
                {ov::element::f32},
                {{3.f, 4.f, 5.f}, ov::element::f32, {1, 1, 3, 1}},
                {{3.f, 4.f, 5.f}, ov::element::f32, {1, 1, 3, 1}}
            }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsNegativeBySpatialDimensions),
        ::testing::Values(ov::op::PadMode::EDGE),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForEdgeMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForEdgeModeNegative

namespace testCasesForEdgeModeMixed {
const std::vector<PadTransformationTestValues> testValuesForEdgeMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}}
            }
        },
        {
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            {
                {ov::element::f32},
                {{1.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f}, ov::element::f32, {1, 1, 7, 1}},
                {{1.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f}, ov::element::f32, {1, 1, 7, 1}}
            }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsMixedBySpatialDimensions),
        ::testing::Values(ov::op::PadMode::EDGE),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForEdgeMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForEdgeModeMixed

namespace testCasesForReflectMode {
const std::vector<PadTransformationTestValues> testValuesForReflectMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}}
            }
        },
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}}
            },
            ov::element::f32,
            {{}, {}, {}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(padsBySpatialDimensions),
        ::testing::Values(ov::op::PadMode::REFLECT),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForReflectMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForReflectMode


namespace testCasesForSymetricModePositive {
const std::vector<PadTransformationTestValues> testValuesForSymetricMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}}
            }
        },

        {
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            {
                {ov::element::f32},
                {{2.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 6.f}, ov::element::f32, {1, 1, 9, 1}},
                {{2.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 6.f}, ov::element::f32, {1, 1, 9, 1}}
            }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsPositiveBySpatialDimensions),
        ::testing::Values(ov::op::PadMode::SYMMETRIC),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForSymetricMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForSymetricModePositive

namespace testCasesForSymetricModeNegative {
const std::vector<PadTransformationTestValues> testValuesForSymetricMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}}
            }
        },

        {
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            {
                {ov::element::f32},
                {{3.f, 4.f, 5.f}, ov::element::f32, {1, 1, 3, 1}},
                {{3.f, 4.f, 5.f}, ov::element::f32, {1, 1, 3, 1}}
            }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsNegativeBySpatialDimensions),
        ::testing::Values(ov::op::PadMode::SYMMETRIC),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForSymetricMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForSymetricModeNegative

namespace testCasesForSymetricModeMixed {
const std::vector<PadTransformationTestValues> testValuesForSymetricMode = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}},
                {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, ov::element::f32, {1, 1, 6, 1}}
            }
        },

        {
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            {
                {ov::element::f32},
                {{2.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f}, ov::element::f32, {1, 1, 7, 1}},
                {{2.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f}, ov::element::f32, {1, 1, 7, 1}}
            }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(padsMixedBySpatialDimensions),
        ::testing::Values(ov::op::PadMode::SYMMETRIC),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForSymetricMode)),
    PadTransformation::getTestCaseName);
} // namespace testCasesForSymetricModeMixed

namespace testCasesWithDynamicRank {
const std::vector<ov::PartialShape> inputShapesWithDynamicRank = {
    ov::PartialShape::dynamic()
};

std::vector<ov::op::PadMode> allModes = {
    ov::op::PadMode::EDGE,
    ov::op::PadMode::REFLECT,
    ov::op::PadMode::SYMMETRIC,
    ov::op::PadMode::CONSTANT,
};

const std::vector<PadTransformationTestValues> testValuesForDynamicRank = {
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {3.f}}
        },
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {3.f}},
            ov::element::f32,
            {{}, {}, {}},
        }
    },
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{3.f, 1.f, 2.f}}}
        },
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{3.f, 1.f, 2.f}}},
            ov::element::f32,
            {{}, {}, {}},
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    PadTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesWithDynamicRank),
        ::testing::ValuesIn(padsBySpatialDimensions),
        ::testing::ValuesIn(allModes),
        ::testing::Values(0.f),
        ::testing::ValuesIn(testValuesForDynamicRank)),
    PadTransformation::getTestCaseName);
} // namespace testCasesWithDynamicRank
} // namespace
