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
#include "low_precision/relu.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/relu.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ov;
using namespace ov::pass;

class ReluTransformationTestValues {
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

typedef std::tuple<
    ov::PartialShape,
    ReluTransformationTestValues> ReluTransformationParams;

class ReluTransformation : public LayerTransformation, public testing::WithParamInterface<ReluTransformationParams> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto testValues = std::get<1>(GetParam());

        actualFunction = ov::builder::subgraph::ReluFunction::getOriginal(
            inputShape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ov::pass::low_precision::ReluTransformation, ov::op::v0::PRelu>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::ReluFunction::getReference(
            inputShape,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ReluTransformationParams> obj) {
        const auto inputShape = std::get<0>(obj.param);
        const auto testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
            toString(testValues.params) << "_" <<
            inputShape << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> actualFunction;
    std::shared_ptr<ov::Model> referenceFunction;
};

TEST_P(ReluTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ov::PartialShape> shapes = {
    { 1, 3, 16, 16 },
    { -1, -1, -1, -1 },
};

const std::vector<ReluTransformationTestValues> testValues = {
    // U8: no subtract
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {0.1f}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {{ov::element::f32}, {}, {0.1f}}
        }
    },
    // U8: no subtract
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, 0.2f, 0.3f}}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, 0.2f, 0.3f}}}
        }
    },
    // U8: no subtract
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, -0.2f, 0.3f}}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, -0.2f, 0.3f}}},
            ov::element::f32,
            {{}, {}, {}}
        }
    },
    // I8: no subtract
    {
        LayerTransformation::createParamsI8I8(),
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {0.1f}}
        },
        {
            ov::element::i8,
            {{}, {}, {}},
            ov::element::i8,
            {{ov::element::f32}, {}, {0.1f}}
        }
    },
    // U8: with subtract value
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, { 128 }, {0.1f}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, { 128 }, {0.1f}},
            ov::element::f32,
            {{}, {}, {}}
        }
    },
    // I8: with subtract value
    {
        LayerTransformation::createParamsI8I8().setSupportAsymmetricQuantization(true),
        {
            ov::element::i8,
            {{ov::element::f32}, { 127 }, {0.1f}}
        },
        {
            ov::element::i8,
            {{ov::element::f32}, { 127 }, {0.1f}},
            ov::element::f32,
            {{}, {}, {}}
        }
    },
    // I8: with subtract value
    {
        LayerTransformation::createParamsI8I8().setSupportAsymmetricQuantization(false),
        {
            ov::element::i8,
            {{ov::element::f32}, { 127 }, {0.1f}}
        },
        {
            ov::element::i8,
            {{ov::element::f32}, { 127 }, {0.1f}},
            ov::element::f32,
            {{}, {}, {}}
        }
    },
    // U8: empty
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {}
        }
    },
    // FP32: empty
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::f32,
            {}
        },
        {
            ov::element::f32,
            {},
            ov::element::f32,
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ReluTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ReluTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ov::PartialShape> shapesWithDynamicRank = {
    PartialShape::dynamic()
};

const std::vector<ReluTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {0.1f}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {0.1f}},
            ov::element::f32,
            {{}, {}, {}}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ReluTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapesWithDynamicRank),
        ::testing::ValuesIn(testValues)),
    ReluTransformation::getTestCaseName);
} // namespace testValues2
} // namespace
