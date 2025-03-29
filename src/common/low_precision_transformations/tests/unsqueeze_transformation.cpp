// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"
#include "low_precision/unsqueeze.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/unsqueeze.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;

using ov::builder::subgraph::UnsqueezeFunction;

class UnsqueezeTransformationTestValues {
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

    ov::PartialShape inputShape;
    std::vector<float> axes;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

class UnsqueezeTransformation : public LayerTransformation, public testing::WithParamInterface<UnsqueezeTransformationTestValues> {
public:
    void SetUp() override {
        const UnsqueezeTransformationTestValues testValues = GetParam();

        actualFunction = ov::builder::subgraph::UnsqueezeFunction::getOriginal(
            testValues.inputShape,
            testValues.axes,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::UnsqueezeTransformation, ov::op::v0::Unsqueeze>(testValues.params);

        transform.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::UnsqueezeFunction::getReference(
            testValues.inputShape,
            testValues.axes,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<UnsqueezeTransformationTestValues> obj) {
        const UnsqueezeTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.inputShape << "_" <<
            testValues.axes << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore;

        return result.str();
    }
};

TEST_P(UnsqueezeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<UnsqueezeTransformationTestValues> testValues = {
    {
        { 1, 1, 16, 16 }, // Input shape
        { 0.0f }, // Unsqueeze axes
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false), // Layer params

        /* Actual */
        {
            ov::element::u8, // Precision before dequantization
            /* Dequantization */
            {
                {ov::element::f32}, // Convert
                {-0.32f}, // Subtract
                {0.45f} // Multiply
            }
        },
        /* Expected */
        {
            ov::element::u8, // Precision before dequantization
            /* Dequantization before */
            {},
            ov::element::u8, // Precision after dequantization
            /* Dequantization after */
            {
                {ov::element::f32}, // Convert
                {-0.32f}, // Subtract
                {0.45f} // Multiply
            }
        }
    },
    {
        { 1, 3, 16, 16 }, // Input shape
        { 0.0f }, // Unsqueeze axes
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false), // Layer params

        /* Actual */
        {
            ov::element::u8, // Precision before dequantization
            /* Dequantization */
            {
                {ov::element::f32}, // Convert
                {{128.f, 64.f, 32.f}, ov::element::f32, {1, 3, 1, 1}}, // Subtract
                {{0.1f, 0.2f, 0.3f}, ov::element::f32, {1, 3, 1, 1}} // Multiply
            }
        },
        /* Expected */
        {
            ov::element::u8, // Precision before dequantization
            /* Dequantization before */
            {},
            ov::element::u8, // Precision after dequantization
            /* Dequantization after */
            {
                {ov::element::f32}, // Convert
                {{128.f, 64.f, 32.f}, ov::element::f32, {1, 1, 3, 1, 1}}, // Subtract
                {{0.1f, 0.2f, 0.3f}, ov::element::f32, {1, 1, 3, 1, 1}} // Multiply
            }
        }
    },
    {
        { Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic() }, // Input shape
        { 0.0f }, // Unsqueeze axes
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false), // Layer params

        /* Actual */
        {
            ov::element::u8, // Precision before dequantization
            /* Dequantization */
            {
                {ov::element::f32}, // Convert
                {{128.f, 64.f, 32.f}, ov::element::f32, {1, 3, 1, 1}}, // Subtract
                {{0.1f, 0.2f, 0.3f}, ov::element::f32, {1, 3, 1, 1}} // Multiply
            }
        },
        /* Expected */
        {
            ov::element::u8, // Precision before dequantization
            /* Dequantization before */
            {},
            ov::element::u8, // Precision after dequantization
            /* Dequantization after */
            {
                {ov::element::f32}, // Convert
                {{128.f, 64.f, 32.f}, ov::element::f32, {1, 1, 3, 1, 1}}, // Subtract
                {{0.1f, 0.2f, 0.3f}, ov::element::f32, {1, 1, 3, 1, 1}} // Multiply
            }
        }
    },
    {
        { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() }, // Input shape
        { 0.0f }, // Unsqueeze axes
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false), // Layer params

        /* Actual */
        {
            ov::element::u8, // Precision before dequantization
            /* Dequantization */
            {
                {ov::element::f32}, // Convert
                {-0.32f}, // Subtract
                {0.45f} // Multiply
            }
        },
        /* Expected */
        {
            ov::element::u8, // Precision before dequantization
            /* Dequantization before */
            {},
            ov::element::u8, // Precision after dequantization
            /* Dequantization after */
            {
                {ov::element::f32}, // Convert
                {-0.32f}, // Subtract
                {0.45f} // Multiply
            }
        }
    },
    {
        { 1, 1, 1000 }, // Input shape
        { 1.0f }, // Unsqueeze axes
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true), // Layer params

        /* Actual */
        {
            ov::element::i8, // Precision before dequantization
            /* Dequantization */
            {
                {ov::element::f32}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        },
        /* Expected */
        {
            ov::element::i8, // Precision before dequantization
            /* Dequantization before */
            {},
            ov::element::i8, // Precision after dequantization
            /* Dequantization after */
            {
                {ov::element::f32}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        }
    },
    {
        { 1, 1, 1000 }, // Input shape
        { 2.0f }, // Unqueeze axes
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false), // Layer params

        /* Actual */
        {
            ov::element::f32, // Precision before dequantization
            /* Dequantization */
            {
                {}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        },
        /* Expected */
        {
            ov::element::f32, // Precision before dequantization
            /* Dequantization before */
            {},
            ov::element::f32, // Precision after dequantization
            /* Dequantization after */
            {
                {}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        }
    },
    {
        { 1, 1, 1000, 1000 }, // Input shape
        { 0.0f }, // Unsqueeze axes
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true), // Layer params

        /* Actual */
        {
            ov::element::f32, // Precision before dequantization
            /* Dequantization */
            {
                {}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        },
        /* Expected */
        {
            ov::element::f32, // Precision before dequantization
            /* Dequantization before */
            {},
            ov::element::f32, // Precision after dequantization
            /* Dequantization after */
            {
                {}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        }
    },
    {
        { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() }, // Input shape
        { 0.0f }, // Unsqueeze axes
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true), // Layer params

        /* Actual */
        {
            ov::element::f32, // Precision before dequantization
            /* Dequantization */
            {
                {}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        },
        /* Expected */
        {
            ov::element::f32, // Precision before dequantization
            /* Dequantization before */
            {},
            ov::element::f32, // Precision after dequantization
            /* Dequantization after */
            {
                {}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        }
    },
    {
        PartialShape::dynamic(), // Input shape
        { 0.0f }, // Squeeze axes
        LayerTransformation::createParamsU8I8(), // Layer params

        /* Actual */
        {
            ov::element::u8, // Precision before dequantization
            /* Dequantization */
            {
                {ov::element::f32}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        },
        /* Expected */
        {
            ov::element::u8, // Precision before dequantization
            /* Dequantization before */
            {
                {ov::element::f32}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            },
            ov::element::f32, // Precision after dequantization
            /* Dequantization after */
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    UnsqueezeTransformation,
    ::testing::ValuesIn(testValues),
    UnsqueezeTransformation::getTestCaseName);
} // namespace
