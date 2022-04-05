// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <gtest/gtest.h>

#include <transformations/init_node_info.hpp>
#include <low_precision/broadcast.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/broadcast_function.hpp"
#include "simple_low_precision_transformer.hpp"


namespace {
using namespace testing;
using namespace ngraph::pass;
using namespace ngraph;

class BroadcastTransformationTestValues {
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
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple <
    ngraph::PartialShape,
    BroadcastTransformationTestValues,
    size_t, // opset
    std::string // broadcast mode
> BroadcastTransformationParams;

class BroadcastTransformation : public LayerTransformation, public testing::WithParamInterface<BroadcastTransformationParams> {
public:
    void SetUp() override {
        const ngraph::PartialShape inputShape = std::get<0>(GetParam());
        const BroadcastTransformationTestValues testValues = std::get<1>(GetParam());
        const size_t opset = std::get<2>(GetParam());
        const std::string mode = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::BroadcastFunction::getOriginal(
                inputShape,
                testValues.actual.precisionBeforeDequantization,
                testValues.actual.dequantization,
                opset,
                mode);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::BroadcastTransformation, ngraph::opset1::Clamp>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::BroadcastFunction::getReference(
                inputShape,
                testValues.expected.precisionBeforeDequantization,
                testValues.expected.dequantizationBefore,
                testValues.expected.dequantizationAfter,
                opset,
                mode);
    }

static std::string getTestCaseName(testing::TestParamInfo<BroadcastTransformationParams> obj) {
    const ngraph::PartialShape inputShape = std::get<0>(obj.param);
    const BroadcastTransformationTestValues testValues = std::get<1>(obj.param);
    const size_t opset = std::get<2>(obj.param);
    const std::string mode = std::get<3>(obj.param);

    std::ostringstream result;
    result << toString(testValues.params) << "_" <<
           inputShape << "_" <<
           testValues.actual.precisionBeforeDequantization << "_" <<
           testValues.actual.dequantization << "_" <<
           testValues.expected.dequantizationBefore << "_" <<
           opset << "_" << mode;
    return result.str();
}
};

TEST_P(BroadcastTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ngraph::PartialShape> inputShapes = {
    ngraph::PartialShape({ 1, 3, 50, 1 }),
    ngraph::PartialShape({ -1, -1, -1, 1 }),
};

const std::vector<size_t> opsets = {
    1,
    3
};

const std::vector<std::string> modes = {
    "numpy",
    "explicit",
    "bidirectional"
};

const std::vector<BroadcastTransformationTestValues> testValues = {
    // U8 per tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            element::u8,
            {{element::f32}, {128.f}, {3.f}}
        },
        {
            element::u8,
            {{}, {}, {}},
            {{element::f32}, {128.f}, {3.f}}
        }
    },
    // FP32 without convert
    {
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {{}, {128.f}, {3.f}}
        },
        {
            element::f32,
            {{}, {}, {}},
            {{}, {128.f}, {3.f}}
        }
    },
    // U8 without subtract
    {
        LayerTransformation::createParamsU8I8(),
        {
            element::u8,
            {{element::f32}, {}, {3.f}}
        },
        {
            element::u8,
            {{}, {}, {}},
            {{element::f32}, {}, {3.f}}
        }
    },
    // U8 per channel quantization with different values
    {
        LayerTransformation::createParamsU8I8(),
        {
            element::u8,
            {
                {element::f32},
                {{128.f, 0.f, 128.f / 2}},
                {{3.f, 1.f, 2.f}}
            }
        },
        {
            element::u8,
            {
                {element::f32},
                {{128.f, 0.f, 128.f / 2}},
                {{3.f, 1.f, 2.f}}
            },
            {{}, {}, {}}
        }
    },
    // U8 per channel quantization with the same values
    {
        LayerTransformation::createParamsU8I8(),
        {
            element::u8,
            {
                {element::f32},
                {{128.f, 128.f, 128.f}},
                {{3.f, 3.f, 3.f}}
            }
        },
        {
            element::u8,
            {{}, {}, {}},
            { {element::f32}, {128.f}, {3.f} },
        }
    },
    // without dequantization
    {
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {{}, {}, {}}
        },
        {
            element::f32,
            {{}, {}, {}},
            {{}, {}, {}}
        },
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    BroadcastTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(opsets),
        ::testing::ValuesIn(modes)),
    BroadcastTransformation::getTestCaseName);
} // namespace testValues1
} // namespace
