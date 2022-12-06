// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/gather.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/gather_function.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {
using namespace testing;
using namespace ngraph::pass;
using namespace ngraph;

class GatherTransformationTestValues {
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

    std::vector<int> gatherIndicesValues;
    std::vector<int> axis;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::PartialShape,
    GatherTransformationTestValues
> GatherTransformationParams;

class GatherTransformation : public LayerTransformation, public testing::WithParamInterface<GatherTransformationParams> {
public:
    void SetUp() override {
        const ngraph::PartialShape inputShape = std::get<0>(GetParam());
        const GatherTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::GatherFunction::getOriginal(
            inputShape,
            testValues.gatherIndicesValues,
            testValues.axis,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::GatherTransformation, ngraph::opset1::Gather>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::GatherFunction::getReference(
            inputShape,
            testValues.gatherIndicesValues,
            testValues.axis,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<GatherTransformationParams> obj) {
        const ngraph::PartialShape inputShape = std::get<0>(obj.param);
        const GatherTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result << "_" << 
            inputShape << "_" <<
            testValues.gatherIndicesValues << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore;
        return result.str();
    }
};

TEST_P(GatherTransformation, CompareFunctions) {
    InitNodeInfo().run_on_model(actualFunction);
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ngraph::PartialShape> inputShapes3D = {
    { 3, 3, 4 }
};

const std::vector<GatherTransformationTestValues> testValues = {
    // U8: per-tensor quantization
    {
        {0},
        {0},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                { {128}, ngraph::element::f32, {}, true, 1, ngraph::element::u8, true },
                {0.1f}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                { {128}, ngraph::element::f32, {}, true, 1, ngraph::element::u8, true },
                {0.1f}
            }
        }
    },
    // U8: per-tensor quantization
    {
        {0},
        {2},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {128}, {0.1f}}
        }
    },

    // U8: per-channel quantization with the same values
    {
        {1},
        {2},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {{128.f}, element::undefined, {1, 3, 1}, false, 1ul, element::u8, true},
                {{0.1}, ngraph::element::f32, {1, 3, 1}}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                { ngraph::element::f32 },
                {{128.f}, element::undefined, {}, false, 1ul, element::u8, true},
                {{0.1}, ngraph::element::f32, {}}
            }
        }
    },
    // U8: per-tensor quantization, gather channel dimension
    {
        {1},
        {2},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {128}, {0.1f}}
        }
    },
    // empty
    {
        {0},
        {0},
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
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    GatherTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes3D),
        ::testing::ValuesIn(testValues)),
    GatherTransformation::getTestCaseName);
} // namespace testValues1
} // namespace
