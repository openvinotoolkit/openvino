// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/fake_quantize.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include "lpt_ngraph_functions/common/add.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/fuse_fake_quantize_function.hpp"

#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class FuseFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeAdd;
        ngraph::builder::subgraph::Add add;
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        ngraph::element::Type precisionAfterDequantization;
        ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeAdd;
        ngraph::builder::subgraph::Add add;
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        ngraph::element::Type precisionAfterDequantization;
        ngraph::element::Type precisionFakeQuantizeOnData;
        ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
    };

    ngraph::PartialShape inputShape;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

class FuseFakeQuantizeTransformation : public LayerTransformation, public testing::WithParamInterface<FuseFakeQuantizeTransformationTestValues> {
public:
    void SetUp() override {
        const FuseFakeQuantizeTransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::FuseFakeQuantizeFunction::getOriginal(
            testValues.inputShape,
            testValues.actual.precisionBeforeAdd,
            testValues.actual.add,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization,
            testValues.actual.precisionAfterDequantization,
            testValues.actual.precisionAfterDequantization,
            testValues.actual.fakeQuantizeOnData);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transformer.transform(actualFunction);

        transformer.add<ngraph::pass::low_precision::FakeQuantizeTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FuseFakeQuantizeFunction::getReference(
            testValues.inputShape,
            testValues.expected.precisionBeforeAdd,
            testValues.expected.add,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantization,
            testValues.expected.precisionAfterDequantization,
            testValues.expected.precisionFakeQuantizeOnData,
            testValues.expected.fakeQuantizeOnData);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FuseFakeQuantizeTransformationTestValues> obj) {
        const FuseFakeQuantizeTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.inputShape << "_" <<
            testValues.params.updatePrecisions << "_" <<
            testValues.actual.precisionBeforeAdd << "_" <<
            testValues.actual.add.values.size() << "_" <<
            testValues.actual.add.outPrecision << "_" <<
            testValues.actual.add.constantShape << "_" <<
            testValues.actual.precisionBeforeDequantization <<
            testValues.actual.dequantization << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.fakeQuantizeOnData << "_" <<
            testValues.expected.dequantization << "_" <<
            testValues.expected.add.values.size() << "_" <<
            testValues.expected.add.outPrecision << "_" <<
            testValues.expected.add.constantShape;
        return result.str();
    }
};

TEST_P(FuseFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<FuseFakeQuantizeTransformationTestValues> testValues = {
    // 1) Multiply
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, { 0.01f } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {} },
            element::f32,
            element::f32,
            { 256ul, {}, { 0.f }, { 255.f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Multiply
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {{0.01f, 0.02f, 0.03f}} },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {} },
            element::f32,
            element::f32,
            { 256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
                { 0.f, 0.f, 0.f }, { 255.f, 127.5f, 85.f },
                { 0.f }, { 2.55f }
            }
        }
    },
    // 1) Per-channel multiply and dynamic shape
    {
        {Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {{0.01f, 0.02f, 0.03f}} },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {} },
            element::f32,
            element::f32,
            { 256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
                { 0.f, 0.f, 0.f }, { 255.f, 127.5f, 85.f },
                { 0.f }, { 2.55f }
            }
        }
    },
    // 1) Per-channel multiply and dynamic shape with dynamic channels
    {
        {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {{0.01f, 0.02f, 0.03f}} },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {{0.01f, 0.02f, 0.03f}} },
            element::f32,
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Multiply with different input and output shape
    {
        {128, 1},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, { {0.01f, 0.1f, 1.f}, ngraph::element::f32, {1, 3} } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, { {0.01f, 0.1f, 1.f}, ngraph::element::f32, {1, 3} } },
            element::f32,
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
    // Dynamic shape
    {
        {Dimension::dynamic(), Dimension::dynamic()},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, { {0.01f, 0.1f, 1.f}, ngraph::element::f32, {1, 3} } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, { {0.01f, 0.1f, 1.f}, ngraph::element::f32, {1, 3} } },
            element::f32,
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Multiply + 2) Add
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            { {128}, element::f32 },
            element::f32,
            { {}, {}, { 0.01f } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {} },
            element::f32,
            element::f32,
            { 256ul, {}, { -128.f }, { 127.f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Subtract + Multiply
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::f32,
            { {}, { -128 }, { 0.01f } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {} },
            element::f32,
            element::f32,
            { 256ul, {}, { -128.f }, { 127.f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Convert + Subtract + Multiply
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::u8,
            { {element::f32}, { -128 }, { 0.01f } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::u8,
            { {}, {}, {} },
            element::u8,
            element::f32,
            { 256ul, {}, { -128.f }, { 127.f }, { 0.f }, { 2.55f } }
        }
    },
    // 1) Convert + Subtract + Multiply 2) Add
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            { {127}, element::f32 },
            element::f32,
            { {element::f32}, { -128 }, { 0.01f } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {} },
            element::f32,
            element::f32,
            { 256ul, {}, { -255.f }, { 0.f }, { 0.f }, { 2.55f } }
        }
    },
    // Dynamic shape
    {
        {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            { {127}, element::f32 },
            element::f32,
            { {element::f32}, { -128 }, { 0.01f } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, {}, {} },
            element::f32,
            element::f32,
            { 256ul, {}, { -255.f }, { 0.f }, { 0.f }, { 2.55f } }
        }
    },
    // Dynamic rank
    {
        PartialShape::dynamic(),
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            { {127}, element::f32 },
            element::f32,
            { {element::f32}, { -128 }, { 0.01f } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            { {127}, element::f32 },
            element::f32,
            { {element::f32}, { -128 }, { 0.01f } },
            element::f32,
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
    // negative multiply
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::f32,
            { {}, { -128 }, { -0.01f } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::f32,
            { {}, { -128 }, { -0.01f } },
            element::f32,
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
    // issue #40611 for FP32
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { },
            { },
            ngraph::element::i32,
            { {ngraph::element::f32}, {}, {} },
            ngraph::element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            { },
            { },
            ngraph::element::i32,
            { {ngraph::element::f32}, {}, {} },
            element::f32,
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
    // issue #40611 for FP16
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            { },
            { },
            ngraph::element::i32,
            { {ngraph::element::f16}, {}, {} },
            ngraph::element::f16,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            { },
            { },
            ngraph::element::i32,
            { {ngraph::element::f16}, {}, {} },
            element::f16,
            element::f16,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
    // multiply by zero
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::u8,
            { {element::f32}, { {-128, -128, -128} }, { {0.01f, 0.f, 0.01f} } },
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        },
        {
            element::f32,
            {},
            element::u8,
            { {element::f32}, { {-128, -128, -128} }, { {0.01f, 0.f, 0.01f} } },
            element::f32,
            element::f32,
            { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FuseFakeQuantizeTransformation,
    ::testing::ValuesIn(testValues),
    FuseFakeQuantizeTransformation::getTestCaseName);

} // namespace
