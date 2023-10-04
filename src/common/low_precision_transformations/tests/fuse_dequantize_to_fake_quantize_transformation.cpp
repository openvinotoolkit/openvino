// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/fake_quantize.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include <memory>
#include <sstream>
#include <string>
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/common/add.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/fuse_fake_quantize.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ov;
using namespace ov::pass;

class FuseDequantizeToFakeQuantizeTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBeforeAdd;
        ngraph::builder::subgraph::Add add;
        ov::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        ov::element::Type precisionAfterDequantization;
        ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
    };

    class Expected {
    public:
        ov::element::Type precisionBeforeAdd;
        ngraph::builder::subgraph::Add add;
        ov::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        ov::element::Type precisionAfterDequantization;
        ov::element::Type precisionFakeQuantizeOnData;
        ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantizeOnData;
    };

    ov::PartialShape inputShape;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

class FuseDequantizeToFakeQuantizeTransformation
    : public LayerTransformation,
      public testing::WithParamInterface<FuseDequantizeToFakeQuantizeTransformationTestValues> {
public:
    void SetUp() override {
        const FuseDequantizeToFakeQuantizeTransformationTestValues testValues = GetParam();

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
        transformer.add<ov::pass::low_precision::FakeQuantizeDecompositionTransformation, ov::op::v0::FakeQuantize>(
            testValues.params);
        transformer.transform(actualFunction);

        transformer.add<ov::pass::low_precision::FakeQuantizeTransformation, ov::op::v0::FakeQuantize>(
            testValues.params);
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

    static std::string getTestCaseName(
        testing::TestParamInfo<FuseDequantizeToFakeQuantizeTransformationTestValues> obj) {
        const FuseDequantizeToFakeQuantizeTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result << testValues.inputShape << "_" << testValues.params.updatePrecisions << "_"
               << testValues.actual.precisionBeforeAdd << "_" << testValues.actual.add.values.size() << "_"
               << testValues.actual.add.outPrecision << "_" << testValues.actual.add.constantShape << "_"
               << testValues.actual.precisionBeforeDequantization << testValues.actual.dequantization << "_"
               << testValues.actual.precisionBeforeDequantization << "_" << testValues.actual.fakeQuantizeOnData << "_"
               << testValues.expected.dequantization << "_" << testValues.expected.add.values.size() << "_"
               << testValues.expected.add.outPrecision << "_" << testValues.expected.add.constantShape;
        return result.str();
    }
};

TEST_P(FuseDequantizeToFakeQuantizeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

// clang-format off
const std::vector<FuseDequantizeToFakeQuantizeTransformationTestValues> testValues = {
    // Convert: U8 -> FP32, updatePrecisions = true
    {
        {1, 3, 16, 16},
        TestTransformationParams(true, {ov::element::u8}, {ov::element::i8}),
        {
            element::f32,
            {},
            element::u8,
            {{element::f32}, {}, {}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::u8,
            {{}, {}, {}},
            element::f32,
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, element::u8}
        }
    },
    // Convert: U8 -> FP32, updatePrecisions = false
    {
        {1, 3, 16, 16},
        TestTransformationParams(false, {ov::element::u8}, {ov::element::i8}),
        {
            element::f32,
            {},
            element::u8,
            {{element::f32}, {}, {}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::u8,
            {{element::f32}, {}, {}},
            element::f32,
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f}, element::f32}
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
            {{}, {}, {0.01f}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::f32,
            {{}, {}, {}},
            element::f32,
            element::f32,
            {256ul, {}, {0.f}, {255.f}, {0.f}, {2.55f}}
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
            {{}, {}, {{0.01f, 0.02f, 0.03f}}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::f32,
            {{}, {}, {}},
            element::f32,
            element::f32,
            {256ul, {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}}, {0.f, 0.f, 0.f}, {255.f, 127.5f, 85.f}, {0.f}, {2.55f}}
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
            {{}, {}, {{0.01f, 0.02f, 0.03f}}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::f32,
            {{}, {}, {}},
            element::f32,
            element::f32,
            {256ul, {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}}, {0.f, 0.f, 0.f}, {255.f, 127.5f, 85.f}, {0.f}, {2.55f}}
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
            {{}, {}, {{0.01f, 0.02f, 0.03f}}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::f32,
            {{}, {}, {{0.01f, 0.02f, 0.03f}}},
            element::f32,
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
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
            {{}, {}, {{0.01f, 0.1f, 1.f}, ov::element::f32, {1, 3}}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::f32,
            {{}, {}, {{0.01f, 0.1f, 1.f}, ov::element::f32, {1, 3}}},
            element::f32,
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
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
            {{}, {}, {{0.01f, 0.1f, 1.f}, ov::element::f32, {1, 3}}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::f32,
            {{}, {}, {{0.01f, 0.1f, 1.f}, ov::element::f32, {1, 3}}},
            element::f32,
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        }
    },
    // 1) Multiply + 2) Add
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {{128}, element::f32},
            element::f32,
            {{}, {}, {0.01f}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::f32,
            {{}, {}, {}},
            element::f32,
            element::f32,
            {256ul, {}, {-128.f}, {127.f}, {0.f}, {2.55f}}
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
            {{}, {-128}, {0.01f}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::f32,
            {{}, {}, {}},
            element::f32,
            element::f32,
            {256ul, {}, {-128.f}, {127.f}, {0.f}, {2.55f}}
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
            {{element::f32}, {-128}, {0.01f}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::u8,
            {{}, {}, {}},
            element::u8,
            element::f32,
            {256ul, {}, {-128.f}, {127.f}, {0.f}, {2.55f}}
        }
    },
    // 1) Convert + Subtract + Multiply 2) Add
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {{127}, element::f32},
            element::f32,
            {{element::f32}, {-128}, {0.01f}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::f32,
            {{}, {}, {}},
            element::f32,
            element::f32,
            {256ul, {}, {-255.f}, {0.f}, {0.f}, {2.55f}}
        }
    },
    // Dynamic shape
    {
        {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {{127}, element::f32},
            element::f32,
            {{element::f32}, {-128}, {0.01f}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::f32,
            {{}, {}, {}},
            element::f32,
            element::f32,
            {256ul, {}, {-255.f}, {0.f}, {0.f}, {2.55f}}
        }
    },
    // Dynamic rank
    {
        PartialShape::dynamic(),
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {{127}, element::f32},
            element::f32,
            {{element::f32}, {-128}, {0.01f}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::f32,
            {{}, {}, {}},
            element::f32,
            element::f32,
            {256ul, {}, {-255.f}, {0.f}, {0.f}, {2.55f}}
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
            {{}, {-128}, {-0.01f}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::f32,
            {{}, {-128}, {-0.01f}},
            element::f32,
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        }
    },
    // issue #40611 for FP32
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            {},
            {},
            ov::element::i32,
            {{ov::element::f32}, {}, {}},
            ov::element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            {},
            {},
            ov::element::i32,
            {{ov::element::f32}, {}, {}},
            element::f32,
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        }
    },
    // issue #40611 for FP16
    {
        {1, 3, 16, 16},
        LayerTransformation::createParamsU8I8(),
        {
            {},
            {},
            ov::element::i32,
            {{ov::element::f16}, {}, {}},
            ov::element::f16,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            {},
            {},
            ov::element::i32,
            {{ov::element::f16}, {}, {}},
            element::f16,
            element::f16,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
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
            {{element::f32}, {{-128, -128, -128}}, {{0.01f, 0.f, 0.01f}}},
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::u8,
            {{element::f32}, {{-128, -128, -128}}, {{0.01f, 0.f, 0.01f}}},
            element::f32,
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        }
    },
    // non per-channel operation
    {
        {1, 3, 4},
        LayerTransformation::createParamsU8I8(),
        {
            element::f32,
            {},
            element::u8,
            {
                {element::f32},
                {{-128, -128, -128, -128}, ov::element::f32, {1, 1, 4}},
                {{0.01f, 0.02f, 0.03f, 0.04f}, ov::element::f32, {1, 1, 4}}
            },
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        },
        {
            element::f32,
            {},
            element::u8,
            {
                {element::f32},
                {{-128, -128, -128, -128}, ov::element::f32, {1, 1, 4}},
                {{0.01f, 0.02f, 0.03f, 0.04f}, ov::element::f32, {1, 1, 4}}
            },
            element::f32,
            element::f32,
            {256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f}}
        }
    },
};
// clang-format on

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         FuseDequantizeToFakeQuantizeTransformation,
                         ::testing::ValuesIn(testValues),
                         FuseDequantizeToFakeQuantizeTransformation::getTestCaseName);

}  // namespace
